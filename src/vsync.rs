use std::{
    collections::HashMap,
    sync::{
        Arc, LazyLock, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use gpui::{FrameRequestSource, PlatformFrameSignal};
use gpui_util::ResultExt;
use parking_lot::RwLock;
use smallvec::SmallVec;
use windows::Win32::{
    Graphics::{
        Dwm::{DWM_TIMING_INFO, DwmFlush, DwmGetCompositionTimingInfo},
        Dxgi::{IDXGIAdapter1, IDXGIFactory6, IDXGIOutput},
        Gdi::{
            HMONITOR, MONITOR_DEFAULTTONEAREST, MonitorFromWindow, RDW_INVALIDATE, RedrawWindow,
        },
    },
    System::Performance::QueryPerformanceFrequency,
    UI::WindowsAndMessaging::{IsIconic, IsWindowVisible},
};

use crate::{DirectXDevices, HWND, TrackedWindow};

static QPC_TICKS_PER_SECOND: LazyLock<u64> = LazyLock::new(|| {
    let mut frequency = 0;
    // On systems that run Windows XP or later, the function will always succeed and
    // will thus never return zero.
    unsafe { QueryPerformanceFrequency(&mut frequency).unwrap() };
    frequency as u64
});

const VSYNC_INTERVAL_THRESHOLD: Duration = Duration::from_millis(1);
const DEFAULT_VSYNC_INTERVAL: Duration = Duration::from_micros(16_666); // ~60Hz

/// Blocks until the next vblank of the paced output, or the DWM composition
/// clock for outputs whose vblank is not available.
pub(crate) enum PacingWaiter {
    Output {
        output: IDXGIOutput,
        interval: Duration,
    },
    DwmFlush {
        interval: Duration,
    },
}

impl PacingWaiter {
    pub(crate) fn from_group(group: &RateGroup) -> Self {
        match &group.output {
            Some(output) => Self::Output {
                output: output.clone(),
                interval: group.interval,
            },
            None => Self::DwmFlush {
                interval: group.interval,
            },
        }
    }

    pub(crate) fn wait_for_vsync(&self) -> FrameRequestSource {
        let vsync_start = Instant::now();
        let interval = match self {
            Self::Output { interval, .. } | Self::DwmFlush { interval } => *interval,
        };
        let wait_succeeded = match self {
            Self::Output { output, .. } => unsafe { output.WaitForVBlank().is_ok() },
            Self::DwmFlush { .. } => unsafe { DwmFlush().is_ok() },
        };
        wait_outcome(wait_succeeded, vsync_start.elapsed(), interval)
    }
}

/// WaitForVBlank and DwmFlush return very early instead of waiting until
/// vblank when the monitor goes to sleep or is unplugged (nothing to present
/// due to desktop occlusion). We use 1ms as a threshold for the duration of
/// the wait functions and fallback to Sleep() if it returns before that. This
/// could happen during normal operation for the first call after the vsync
/// thread becomes non-idle, but it shouldn't happen often.
fn wait_outcome(wait_succeeded: bool, elapsed: Duration, interval: Duration) -> FrameRequestSource {
    if !wait_succeeded || elapsed < VSYNC_INTERVAL_THRESHOLD {
        log::trace!("vblank wait took less time than expected; falling back to sleep");
        std::thread::sleep(interval);
        FrameRequestSource::LocalSchedule
    } else {
        FrameRequestSource::NativeCallback
    }
}

/// A set of monitors that share one refresh rate, paced together by one
/// waiter thread waiting on the representative output's vblank.
pub(crate) struct RateGroup {
    /// `None` on the fallback group, which is paced by `DwmFlush` and owns
    /// the windows on monitors with no known output or refresh rate.
    pub(crate) rate_hz: Option<u32>,
    /// The sleep fallback for waits that fail or return early.
    pub(crate) interval: Duration,
    /// The output whose vblank paces the group; `None` on the fallback group.
    pub(crate) output: Option<IDXGIOutput>,
    /// The monitor the representative output belongs to.
    pub(crate) representative: Option<HMONITOR>,
    /// The monitors assigned to this group.
    pub(crate) monitors: SmallVec<[HMONITOR; 4]>,
}

// SAFETY: `HMONITOR` is an opaque handle that is safe to reference from any
// thread; the group only compares handles and hands them to windowing APIs,
// never dereferences them.
unsafe impl Send for RateGroup {}
unsafe impl Sync for RateGroup {}

impl RateGroup {
    fn summary(&self) -> String {
        match self.rate_hz {
            Some(rate_hz) => format!("{rate_hz}Hz x{} monitor(s)", self.monitors.len()),
            None => format!("fallback x{} monitor(s)", self.monitors.len()),
        }
    }
}

/// Groups the connected monitors by their current refresh rate. Monitors
/// whose rate is unknown or that have no DXGI output are paced together on
/// the DWM composition clock, preserving the previous single-thread cadence
/// for them.
pub(crate) fn compute_rate_groups(devices: &DirectXDevices) -> Vec<RateGroup> {
    let outputs = enumerate_outputs(&devices.dxgi_factory);
    let mut grouped: HashMap<u32, RateGroup> = HashMap::new();
    let mut fallback = RateGroup {
        rate_hz: None,
        interval: get_dwm_interval()
            .context("Failed to get DWM interval")
            .log_err()
            .unwrap_or(DEFAULT_VSYNC_INTERVAL),
        output: None,
        representative: None,
        monitors: SmallVec::new(),
    };

    for monitor in crate::display::available_monitors() {
        let rate_hz = crate::display::refresh_hz_for_monitor(monitor);
        let output = outputs
            .iter()
            .find(|(output_monitor, _)| *output_monitor == monitor)
            .map(|(_, output)| output.clone());
        match (rate_hz, output) {
            (Some(rate_hz), Some(output)) => {
                let group = grouped.entry(rate_hz).or_insert_with(|| RateGroup {
                    rate_hz: Some(rate_hz),
                    interval: gpui::refresh_interval_from_hz(f64::from(rate_hz))
                        .unwrap_or(DEFAULT_VSYNC_INTERVAL),
                    output: Some(output.clone()),
                    representative: Some(monitor),
                    monitors: SmallVec::new(),
                });
                group.monitors.push(monitor);
            }
            _ => {
                fallback.monitors.push(monitor);
            }
        }
    }

    let mut groups: Vec<RateGroup> = grouped.into_values().collect();
    groups.sort_unstable_by_key(|group| group.rate_hz);
    groups.push(fallback);
    groups
}

/// Every (monitor, output) pair reachable from the DXGI factory, across all
/// adapters, so windows on outputs of secondary GPUs are paced natively too.
fn enumerate_outputs(factory: &IDXGIFactory6) -> SmallVec<[(HMONITOR, IDXGIOutput); 4]> {
    let mut outputs = SmallVec::new();
    for adapter_index in 0.. {
        let adapter: IDXGIAdapter1 = match unsafe { factory.EnumAdapters1(adapter_index) } {
            Ok(adapter) => adapter,
            Err(_) => break,
        };
        for output_index in 0.. {
            let output: IDXGIOutput = match unsafe { adapter.EnumOutputs(output_index) } {
                Ok(output) => output,
                Err(_) => break,
            };
            if let Some(desc) = unsafe { output.GetDesc() }.log_err() {
                outputs.push((desc.Monitor, output));
            }
        }
    }
    outputs
}

/// The waiter threads that pace the windows, one per [`RateGroup`].
pub(crate) struct PacingThreads {
    shared: Arc<PacingShared>,
    waiters: Vec<WaiterHandle>,
}

struct PacingShared {
    groups: RwLock<Arc<Vec<RateGroup>>>,
    all_windows: Weak<RwLock<SmallVec<[TrackedWindow; 4]>>>,
    stop: Arc<AtomicBool>,
    /// Set while a GPU device loss is being recovered from, so waiters stop
    /// invalidating windows that would draw against the dead device.
    recovering: AtomicBool,
}

struct WaiterHandle {
    stop: Arc<AtomicBool>,
    rate_hz: Option<u32>,
    representative: Option<HMONITOR>,
}

impl PacingThreads {
    pub(crate) fn start(
        devices: &DirectXDevices,
        all_windows: Weak<RwLock<SmallVec<[TrackedWindow; 4]>>>,
        stop: Arc<AtomicBool>,
    ) -> Self {
        let mut this = Self {
            shared: Arc::new(PacingShared {
                groups: RwLock::new(Arc::new(Vec::new())),
                all_windows,
                stop,
                recovering: AtomicBool::new(false),
            }),
            waiters: Vec::new(),
        };
        this.refresh(devices);
        this
    }

    /// Recompute the rate groups and reconcile the waiters with them: waiters
    /// whose group vanished or changed representative are stopped, missing
    /// groups get a new waiter.
    pub(crate) fn refresh(&mut self, devices: &DirectXDevices) {
        let groups = Arc::new(compute_rate_groups(devices));
        let summary = groups
            .iter()
            .map(RateGroup::summary)
            .collect::<Vec<_>>()
            .join(", ");
        log::info!("vsync pacing groups: {summary}");

        *self.shared.groups.write() = groups.clone();
        self.waiters.retain(|waiter| {
            let keep = groups.iter().any(|group| {
                group.rate_hz == waiter.rate_hz && group.representative == waiter.representative
            });
            if !keep {
                waiter.stop.store(true, Ordering::Release);
            }
            keep
        });
        for group in groups.iter() {
            if !self
                .waiters
                .iter()
                .any(|waiter| waiter.rate_hz == group.rate_hz)
            {
                self.spawn_waiter(group);
            }
        }
    }

    fn spawn_waiter(&mut self, group: &RateGroup) {
        let stop = Arc::new(AtomicBool::new(false));
        let shared = self.shared.clone();
        let waiter = PacingWaiter::from_group(group);
        let rate_hz = group.rate_hz;
        let name = match rate_hz {
            Some(rate_hz) => format!("VSyncWaiter{rate_hz}Hz"),
            None => "VSyncWaiterFallback".to_owned(),
        };
        std::thread::Builder::new()
            .name(name)
            .spawn({
                let stop = stop.clone();
                move || {
                    loop {
                        let signal_source = waiter.wait_for_vsync();
                        if stop.load(Ordering::Acquire) || shared.stop.load(Ordering::Acquire) {
                            break;
                        }
                        if shared.recovering.load(Ordering::Acquire) {
                            continue;
                        }
                        let signal_at = PlatformFrameSignal::capture(scheduler::Instant::now);
                        let Some(all_windows) = shared.all_windows.upgrade() else {
                            break;
                        };
                        let groups = Arc::clone(&*shared.groups.read());
                        for hwnd in all_windows.read().iter() {
                            // A window is paced by the group of the output it is
                            // currently on, following moves without re-binding;
                            // a monitor missing from every group falls back to
                            // the DWM-paced group.
                            let monitor = unsafe {
                                MonitorFromWindow(hwnd.as_raw(), MONITOR_DEFAULTTONEAREST)
                            };
                            let owner = groups
                                .iter()
                                .find(|group| group.monitors.contains(&monitor))
                                .map(|group| group.rate_hz)
                                .unwrap_or(None);
                            if owner != rate_hz {
                                continue;
                            }
                            unsafe {
                                let presents = IsWindowVisible(hwnd.as_raw()).as_bool()
                                    && !IsIconic(hwnd.as_raw()).as_bool();
                                if let Some(signal_at) = signal_at {
                                    if presents {
                                        hwnd.frame_signal.record(signal_at, signal_source);
                                    } else {
                                        // Hidden windows may not consume WM_PAINT until shown again.
                                        hwnd.frame_signal.take();
                                    }
                                }
                                // A hidden or minimized window has nothing to
                                // present; invalidating it only wakes the UI
                                // thread for a paint it cannot show. Showing a
                                // hidden window repaints via `WM_SHOWWINDOW`, and
                                // restoring a minimized one recreates and redraws
                                // the swap chain via the `WM_SIZE` path.
                                if presents {
                                    RedrawWindow(Some(hwnd.as_raw()), None, None, RDW_INVALIDATE)
                                        .ok()
                                        .log_err();
                                }
                            }
                        }
                    }
                }
            })
            .unwrap();
        self.waiters.push(WaiterHandle {
            stop,
            rate_hz,
            representative: group.representative,
        });
    }

    /// Stop invalidation while the GPU devices are being recovered; with the
    /// single vsync thread this was implicit in that thread blocking inside
    /// the recovery.
    pub(crate) fn pause(&self) {
        self.shared.recovering.store(true, Ordering::Release);
    }

    /// Resume invalidation after the GPU devices were recovered. The groups
    /// are recomputed because recovery recreated the DXGI factory.
    pub(crate) fn resume(&mut self, devices: &DirectXDevices) {
        self.refresh(devices);
        self.shared.recovering.store(false, Ordering::Release);
    }
}

fn get_dwm_interval() -> Result<Duration> {
    let mut timing_info = DWM_TIMING_INFO {
        cbSize: std::mem::size_of::<DWM_TIMING_INFO>() as u32,
        ..Default::default()
    };
    unsafe { DwmGetCompositionTimingInfo(HWND::default(), &mut timing_info) }?;
    let interval = retrieve_duration(timing_info.qpcRefreshPeriod, *QPC_TICKS_PER_SECOND);
    // Check for interval values that are impossibly low. A 29 microsecond
    // interval was seen (from a qpcRefreshPeriod of 60).
    if interval < VSYNC_INTERVAL_THRESHOLD {
        Ok(retrieve_duration(
            timing_info.rateRefresh.uiDenominator as u64,
            timing_info.rateRefresh.uiNumerator as u64,
        ))
    } else {
        Ok(interval)
    }
}

#[inline]
fn retrieve_duration(counts: u64, ticks_per_second: u64) -> Duration {
    let ticks_per_microsecond = ticks_per_second / 1_000_000;
    Duration::from_micros(counts / ticks_per_microsecond)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successful_compositor_wait_reports_native_callback() {
        assert_eq!(
            wait_outcome(true, VSYNC_INTERVAL_THRESHOLD * 2, Duration::ZERO),
            FrameRequestSource::NativeCallback
        );
    }

    #[test]
    fn failed_compositor_wait_reports_local_schedule_even_after_threshold() {
        assert_eq!(
            wait_outcome(false, VSYNC_INTERVAL_THRESHOLD * 2, Duration::ZERO),
            FrameRequestSource::LocalSchedule
        );
    }

    #[test]
    fn early_compositor_wait_reports_local_schedule() {
        assert_eq!(
            wait_outcome(true, Duration::ZERO, Duration::ZERO),
            FrameRequestSource::LocalSchedule
        );
    }
}

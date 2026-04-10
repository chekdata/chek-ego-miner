use std::fs;
use std::path::Path;

use sysinfo::{Components, Disks, System};

#[derive(Clone, Copy, Debug, Default)]
struct LinuxCpuTimes {
    total: u64,
    iowait: u64,
}

pub struct HostMetricsCollector {
    system: System,
    disks: Disks,
    components: Components,
    last_linux_cpu_times: Option<LinuxCpuTimes>,
}

impl HostMetricsCollector {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        Self {
            system,
            disks: Disks::new_with_refreshed_list(),
            components: Components::new_with_refreshed_list(),
            last_linux_cpu_times: read_linux_cpu_times(),
        }
    }

    pub fn emit(&mut self) {
        self.system.refresh_cpu_usage();
        self.system.refresh_memory();
        self.disks.refresh();
        self.components.refresh();

        metrics::gauge!("host_cpu_usage_percent")
            .set(self.system.global_cpu_info().cpu_usage() as f64);

        let total_memory = self.system.total_memory();
        let used_memory = self.system.used_memory();
        let memory_used_percent = if total_memory == 0 {
            0.0
        } else {
            (used_memory as f64 / total_memory as f64) * 100.0
        };
        metrics::gauge!("host_memory_used_percent").set(memory_used_percent);

        let (disk_used_percent, disk_available) = collect_disk_used_percent(&self.disks);
        metrics::gauge!("host_disk_used_percent_available").set(bool_to_f64(disk_available));
        if let Some(value) = disk_used_percent {
            metrics::gauge!("host_disk_used_percent").set(value);
        }

        let component_temps = self
            .components
            .iter()
            .filter_map(|component| {
                let temperature = component.temperature();
                temperature.is_finite().then_some(temperature as f64)
            })
            .collect::<Vec<_>>();
        metrics::gauge!("host_temperature_available").set(bool_to_f64(!component_temps.is_empty()));
        if !component_temps.is_empty() {
            let mean_temp = component_temps.iter().sum::<f64>() / component_temps.len() as f64;
            metrics::gauge!("host_temperature_c").set(mean_temp);
        }

        let nvme_temps = self
            .components
            .iter()
            .filter_map(|component| {
                let label = component.label().to_ascii_lowercase();
                if !label.contains("nvme") {
                    return None;
                }
                let temperature = component.temperature();
                temperature.is_finite().then_some(temperature as f64)
            })
            .collect::<Vec<_>>();
        metrics::gauge!("host_nvme_temperature_available").set(bool_to_f64(!nvme_temps.is_empty()));
        if !nvme_temps.is_empty() {
            let mean_temp = nvme_temps.iter().sum::<f64>() / nvme_temps.len() as f64;
            metrics::gauge!("host_nvme_temperature_c").set(mean_temp);
        }

        let current_linux_cpu_times = read_linux_cpu_times();
        let iowait_percent = current_linux_cpu_times
            .zip(self.last_linux_cpu_times)
            .and_then(|(current, previous)| calculate_iowait_percent(previous, current));
        metrics::gauge!("host_iowait_percent_available").set(bool_to_f64(iowait_percent.is_some()));
        if let Some(value) = iowait_percent {
            metrics::gauge!("host_iowait_percent").set(value);
        }
        self.last_linux_cpu_times = current_linux_cpu_times;
    }
}

fn collect_disk_used_percent(disks: &Disks) -> (Option<f64>, bool) {
    let mut total_space: u128 = 0;
    let mut used_space: u128 = 0;
    for disk in disks.iter() {
        let total = disk.total_space() as u128;
        if total == 0 {
            continue;
        }
        total_space += total;
        used_space += total.saturating_sub(disk.available_space() as u128);
    }
    if total_space == 0 {
        return (None, false);
    }
    (Some((used_space as f64 / total_space as f64) * 100.0), true)
}

fn calculate_iowait_percent(previous: LinuxCpuTimes, current: LinuxCpuTimes) -> Option<f64> {
    let total_delta = current.total.checked_sub(previous.total)?;
    let iowait_delta = current.iowait.checked_sub(previous.iowait)?;
    if total_delta == 0 {
        return None;
    }
    Some((iowait_delta as f64 / total_delta as f64) * 100.0)
}

fn read_linux_cpu_times() -> Option<LinuxCpuTimes> {
    let raw = fs::read_to_string("/proc/stat").ok()?;
    let line = raw.lines().find(|line| line.starts_with("cpu "))?;
    let values = line
        .split_whitespace()
        .skip(1)
        .filter_map(|value| value.parse::<u64>().ok())
        .collect::<Vec<_>>();
    if values.len() < 5 {
        return None;
    }
    Some(LinuxCpuTimes {
        total: values.iter().copied().sum(),
        iowait: values.get(4).copied().unwrap_or_default(),
    })
}

fn bool_to_f64(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}

#[allow(dead_code)]
fn read_sysfs_temperature(path: &Path) -> Option<f64> {
    let raw = fs::read_to_string(path).ok()?;
    let value = raw.trim().parse::<f64>().ok()?;
    Some(value / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::{calculate_iowait_percent, LinuxCpuTimes};

    #[test]
    fn iowait_percent_uses_delta() {
        let previous = LinuxCpuTimes {
            total: 1_000,
            iowait: 100,
        };
        let current = LinuxCpuTimes {
            total: 1_400,
            iowait: 140,
        };
        let percent = calculate_iowait_percent(previous, current).unwrap();
        assert!((percent - 10.0).abs() < f64::EPSILON);
    }
}

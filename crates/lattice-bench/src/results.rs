//! Benchmark results tracking
//!
//! Tracks benchmark results over time to detect regressions and improvements.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Results file path
pub const RESULTS_FILE: &str = "crates/lattice-bench/benchmark_results.json";

/// Single benchmark measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    /// LatticeDB time in microseconds
    pub lattice_us: f64,
    /// Neo4j time in microseconds (if available)
    pub neo4j_us: Option<f64>,
}

impl BenchmarkMeasurement {
    pub fn new(lattice_us: f64, neo4j_us: Option<f64>) -> Self {
        Self {
            lattice_us,
            neo4j_us,
        }
    }

    /// Calculate speedup ratio (Neo4j time / LatticeDB time)
    pub fn speedup(&self) -> Option<f64> {
        self.neo4j_us.map(|neo4j| neo4j / self.lattice_us)
    }
}

/// A single benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    /// Timestamp of the run
    pub timestamp: DateTime<Utc>,
    /// Git commit hash (if available)
    pub commit: Option<String>,
    /// List of optimizations applied
    pub optimizations: Vec<String>,
    /// Benchmark results keyed by benchmark name
    pub results: HashMap<String, BenchmarkMeasurement>,
}

impl BenchmarkRun {
    pub fn new(optimizations: Vec<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            commit: get_git_commit(),
            optimizations,
            results: HashMap::new(),
        }
    }

    pub fn add_result(&mut self, name: &str, measurement: BenchmarkMeasurement) {
        self.results.insert(name.to_string(), measurement);
    }
}

/// Collection of benchmark runs over time
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkHistory {
    pub runs: Vec<BenchmarkRun>,
}

impl BenchmarkHistory {
    /// Load from file, creating empty if not exists
    pub fn load() -> Self {
        Self::load_from(RESULTS_FILE)
    }

    pub fn load_from<P: AsRef<Path>>(path: P) -> Self {
        match fs::read_to_string(path) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Save to file
    pub fn save(&self) -> std::io::Result<()> {
        self.save_to(RESULTS_FILE)
    }

    pub fn save_to<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)
    }

    /// Add a new run
    pub fn add_run(&mut self, run: BenchmarkRun) {
        self.runs.push(run);
    }

    /// Get the most recent run
    pub fn latest(&self) -> Option<&BenchmarkRun> {
        self.runs.last()
    }

    /// Get the previous run (second to last)
    pub fn previous(&self) -> Option<&BenchmarkRun> {
        if self.runs.len() >= 2 {
            Some(&self.runs[self.runs.len() - 2])
        } else {
            None
        }
    }

    /// Compare latest run to previous and detect regressions
    pub fn detect_regressions(&self, threshold: f64) -> Vec<RegressionReport> {
        let latest = match self.latest() {
            Some(r) => r,
            None => return vec![],
        };

        let previous = match self.previous() {
            Some(r) => r,
            None => return vec![],
        };

        let mut regressions = vec![];

        for (name, latest_measurement) in &latest.results {
            if let Some(prev_measurement) = previous.results.get(name) {
                let change = latest_measurement.lattice_us / prev_measurement.lattice_us;
                if change > 1.0 + threshold {
                    regressions.push(RegressionReport {
                        benchmark: name.clone(),
                        previous_us: prev_measurement.lattice_us,
                        current_us: latest_measurement.lattice_us,
                        regression_percent: (change - 1.0) * 100.0,
                    });
                }
            }
        }

        regressions
    }

    /// Print a summary comparing latest to previous
    pub fn print_summary(&self) {
        let latest = match self.latest() {
            Some(r) => r,
            None => {
                println!("No benchmark runs recorded.");
                return;
            }
        };

        println!("\n=== Benchmark Results ===");
        println!("Timestamp: {}", latest.timestamp);
        if let Some(commit) = &latest.commit {
            println!("Commit: {}", commit);
        }
        println!("Optimizations: {:?}", latest.optimizations);
        println!();

        // Table header
        println!(
            "{:<35} {:>12} {:>12} {:>10} {:>10}",
            "Benchmark", "LatticeDB", "Neo4j", "Speedup", "Change"
        );
        println!("{}", "-".repeat(80));

        let previous = self.previous();

        let mut benchmarks: Vec<_> = latest.results.keys().collect();
        benchmarks.sort();

        for name in benchmarks {
            let measurement = &latest.results[name];
            let neo4j_str = measurement
                .neo4j_us
                .map(|us| format!("{:.1}µs", us))
                .unwrap_or_else(|| "-".to_string());

            let speedup_str = measurement
                .speedup()
                .map(|s| format!("{:.1}x", s))
                .unwrap_or_else(|| "-".to_string());

            let change_str = if let Some(prev) = previous {
                if let Some(prev_measurement) = prev.results.get(name) {
                    let change = measurement.lattice_us / prev_measurement.lattice_us;
                    if change < 0.95 {
                        format!("\x1b[32m{:.1}%\x1b[0m", (1.0 - change) * 100.0)
                    // Green for improvement
                    } else if change > 1.05 {
                        format!("\x1b[31m+{:.1}%\x1b[0m", (change - 1.0) * 100.0)
                    // Red for regression
                    } else {
                        "~".to_string()
                    }
                } else {
                    "new".to_string()
                }
            } else {
                "-".to_string()
            };

            println!(
                "{:<35} {:>12} {:>12} {:>10} {:>10}",
                name,
                format!("{:.1}µs", measurement.lattice_us),
                neo4j_str,
                speedup_str,
                change_str
            );
        }

        // Print regressions
        let regressions = self.detect_regressions(0.05); // 5% threshold
        if !regressions.is_empty() {
            println!("\n\x1b[31m⚠️  Regressions detected:\x1b[0m");
            for reg in regressions {
                println!(
                    "  - {}: {:.1}µs → {:.1}µs ({:+.1}%)",
                    reg.benchmark, reg.previous_us, reg.current_us, reg.regression_percent
                );
            }
        }
    }
}

/// Report of a performance regression
#[derive(Debug)]
pub struct RegressionReport {
    pub benchmark: String,
    pub previous_us: f64,
    pub current_us: f64,
    pub regression_percent: f64,
}

/// Get the current git commit hash
fn get_git_commit() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_measurement() {
        let m = BenchmarkMeasurement::new(100.0, Some(500.0));
        assert_eq!(m.speedup(), Some(5.0));
    }

    #[test]
    fn test_history_regression_detection() {
        let mut history = BenchmarkHistory::default();

        // First run (baseline)
        let mut run1 = BenchmarkRun::new(vec!["baseline".to_string()]);
        run1.add_result("test", BenchmarkMeasurement::new(100.0, None));
        history.add_run(run1);

        // Second run (regression)
        let mut run2 = BenchmarkRun::new(vec!["opt1".to_string()]);
        run2.add_result("test", BenchmarkMeasurement::new(150.0, None)); // 50% slower
        history.add_run(run2);

        let regressions = history.detect_regressions(0.05);
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].benchmark, "test");
    }
}

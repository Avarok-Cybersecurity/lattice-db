#!/usr/bin/env python3
"""
LatticeDB vs Neo4j Benchmark Visualization

Usage:
    cargo bench -p lattice-bench -- order_by 2>&1 | tee benchmark_output.txt
    python plot_results.py benchmark_output.txt
    # OR with sample data:
    python plot_results.py --sample
"""

import re
import sys
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Installing plotly...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


def parse_benchmark_output(content: str) -> dict:
    """Parse criterion benchmark output to extract timing data."""
    results = {"LatticeDB": {}, "Neo4j": {}}

    # Pattern: order_by/LatticeDB/1000  time:   [910.21 µs 915.42 µs 920.83 µs]
    # Also matches: time:   [5.0821 ms 5.1234 ms 5.1647 ms]
    pattern = r"order_by/(LatticeDB|Neo4j)/(\d+)\s+time:\s+\[[\d.]+ [µm]s ([\d.]+) ([µm]s)"

    for match in re.finditer(pattern, content):
        db, size, time_val, unit = match.groups()
        time_us = float(time_val)
        if unit == "ms":
            time_us *= 1000  # Convert ms to µs
        results[db][int(size)] = time_us

    return results


def create_visualization(results: dict, output_path: str = "benchmark_results.html"):
    """Create an interactive plotly visualization."""

    # Extract data
    sizes = sorted(set(results["LatticeDB"].keys()) | set(results["Neo4j"].keys()))
    lattice_times = [results["LatticeDB"].get(s, None) for s in sizes]
    neo4j_times = [results["Neo4j"].get(s, None) for s in sizes]

    # Calculate per-row times
    lattice_per_row = [t/s if t else None for t, s in zip(lattice_times, sizes)]
    neo4j_per_row = [t/s if t else None for t, s in zip(neo4j_times, sizes)]

    # Calculate speedup (positive = LatticeDB faster)
    speedups = []
    for lt, nt in zip(lattice_times, neo4j_times):
        if lt and nt:
            speedups.append(nt / lt)  # >1 means LatticeDB faster
        else:
            speedups.append(None)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Absolute Time (log scale)',
            'Time per Row',
            'Speedup (LatticeDB vs Neo4j)',
            'Scaling Analysis'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Color scheme
    lattice_color = '#3366CC'
    neo4j_color = '#DC3912'

    # Plot 1: Absolute time (log scale)
    fig.add_trace(
        go.Scatter(
            x=sizes, y=lattice_times,
            name='LatticeDB',
            mode='lines+markers',
            line=dict(color=lattice_color, width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sizes, y=neo4j_times,
            name='Neo4j',
            mode='lines+markers',
            line=dict(color=neo4j_color, width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    fig.update_yaxes(type="log", title_text="Time (µs)", row=1, col=1)
    fig.update_xaxes(title_text="Dataset Size (N)", row=1, col=1)

    # Plot 2: Per-row time
    fig.add_trace(
        go.Scatter(
            x=sizes, y=lattice_per_row,
            name='LatticeDB/row',
            mode='lines+markers',
            line=dict(color=lattice_color, width=2, dash='dot'),
            marker=dict(size=8),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=sizes, y=neo4j_per_row,
            name='Neo4j/row',
            mode='lines+markers',
            line=dict(color=neo4j_color, width=2, dash='dot'),
            marker=dict(size=8),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.update_yaxes(title_text="Time per Row (µs)", row=1, col=2)
    fig.update_xaxes(title_text="Dataset Size (N)", row=1, col=2)

    # Plot 3: Speedup
    speedup_colors = ['green' if s and s >= 1 else 'red' for s in speedups]
    fig.add_trace(
        go.Bar(
            x=sizes, y=speedups,
            name='Speedup',
            marker_color=speedup_colors,
            text=[f'{s:.2f}x' if s else 'N/A' for s in speedups],
            textposition='auto',
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_yaxes(title_text="Speedup (>1 = LatticeDB faster)", row=2, col=1)
    fig.update_xaxes(title_text="Dataset Size (N)", row=2, col=1)

    # Plot 4: Scaling analysis (O(n) vs O(n log n))
    if lattice_times[0] and neo4j_times[0]:
        base_n = sizes[0]
        base_lattice = lattice_times[0]
        base_neo4j = neo4j_times[0]

        # Expected O(n log n) scaling
        import math
        expected_nlogn = [base_lattice * (n * math.log2(n)) / (base_n * math.log2(base_n)) for n in sizes]
        expected_linear = [base_lattice * n / base_n for n in sizes]

        fig.add_trace(
            go.Scatter(
                x=sizes, y=lattice_times,
                name='LatticeDB (actual)',
                mode='lines+markers',
                line=dict(color=lattice_color, width=2),
                marker=dict(size=8),
                showlegend=False
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=sizes, y=expected_nlogn,
                name='O(n log n)',
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=True
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=sizes, y=expected_linear,
                name='O(n)',
                mode='lines',
                line=dict(color='lightgray', width=1, dash='dot'),
                showlegend=True
            ),
            row=2, col=2
        )

    fig.update_yaxes(type="log", title_text="Time (µs)", row=2, col=2)
    fig.update_xaxes(title_text="Dataset Size (N)", row=2, col=2)

    # Update layout
    fig.update_layout(
        title={
            'text': 'LatticeDB vs Neo4j: ORDER BY Performance Analysis',
            'font': {'size': 20}
        },
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Save and show
    fig.write_html(output_path)
    print(f"Visualization saved to: {output_path}")

    # Print summary
    print("\n=== Benchmark Summary ===")
    print(f"{'Size':<10} {'LatticeDB':<15} {'Neo4j':<15} {'Speedup':<10}")
    print("-" * 50)
    for i, size in enumerate(sizes):
        lt = lattice_times[i]
        nt = neo4j_times[i]
        sp = speedups[i]
        lt_str = f"{lt:.1f}µs" if lt else "N/A"
        nt_str = f"{nt:.1f}µs" if nt else "N/A"
        sp_str = f"{sp:.2f}x" if sp else "N/A"
        winner = "✓ LatticeDB" if sp and sp > 1 else "✗ Neo4j" if sp else ""
        print(f"{size:<10} {lt_str:<15} {nt_str:<15} {sp_str:<10} {winner}")

    return fig


def sample_data():
    """Return sample benchmark data for testing."""
    return {
        "LatticeDB": {
            100: 131,
            500: 474,
            1000: 910,
            5000: 5080,
            10000: 10500,
        },
        "Neo4j": {
            100: 740,
            500: 813,
            1000: 871,
            5000: 1420,
            10000: 2500,
        }
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--sample":
        print("Using sample data...")
        results = sample_data()
    else:
        input_file = Path(sys.argv[1])
        if not input_file.exists():
            print(f"Error: File not found: {input_file}")
            sys.exit(1)

        content = input_file.read_text()
        results = parse_benchmark_output(content)

        if not results["LatticeDB"] and not results["Neo4j"]:
            print("Warning: No benchmark data found in file. Using sample data...")
            results = sample_data()

    output_path = "benchmark_results.html"
    fig = create_visualization(results, output_path)

    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f"file://{Path(output_path).absolute()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LatticeDB vs Neo4j Benchmark Analysis

Usage:
    cargo bench -p lattice-bench -- order_by 2>&1 | tee benchmark_output.txt
    python3 analyze_results.py benchmark_output.txt
"""

import re
import sys
from pathlib import Path


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


def create_html_visualization(results: dict, output_path: str = "benchmark_results.html"):
    """Create a simple HTML visualization without plotly."""
    sizes = sorted(set(results["LatticeDB"].keys()) | set(results["Neo4j"].keys()))

    # Generate data for charts
    rows = []
    for size in sizes:
        lt = results["LatticeDB"].get(size)
        nt = results["Neo4j"].get(size)
        speedup = nt / lt if lt and nt else None
        per_row_lt = lt / size if lt else None
        per_row_nt = nt / size if nt else None

        rows.append({
            "size": size,
            "lattice": lt,
            "neo4j": nt,
            "speedup": speedup,
            "per_row_lt": per_row_lt,
            "per_row_nt": per_row_nt
        })

    # Generate simple HTML with embedded chart
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LatticeDB vs Neo4j Benchmark Results</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 900px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px 16px; text-align: right; border-bottom: 1px solid #eee; }}
        th {{ background: #333; color: white; }}
        tr:hover {{ background: #f9f9f9; }}
        .faster {{ color: #22c55e; font-weight: bold; }}
        .slower {{ color: #ef4444; font-weight: bold; }}
        .bar-container {{ display: flex; align-items: center; gap: 8px; }}
        .bar {{ height: 20px; border-radius: 3px; }}
        .bar-lattice {{ background: #3366CC; }}
        .bar-neo4j {{ background: #DC3912; }}
        .legend {{ display: flex; gap: 20px; margin: 20px 0; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 3px; }}
        .summary {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 900px; }}
        .chart-container {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 900px; }}
    </style>
</head>
<body>
    <h1>LatticeDB vs Neo4j: ORDER BY Performance</h1>

    <div class="legend">
        <div class="legend-item"><div class="legend-color bar-lattice"></div>LatticeDB</div>
        <div class="legend-item"><div class="legend-color bar-neo4j"></div>Neo4j</div>
    </div>

    <div class="summary">
        <h3>Analysis Summary</h3>
        <ul>
            <li><strong>Small datasets (N &lt; 1000):</strong> LatticeDB wins due to lower overhead</li>
            <li><strong>Crossover point:</strong> Around N=1000, performance is roughly equal</li>
            <li><strong>Large datasets (N &gt; 1000):</strong> Neo4j wins due to parallel processing</li>
            <li><strong>Root cause:</strong> LatticeDB uses Rc&lt;str&gt; (not Send), preventing parallel sort</li>
        </ul>
        <h4>Options to Beat Neo4j at Scale:</h4>
        <ol>
            <li><strong>Arc&lt;str&gt; wrapper:</strong> Conditional compile Rc vs Arc, enabling parallel CypherValue sort</li>
            <li><strong>Index-based parallel sort:</strong> Extract (i64, index) pairs (Send), sort those in parallel, then reorder</li>
            <li><strong>Hybrid approach:</strong> Use parallel for numeric-only sorts, sequential for complex types</li>
        </ol>
    </div>

    <table>
        <thead>
            <tr>
                <th>Size (N)</th>
                <th>LatticeDB</th>
                <th>Neo4j</th>
                <th>Speedup</th>
                <th>Winner</th>
                <th>Per-Row (LatticeDB)</th>
                <th>Per-Row (Neo4j)</th>
            </tr>
        </thead>
        <tbody>
"""

    for row in rows:
        lt_str = f"{row['lattice']:.0f} µs" if row['lattice'] else "N/A"
        nt_str = f"{row['neo4j']:.0f} µs" if row['neo4j'] else "N/A"

        if row['speedup']:
            if row['speedup'] >= 1:
                speedup_str = f"{row['speedup']:.2f}x"
                winner_class = "faster"
                winner = "LatticeDB"
            else:
                speedup_str = f"{1/row['speedup']:.2f}x"
                winner_class = "slower"
                winner = "Neo4j"
        else:
            speedup_str = "N/A"
            winner_class = ""
            winner = "N/A"

        per_row_lt = f"{row['per_row_lt']:.2f} µs" if row['per_row_lt'] else "N/A"
        per_row_nt = f"{row['per_row_nt']:.2f} µs" if row['per_row_nt'] else "N/A"

        html += f"""            <tr>
                <td>{row['size']:,}</td>
                <td>{lt_str}</td>
                <td>{nt_str}</td>
                <td class="{winner_class}">{speedup_str}</td>
                <td class="{winner_class}">{winner}</td>
                <td>{per_row_lt}</td>
                <td>{per_row_nt}</td>
            </tr>
"""

    html += """        </tbody>
    </table>

    <div class="chart-container">
        <h3>Visual Comparison</h3>
"""

    # Add bar chart
    max_time = max(max(r['lattice'] or 0 for r in rows), max(r['neo4j'] or 0 for r in rows))

    for row in rows:
        lt_width = (row['lattice'] / max_time * 400) if row['lattice'] else 0
        nt_width = (row['neo4j'] / max_time * 400) if row['neo4j'] else 0
        lt_label = f"{row['lattice']:.0f}" if row['lattice'] else "N/A"
        nt_label = f"{row['neo4j']:.0f}" if row['neo4j'] else "N/A"

        html += f"""        <div style="margin: 10px 0;">
            <strong>N={row['size']:,}</strong>
            <div class="bar-container">
                <div class="bar bar-lattice" style="width: {lt_width}px"></div>
                <span>{lt_label} µs</span>
            </div>
            <div class="bar-container">
                <div class="bar bar-neo4j" style="width: {nt_width}px"></div>
                <span>{nt_label} µs</span>
            </div>
        </div>
"""

    html += """    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    content = input_file.read_text()
    results = parse_benchmark_output(content)

    if not results["LatticeDB"] and not results["Neo4j"]:
        print("Error: No benchmark data found in file")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 70)
    print("LatticeDB vs Neo4j: ORDER BY Benchmark Results")
    print("=" * 70)

    sizes = sorted(set(results["LatticeDB"].keys()) | set(results["Neo4j"].keys()))

    print(f"\n{'Size':<10} {'LatticeDB':<15} {'Neo4j':<15} {'Speedup':<12} {'Winner':<15}")
    print("-" * 67)

    for size in sizes:
        lt = results["LatticeDB"].get(size)
        nt = results["Neo4j"].get(size)

        lt_str = f"{lt:.0f} µs" if lt and lt < 1000 else f"{lt/1000:.2f} ms" if lt else "N/A"
        nt_str = f"{nt:.0f} µs" if nt and nt < 1000 else f"{nt/1000:.2f} ms" if nt else "N/A"

        if lt and nt:
            speedup = nt / lt
            if speedup >= 1:
                speedup_str = f"{speedup:.2f}x"
                winner = "✓ LatticeDB"
            else:
                speedup_str = f"{1/speedup:.2f}x"
                winner = "✗ Neo4j"
        else:
            speedup_str = "N/A"
            winner = "N/A"

        print(f"{size:<10} {lt_str:<15} {nt_str:<15} {speedup_str:<12} {winner:<15}")

    print("\n" + "-" * 67)
    print("Per-row analysis:")
    print(f"\n{'Size':<10} {'LatticeDB/row':<15} {'Neo4j/row':<15} {'Scaling':<20}")
    print("-" * 60)

    for size in sizes:
        lt = results["LatticeDB"].get(size)
        nt = results["Neo4j"].get(size)

        lt_per_row = lt / size if lt else None
        nt_per_row = nt / size if nt else None

        lt_str = f"{lt_per_row:.3f} µs" if lt_per_row else "N/A"
        nt_str = f"{nt_per_row:.3f} µs" if nt_per_row else "N/A"

        if lt_per_row and nt_per_row:
            # Compare to first row's per-row time
            scaling = "Linear" if lt_per_row < 2 else "Super-linear"
        else:
            scaling = "N/A"

        print(f"{size:<10} {lt_str:<15} {nt_str:<15} {scaling:<20}")

    # Generate HTML report
    output_path = create_html_visualization(results)
    print(f"\n✓ HTML report generated: {Path(output_path).absolute()}")

    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)
    print("""
Neo4j benefits from parallel processing at larger N:
- Per-row cost DECREASES as N increases (parallel overhead amortized)

LatticeDB has linear per-row cost:
- Sequential sort means O(n log n) total, O(log n) per row

To beat Neo4j at scale:
1. Switch Rc<str> to Arc<str> (enables parallel CypherValue sort)
2. Use index-based parallel sort (extract i64+index, sort in parallel, reorder)
3. Hybrid: parallel for numeric sorts, sequential for complex types
""")


if __name__ == "__main__":
    main()

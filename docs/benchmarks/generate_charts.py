#!/usr/bin/env python3
"""
Generate benchmark charts for LatticeDB README.

Requirements:
    pip install plotly kaleido pandas

Usage:
    python generate_charts.py
"""

import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Load benchmark data
with open('benchmark_results.json', 'r') as f:
    data = json.load(f)

vector_results = data['vector_benchmarks']['results']

# Color scheme
LATTICE_COLOR = '#4F46E5'  # Indigo
QDRANT_COLOR = '#10B981'   # Emerald
NEO4J_COLOR = '#F59E0B'    # Amber

# Chart 1: Vector Operations Bar Chart (LatticeDB vs Qdrant)
def create_vector_comparison_chart():
    operations = [r['operation'].capitalize() for r in vector_results]
    lattice_times = [r['lattice_us'] for r in vector_results]
    qdrant_times = [r['qdrant_us'] for r in vector_results]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='LatticeDB',
        x=operations,
        y=lattice_times,
        marker_color=LATTICE_COLOR,
        text=[f'{t:.1f}µs' for t in lattice_times],
        textposition='outside',
    ))

    fig.add_trace(go.Bar(
        name='Qdrant',
        x=operations,
        y=qdrant_times,
        marker_color=QDRANT_COLOR,
        text=[f'{t:.1f}µs' for t in qdrant_times],
        textposition='outside',
    ))

    fig.update_layout(
        title={
            'text': 'Vector Operations: LatticeDB vs Qdrant',
            'font': {'size': 20, 'color': '#1F2937'},
            'x': 0.5,
        },
        xaxis_title='Operation',
        yaxis_title='Latency (µs)',
        yaxis_type='log',
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        template='plotly_white',
        font={'family': 'Inter, system-ui, sans-serif'},
        width=800,
        height=500,
        margin=dict(t=100, b=80),
    )

    # Add annotation for the winner
    fig.add_annotation(
        text="<b>LatticeDB wins 3 of 4 operations</b><br>141x faster upsert | 44x faster retrieve | 4.6x faster scroll",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=12, color='#4B5563'),
    )

    return fig


# Chart 2: Speedup comparison (horizontal bar)
def create_speedup_chart():
    operations = [r['operation'].capitalize() for r in vector_results]
    speedups = [r['speedup'] for r in vector_results]

    # Color based on who wins
    colors = [LATTICE_COLOR if s > 1 else QDRANT_COLOR for s in speedups]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=speedups,
        y=operations,
        orientation='h',
        marker_color=colors,
        text=[f'{s:.1f}x' if s >= 1 else f'{1/s:.1f}x slower' for s in speedups],
        textposition='outside',
    ))

    # Add vertical line at 1x
    fig.add_vline(x=1, line_dash='dash', line_color='#9CA3AF')

    fig.update_layout(
        title={
            'text': 'LatticeDB Performance vs Qdrant',
            'font': {'size': 20, 'color': '#1F2937'},
            'x': 0.5,
        },
        xaxis_title='Speedup Factor (higher = LatticeDB faster)',
        yaxis_title='',
        xaxis_type='log',
        template='plotly_white',
        font={'family': 'Inter, system-ui, sans-serif'},
        width=700,
        height=400,
        margin=dict(l=100, r=100, t=80, b=80),
    )

    fig.add_annotation(
        text="Values > 1x: LatticeDB faster | Values < 1x: Qdrant faster",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=11, color='#6B7280'),
    )

    return fig


# Chart 3: Simple summary card-style visualization
def create_summary_chart():
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Upsert', 'Search (k=10)', 'Retrieve', 'Scroll'),
        vertical_spacing=0.25,
        horizontal_spacing=0.15,
    )

    for i, result in enumerate(vector_results):
        row = i // 2 + 1
        col = i % 2 + 1

        lattice_time = result['lattice_us']
        qdrant_time = result['qdrant_us']

        fig.add_trace(
            go.Bar(
                x=['LatticeDB', 'Qdrant'],
                y=[lattice_time, qdrant_time],
                marker_color=[LATTICE_COLOR, QDRANT_COLOR],
                text=[f'{lattice_time:.1f}µs', f'{qdrant_time:.1f}µs'],
                textposition='outside',
                showlegend=False,
            ),
            row=row, col=col
        )

    fig.update_layout(
        title={
            'text': 'Vector Operation Latency Comparison (1000 points, 128D)',
            'font': {'size': 18, 'color': '#1F2937'},
            'x': 0.5,
        },
        template='plotly_white',
        font={'family': 'Inter, system-ui, sans-serif'},
        width=900,
        height=600,
        showlegend=False,
    )

    # Set log scale for all y-axes
    fig.update_yaxes(type='log', title_text='µs')

    return fig


def main():
    os.makedirs('charts', exist_ok=True)

    print("Generating vector comparison chart...")
    fig1 = create_vector_comparison_chart()
    fig1.write_image('charts/vector_comparison.svg')
    fig1.write_image('charts/vector_comparison.png', scale=2)
    print("  -> charts/vector_comparison.svg")

    print("Generating speedup chart...")
    fig2 = create_speedup_chart()
    fig2.write_image('charts/speedup_comparison.svg')
    fig2.write_image('charts/speedup_comparison.png', scale=2)
    print("  -> charts/speedup_comparison.svg")

    print("Generating summary chart...")
    fig3 = create_summary_chart()
    fig3.write_image('charts/summary.svg')
    fig3.write_image('charts/summary.png', scale=2)
    print("  -> charts/summary.svg")

    print("\nDone! Charts saved to docs/benchmarks/charts/")


if __name__ == '__main__':
    main()

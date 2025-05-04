import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib as mpl
from cycler import cycler


def set_paper_style():
    """Set publication-quality style for plots using IEEE format guidelines."""
    # Start with a clean slate - reset to defaults then customize
    plt.rcdefaults()

    # Use the seaborn style with white background as base
    plt.style.use('seaborn-v0_8-whitegrid')

    # Set font to Times New Roman (IEEE standard) for professional publication look
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'serif'],
        'font.size': 11,
        'font.weight': 'normal',

        # Text settings
        'text.color': 'black',
        'text.usetex': False,  # Set to True if you have LaTeX installed

        # Axes settings
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.labelsize': 8,
        'axes.labelweight': 'bold',
        'axes.linewidth': 0.8,
        'axes.edgecolor': 'black',
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.bottom': True,
        'axes.spines.left': True,

        # Tick settings
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,

        # Legend settings
        'legend.fontsize': 6,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.borderpad': 0.4,

        # Figure settings
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',

        # Grid settings
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.axisbelow': True,

        # Image saving settings
        'savefig.dpi': 600,  # Higher DPI for publications
        'savefig.format': 'pdf',  # Default save format for publications
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.transparent': False,
    })

    # IEEE-inspired professional color scheme (colorblind-friendly)
    colors = [
        '#0072BD',  # Deep blue
        '#D95319',  # Burnt orange
        '#009E73',  # Forest green
        '#CC0000',  # Crimson red
        '#7E2F8E',  # Royal purple
        '#EDB120',  # Golden yellow
        '#4DBEEE',  # Sky blue
        '#A2142F',  # Burgundy
    ]
    plt.rcParams['axes.prop_cycle'] = cycler('color', colors)


def plot_qformer_losses(results_dir, output_dir):
    """Create a 2x4 grid of QFormer component losses."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find QFormer metrics files
    clip_file = results_dir / "qformer_clip_metrics_run0.csv"
    bert_file = results_dir / "qformer_bert_metrics_run0.csv"

    if not clip_file.exists() or not bert_file.exists():
        print(f"Missing QFormer metrics files: clip={clip_file.exists()}, bert={bert_file.exists()}")
        return

    # Create figure with 2x4 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)

    try:
        # Read data
        clip_df = pd.read_csv(clip_file)
        bert_df = pd.read_csv(bert_file)

        # Define components to plot
        components = ['loss_itc', 'loss_itm', 'loss_igt', 'loss_answer']
        component_titles = ['ITC Loss', 'ITM Loss', 'IGT Loss', 'Answer Loss']

        # Colors for train and validation
        train_color = '#1f77b4'  # blue
        val_color = '#ff7f0e'  # orange

        # Top row: BERT losses
        for i, (component, title) in enumerate(zip(components, component_titles)):
            ax = axes[0, i]

            # Plot training loss
            train_col = f'train_{component}'
            ax.plot(bert_df['epoch'], bert_df[train_col],
                    label='Train',
                    color=train_color,
                    linewidth=1.5)

            # Plot validation loss
            val_col = f'val_{component}'
            ax.plot(bert_df['epoch'], bert_df[val_col],
                    label='Validation',
                    color=val_color,
                    linewidth=1.5)

            # Format subplot
            ax.set_title(f'QFormer+BERT: {title}', fontsize=10)
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel('Loss', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Adjust y-axis limits based on data
            if component == 'loss_itc':
                ax.set_ylim(0, 5.5)
            elif component == 'loss_itm':
                ax.set_ylim(0.56, 0.68)
            elif component == 'loss_igt':
                ax.set_ylim(0, 5)
            elif component == 'loss_answer':
                ax.set_ylim(0, 1.1)

        # Bottom row: CLIP losses
        for i, (component, title) in enumerate(zip(components, component_titles)):
            ax = axes[1, i]

            # Plot training loss
            train_col = f'train_{component}'
            ax.plot(clip_df['epoch'], clip_df[train_col],
                    label='Train',
                    color=train_color,
                    linewidth=1.5)

            # Plot validation loss
            val_col = f'val_{component}'
            ax.plot(clip_df['epoch'], clip_df[val_col],
                    label='Validation',
                    color=val_color,
                    linewidth=1.5)

            # Format subplot
            ax.set_title(f'QFormer+CLIP: {title}', fontsize=10)
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel('Loss', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Adjust y-axis limits based on data
            if component == 'loss_itc':
                ax.set_ylim(4.7, 5.0)
            elif component == 'loss_itm':
                ax.set_ylim(0.5, 0.8)
            elif component == 'loss_igt':
                ax.set_ylim(0, 5.5)
            elif component == 'loss_answer':
                ax.set_ylim(0, 2.0)

    except Exception as e:
        print(f"Error creating QFormer component plots: {e}")

    # Add overall title to the figure
    fig.suptitle('QFormer Component Losses: Train and Validation', fontsize=14, fontweight='bold')

    # Save the figure
    plt.savefig(output_dir / 'qformer_component_losses.pdf', dpi=600, bbox_inches='tight',
                pad_inches=0.1, figsize=(9, 6))
    plt.savefig(output_dir / 'qformer_component_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("QFormer component losses plot generated successfully")


def main():
    """Parse arguments and generate the plot."""
    parser = argparse.ArgumentParser(description="Generate QFormer component losses plot")

    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Directory containing results files (default: '../results')")

    parser.add_argument("--output_dir", type=str, default="../plots",
                        help="Directory to save plots (default: '../plots')")

    args = parser.parse_args()

    # Ensure directories exist
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Apply publication-quality style
    set_paper_style()

    print(f"Reading results from: {results_dir}")
    print(f"Saving plots to: {output_dir}")

    # Generate the plot
    plot_qformer_losses(results_dir, output_dir)

    print(f"\nPlot generated successfully!")


if __name__ == "__main__":
    main()
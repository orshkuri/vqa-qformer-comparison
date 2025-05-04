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
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.linewidth': 0.8,
        'axes.edgecolor': 'black',
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.bottom': True,
        'axes.spines.left': True,

        # Tick settings
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,

        # Legend settings
        'legend.fontsize': 9,
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


def get_colors():
    """Define consistent colors for each model-encoder combination using IEEE color scheme."""
    return {
        'qformer_clip': '#0072BD',  # Deep blue
        'qformer_bert': '#D95319',  # Burnt orange
        'cross_attention_clip': '#009E73',  # Forest green
        'cross_attention_bert': '#CC0000',  # Crimson red
        'concat_clip': '#7E2F8E',  # Royal purple
        'concat_bert': '#A2142F'  # Burgundy
    }


def plot_training_validation_accuracy(results_dir, output_dir):
    """Create a 2x3 grid of training and validation accuracy curves."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set colors for consistency
    colors = get_colors()

    # Create figure with 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    axes = axes.flatten()

    # Models and encoders for the accuracy plots
    models = ["qformer", "cross_attention", "concat"]
    encoders = ["clip", "bert"]

    # Counter for tracking which subplot we're on
    plot_idx = 0

    # Loop through models and encoders to create accuracy plots
    for model in models:
        for encoder in encoders:
            file_path = results_dir / f"{model}_{encoder}_metrics_run0.csv"

            if file_path.exists() and plot_idx < len(axes):
                try:
                    df = pd.read_csv(file_path)
                    ax = axes[plot_idx]

                    # Get color for this model + encoder
                    color = colors[f"{model}_{encoder}"]

                    # Plot training accuracy
                    ax.plot(df['epoch'], df['train_accuracy'],
                            label='Train Accuracy',
                            color=color,
                            linewidth=1.5)

                    # Plot validation accuracy with dashed line
                    ax.plot(df['epoch'], df['val_accuracy'],
                            label='Val Accuracy',
                            color=color,
                            linewidth=1.5,
                            linestyle='--')

                    # Format subplot
                    model_name = model.replace("_", " ").title()
                    encoder_name = encoder.upper()
                    ax.set_title(f'{model_name} + {encoder_name}', fontsize=12)
                    ax.set_xlabel('Epoch', fontsize=10)
                    ax.set_ylabel('Accuracy', fontsize=10)
                    ax.set_ylim(0.4, 1.0)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                    plot_idx += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Add overall title to the figure
    fig.suptitle('Training and Validation Accuracy Curves', fontsize=14, fontweight='bold')

    # Save the figure
    plt.savefig(output_dir / 'training_validation_accuracy.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'training_validation_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Training and validation accuracy curves generated successfully")


def main():
    """Parse arguments and generate the plot."""
    parser = argparse.ArgumentParser(description="Generate training and validation accuracy curves")

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
    plot_training_validation_accuracy(results_dir, output_dir)

    print(f"\nPlot generated successfully!")


if __name__ == "__main__":
    main()
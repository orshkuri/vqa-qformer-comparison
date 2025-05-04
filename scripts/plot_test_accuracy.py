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
        'legend.fontsize': 10,
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


def plot_test_accuracy(results_dir, output_dir):
    """Create a bar plot of test accuracy by model and encoder."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set colors for consistency
    colors = get_colors()

    # Create figure
    plt.figure(figsize=(8, 6))

    # Process summary files for test metrics
    summary_files = list(results_dir.glob("*_summary_results.csv"))
    model_encoder_results = {}

    for file_path in summary_files:
        try:
            # Extract model and encoder from filename
            filename = file_path.stem
            parts = filename.split('_')

            if len(parts) >= 3:
                if parts[0] == 'cross' and parts[1] == 'attention':
                    model = 'cross_attention'
                    encoder = parts[2]
                else:
                    model = parts[0]
                    encoder = parts[1]

                # Read data
                df = pd.read_csv(file_path)

                # Calculate mean and standard error for accuracy
                accuracy_mean = df['test_accuracy'].mean()
                accuracy_se = df['test_accuracy'].std() / np.sqrt(len(df))

                model_encoder_results[(model, encoder)] = {
                    'accuracy_mean': accuracy_mean,
                    'accuracy_se': accuracy_se,
                    'n_runs': len(df)
                }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not model_encoder_results:
        print("No valid results found for plotting")
        return

    # Convert to DataFrame for easier plotting
    plot_data = []
    for (model, encoder), metrics in model_encoder_results.items():
        model_label = model.replace('_', ' ').title()
        plot_data.append({
            'Model': f"{model_label}\n({encoder.upper()})",
            'model_encoder': f"{model}_{encoder}",
            'Accuracy': metrics['accuracy_mean'],
            'Accuracy_SE': metrics['accuracy_se'],
            'n_runs': metrics['n_runs']
        })

    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values('model_encoder')

    # Plot test accuracy with error bars
    x_positions = np.arange(len(plot_df))
    bar_colors = [colors[model_encoder] for model_encoder in plot_df['model_encoder']]

    bars = plt.bar(x_positions, plot_df['Accuracy'], yerr=plot_df['Accuracy_SE'],
                   capsize=6, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1,
                   error_kw={'ecolor': 'black', 'lw': 1.5, 'capthick': 1.5, 'capsize': 5})

    # Add value labels on top of bars
    for bar, acc in zip(bars, plot_df['Accuracy']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Format the plot
    plt.title('Test Accuracy by Model and Encoder', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xticks(x_positions, plot_df['Model'], fontsize=9)
    plt.ylim(0, min(1.0, plot_df['Accuracy'].max() + plot_df['Accuracy_SE'].max() + 0.05))
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Add note about standard error
    plt.text(0.02, 0.02, f'Error bars show standard error (n={plot_df["n_runs"].iloc[0]} runs)',
             transform=plt.gca().transAxes, fontsize=8, verticalalignment='bottom')

    # Add border to plot
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('gray')

    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / 'test_accuracy_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'test_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Test accuracy comparison plot generated successfully")


def main():
    """Parse arguments and generate the plot."""
    parser = argparse.ArgumentParser(description="Generate test accuracy bar plot")

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
    plot_test_accuracy(results_dir, output_dir)

    print(f"\nPlot generated successfully!")


if __name__ == "__main__":
    main()
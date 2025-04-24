#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
import re
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality plots from training results")

    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Directory containing result CSV files (default: '../results')")
    parser.add_argument("--output_dir", type=str, default="../plots",
                        help="Directory to save plots (default: '../plots')")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved figures (default: 300)")
    parser.add_argument("--style", type=str, default="seaborn-v0_8-paper",
                        choices=["seaborn-v0_8-paper", "seaborn-v0_8-whitegrid", "ggplot", "bmh"],
                        help="Plot style to use (default: 'seaborn-v0_8-paper')")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set the plotting style
    plt.style.use(args.style)

    # Define model colors matching your examples
    model_colors = {
        'qformer': '#0000FF',  # Pure Blue
        'cross_attention': '#FF0000',  # Pure Red
        'concat': '#008000'  # Pure Green
    }

    # Find all metrics files and test results
    metrics_files = list(results_dir.glob("*_metrics.csv"))
    test_results_files = list(results_dir.glob("*_test_results.csv"))

    if not metrics_files:
        print(f"No metrics files found in {results_dir}")
        return

    print(f"Found {len(metrics_files)} metrics files")

    # Process each metrics file
    all_training_data = []

    for metrics_file in metrics_files:
        # Extract model and encoder information from filename
        filename = metrics_file.name
        match = re.match(r"(\w+)_(\w+)_metrics\.csv", filename)
        if not match:
            print(f"Skipping file with unexpected naming pattern: {metrics_file}")
            continue

        model_name = match.group(1)
        encoder_type = match.group(2)

        # Read the metrics
        try:
            df = pd.read_csv(metrics_file)
            df['model'] = model_name
            df['encoder'] = encoder_type
            all_training_data.append(df)
            print(f"Processed {metrics_file}")
        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")

    if not all_training_data:
        print("No data could be loaded from metrics files")
        return

    # Combine all training data
    combined_df = pd.concat(all_training_data, ignore_index=True)

    # Process test results
    test_results = []
    for result_file in test_results_files:
        filename = result_file.name
        match = re.match(r"(\w+)_(\w+)_test_results\.csv", filename)
        if not match:
            print(f"Skipping test file with unexpected naming pattern: {result_file}")
            continue

        model_name = match.group(1)
        encoder_type = match.group(2)

        try:
            df = pd.read_csv(result_file)
            result_dict = {'model': model_name, 'encoder': encoder_type}
            for _, row in df.iterrows():
                result_dict[row['metric']] = row['value']
            test_results.append(result_dict)
            print(f"Processed test results from {result_file}")
        except Exception as e:
            print(f"Error processing {result_file}: {e}")

    test_df = pd.DataFrame(test_results) if test_results else None

    # Create only the specific plots shown in your examples
    plot_training_accuracy(combined_df, model_colors, output_dir, args.dpi)

    if test_df is not None:
        plot_test_answer_accuracy(test_df, model_colors, output_dir, args.dpi)

    # QFormer loss components plot for both CLIP and BERT
    qformer_df = combined_df[combined_df['model'] == 'qformer']
    plot_qformer_loss_components(qformer_df, output_dir, args.dpi)

    print(f"All plots saved to {output_dir}")


def plot_training_accuracy(df, model_colors, output_dir, dpi):
    """Plot the training and validation accuracy across models exactly as shown in example."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define markers for each model
    markers = {'qformer': 'o', 'cross_attention': 's', 'concat': '^'}

    # Define line styles
    train_style = {'linestyle': '-', 'linewidth': 2}
    val_style = {'linestyle': ':', 'linewidth': 2}

    # Plot each model-encoder combination
    for (model, encoder), group in df.groupby(['model', 'encoder']):
        # Get base color by model
        color = model_colors[model]

        # Adjust color shade for BERT vs CLIP (lighter for BERT)
        if encoder == 'bert':
            r, g, b = mcolors.to_rgb(color)
            color = mcolors.to_hex((r * 0.7 + 0.3, g * 0.7 + 0.3, b * 0.7 + 0.3))

        # Create proper label format
        model_label = model.replace('_', ' ').title()
        encoder_label = encoder.upper()

        # Plot training line
        ax.plot(group['epoch'], group['train_accuracy'],
                marker=markers[model], markersize=7,
                color=color, label=f"{model_label} + {encoder_label} (Train)",
                **train_style)

        # Plot validation line
        ax.plot(group['epoch'], group['val_accuracy'],
                marker=markers[model], markersize=7,
                color=color, label=f"{model_label} + {encoder_label} (Val)",
                **val_style)

    # Set up axes
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Training and Validation Accuracy Across Models', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.7)

    # Position legend outside the plot to the right
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=10)

    # Set integer ticks for epochs
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the plot
    fig.savefig(output_dir / 'accuracy_comparison.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'accuracy_comparison.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_qformer_loss_components(df, output_dir, dpi):
    """Plot the QFormer loss components for both CLIP and BERT encoders."""
    # Identify available loss components
    loss_cols = [col for col in df.columns if 'loss_' in col
                 and col not in ['train_loss', 'val_loss']]

    # Get unique loss types
    loss_types = sorted(list(set([col.split('_')[-1] for col in loss_cols
                                  if 'train_loss_' in col or 'val_loss_' in col])))

    if not loss_types:
        print("No loss components found for QFormer model")
        return

    # Create figure for each encoder
    for encoder in df['encoder'].unique():
        encoder_df = df[df['encoder'] == encoder]

        if encoder_df.empty:
            continue

        # Create a 1×4 grid of subplots (based on your example)
        fig, axes = plt.subplots(1, len(loss_types), figsize=(16, 4))

        # Set colors for training and validation
        train_color = 'blue'
        val_color = 'red'

        for i, loss_type in enumerate(loss_types):
            ax = axes[i]
            train_col = f'train_loss_{loss_type}'
            val_col = f'val_loss_{loss_type}'

            if train_col in encoder_df.columns and val_col in encoder_df.columns:
                # Plot training line
                ax.plot(encoder_df['epoch'], encoder_df[train_col],
                        marker='o', markersize=4,
                        color=train_color, label='Training')

                # Plot validation line
                ax.plot(encoder_df['epoch'], encoder_df[val_col],
                        marker='s', markersize=4,
                        color=val_color, linestyle='--', label='Validation')

                # Set title and labels
                ax.set_title(f'{encoder.upper()} - {loss_type.title()} Loss', fontsize=12)
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel(f'{loss_type.title()} Loss', fontsize=10)

                # Add grid
                ax.grid(True, alpha=0.7)

                # Add legend
                ax.legend(fontsize=8)

        # Set the overall title
        fig.suptitle('Qformer Loss Components', fontsize=16, fontweight='bold')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot
        fig.savefig(output_dir / f'qformer_{encoder}_loss_components.png', dpi=dpi, bbox_inches='tight')
        fig.savefig(output_dir / f'qformer_{encoder}_loss_components.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig)


def plot_test_answer_accuracy(test_df, model_colors, output_dir, dpi):
    """Plot the test answer accuracy as horizontal bars as shown in example."""
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a new column for model+encoder
    test_df['model_encoder'] = test_df.apply(
        lambda row: f"{row['model'].replace('_', ' ').title()} + {row['encoder'].upper()}",
        axis=1
    )

    # Find the test_answer_accuracy column or equivalent
    accuracy_col = next((col for col in test_df.columns if 'test_answer_accuracy' in col
                         or ('test' in col and 'accuracy' in col)), None)

    if not accuracy_col:
        print("No test answer accuracy column found")
        return

    # Sort by model and encoder for consistent ordering
    sorted_df = test_df.sort_values(by=['model', 'encoder'], ascending=[False, False])

    # Create bars with colors based on model
    bars = []
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        color = model_colors[row['model']]
        # Adjust color for BERT
        if row['encoder'] == 'bert':
            r, g, b = mcolors.to_rgb(color)
            color = mcolors.to_hex((r * 0.7 + 0.3, g * 0.7 + 0.3, b * 0.7 + 0.3))

        bar = ax.barh(i, row[accuracy_col], color=color)
        bars.append(bar)

        # Add text label for value
        ax.text(row[accuracy_col] + 0.005, i, f"{row[accuracy_col]:.3f}",
                va='center', fontsize=12, fontweight='bold', color='black')

    # Set y-ticks to model+encoder labels
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['model_encoder'])

    # Set labels and title
    ax.set_xlabel('Test Answer Accuracy Epoch', fontsize=12)
    ax.set_ylabel('Model + Encoder', fontsize=12)
    ax.set_title('Answer Accuracy Epoch Comparison', fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')

    # Add a simple legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='QFormer'),
        Patch(facecolor='red', label='Cross Attention'),
        Patch(facecolor='green', label='Concat'),
        Patch(facecolor='gray', label='CLIP'),
        Patch(facecolor='gray', alpha=0.5, label='BERT'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    fig.savefig(output_dir / 'test_answer_accuracy_horizontal.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'test_answer_accuracy_horizontal.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()

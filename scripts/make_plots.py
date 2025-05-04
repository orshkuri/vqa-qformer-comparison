import argparse
import subprocess
from pathlib import Path

def main():
    """Parse arguments and run all plotting scripts."""
    parser = argparse.ArgumentParser(description="Generate all plots")

    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Directory containing results files (default: '../results')")

    parser.add_argument("--output_dir", type=str, default="../plots",
                        help="Directory to save plots (default: '../plots')")

    args = parser.parse_args()

    # Ensure directories exist
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Reading results from: {results_dir}")
    print(f"Saving plots to: {output_dir}")

    # List of plotting scripts to run
    scripts = [
        "plot_accuracy_curves.py",
        "plot_qformer_losses.py",
        "plot_test_accuracy.py",
        "plot_test_auc.py"
    ]

    # Run each script with the provided arguments
    for script in scripts:
        print(f"\nRunning {script}...")
        cmd = [
            "python",
            script,
            f"--results_dir={results_dir}",
            f"--output_dir={output_dir}"
        ]
        subprocess.run(cmd, check=True)
        print(f"Completed {script}")

    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()
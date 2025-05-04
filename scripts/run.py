import subprocess
import argparse
from pathlib import Path
import time
import csv
import pandas as pd
import numpy as np
import threading


def parse_boolean(value):
    """Helper function to parse boolean values from string."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_single_experiment(model, use_clip, encoder_name, gpu_id, run_id, seed, save_learning_curves,
                          results_dir, models_dir, config_dir, data_dir, log_dir, all_runs_file):
    """Run a single experiment."""
    print(f"\nStarting {model} + {encoder_name} run {run_id} (seed {seed}) on GPU {gpu_id}")

    # Update the tracking file to show this run is starting
    with open(all_runs_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([model, encoder_name, run_id, seed, gpu_id, 'RUNNING', '', ''])

    # Construct the command
    cmd = [
        "python", "../trainers/trainer.py",
        "--model_name", model,
        "--use_clip_for_text", str(use_clip),
        "--gpu_device", str(gpu_id),
        "--results_dir", str(results_dir),
        "--models_dir", str(models_dir),
        "--config_dir", str(config_dir),
        "--data_dir", str(data_dir),
        "--seed", str(seed),
        "--run_id", str(run_id),
        "--save_learning_curves", str(save_learning_curves)
    ]

    # Create a log file for this run
    log_file = log_dir / f"{model}_{encoder_name}_run{run_id}_gpu{gpu_id}.log"

    # Run the experiment and wait for it to complete
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        process.wait()  # Wait for this run to complete

    # Update status in the tracking file
    status = "COMPLETED" if process.returncode == 0 else f"FAILED ({process.returncode})"

    # Read the existing CSV and update
    tracking_df = pd.read_csv(all_runs_file)
    mask = (tracking_df['model'] == model) & \
           (tracking_df['encoder'] == encoder_name) & \
           (tracking_df['run_id'] == run_id)

    tracking_df.loc[mask, 'status'] = status

    # Try to read test metrics if the run was successful
    if process.returncode == 0:
        summary_file = results_dir / f"{model}_{encoder_name}_summary_results.csv"
        if summary_file.exists():
            try:
                summary_df = pd.read_csv(summary_file)
                run_data = summary_df[summary_df['run_id'] == run_id]
                if not run_data.empty:
                    tracking_df.loc[mask, 'test_accuracy'] = run_data['test_accuracy'].values[0]
                    tracking_df.loc[mask, 'test_auc'] = run_data['test_auc'].values[0]
            except Exception as e:
                print(f"Error reading summary results: {e}")

    # Write back the updated dataframe
    tracking_df.to_csv(all_runs_file, index=False)

    # Print status message
    if process.returncode == 0:
        print(f"✓ {model} + {encoder_name} run {run_id} completed successfully")
    else:
        print(f"✗ {model} + {encoder_name} run {run_id} failed with code {process.returncode}")

    return process.returncode


def run_experiment_group(model, use_clip, encoder_name, gpu_id, num_runs, base_seed,
                         save_learning_curves, results_dir, models_dir, config_dir,
                         data_dir, log_dir, all_runs_file):
    """Run all experiments for a single model+encoder combination sequentially."""
    print(f"\nStarting experiment group: {model} + {encoder_name} on GPU {gpu_id}")

    # Run each experiment in the group sequentially
    for run_id in range(num_runs):
        seed = base_seed + run_id
        save_curves = save_learning_curves and run_id == 0  # Only save curves for the first run

        return_code = run_single_experiment(
            model, use_clip, encoder_name, gpu_id, run_id, seed, save_curves,
            results_dir, models_dir, config_dir, data_dir, log_dir, all_runs_file
        )

        # Continue with next run regardless of return code

    print(f"\nCompleted all runs for {model} + {encoder_name} on GPU {gpu_id}")


def run_gpu_experiments_sequentially(groups, num_runs, base_seed, save_learning_curves,
                                     results_dir, models_dir, config_dir, data_dir,
                                     log_dir, all_runs_file):
    """Run all experiments assigned to a specific GPU sequentially."""
    for group in groups:
        run_experiment_group(
            group['model'],
            group['use_clip'],
            group['encoder_name'],
            group['gpu_id'],
            num_runs,
            base_seed,
            save_learning_curves,
            results_dir,
            models_dir,
            config_dir,
            data_dir,
            log_dir,
            all_runs_file
        )


def main():
    parser = argparse.ArgumentParser(description="Run multiple training jobs with multiple seeds")

    # Basic parameters
    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Directory to save results (default: '../results')")
    parser.add_argument("--models_dir", type=str, default="../saved_models",
                        help="Directory to save models (default: '../saved_models')")
    parser.add_argument("--config_dir", type=str, default="../configs",
                        help="Directory containing model configs (default: '../configs')")
    parser.add_argument("--data_dir", type=str, default="../data/vqa",
                        help="Directory containing dataset files (default: '../data/vqa')")

    # GPU configuration
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated list of GPU IDs to use (default: '0')")

    # Experiment selection
    parser.add_argument("--models", type=str, default="all",
                        choices=["all", "qformer", "cross_attention", "concat"],
                        help="Which models to run (default: 'all')")
    parser.add_argument("--encoders", type=str, default="all",
                        choices=["all", "clip", "bert"],
                        help="Which text encoders to use (default: 'all')")

    # Multiple runs configuration
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs per experiment (default: 5)")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for random number generation (default: 42)")
    parser.add_argument("--save_learning_curves", type=parse_boolean, default=True,
                        help="Whether to save learning curves for the first run (default: True)")
    parser.add_argument("--generate_plots", type=parse_boolean, default=True,
                        help="Whether to generate plots after completion (default: True)")

    args = parser.parse_args()

    # Create directories
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(exist_ok=True, parents=True)

    # Create a directory for the plots
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Create log directory
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True, parents=True)

    # Determine which models to run
    if args.models == "all":
        models = ["qformer", "cross_attention", "concat"]
    else:
        models = [args.models]

    # Determine which encoders to use
    if args.encoders == "all":
        encoders = [True, False]  # True for CLIP, False for BERT
        encoder_names = ["clip", "bert"]
    elif args.encoders == "clip":
        encoders = [True]
        encoder_names = ["clip"]
    else:
        encoders = [False]
        encoder_names = ["bert"]

    # Get available GPUs
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(',')]

    # Create experiment groups - each group is (model, encoder) with a GPU assignment
    experiment_groups = []
    gpu_assignment_idx = 0

    # For each model+encoder combination, create all runs as a single group
    for model in models:
        for encoder, encoder_name in zip(encoders, encoder_names):
            # Assign the entire combination to a single GPU
            gpu_id = gpu_ids[gpu_assignment_idx % len(gpu_ids)]
            experiment_groups.append({
                "model": model,
                "use_clip": encoder,
                "encoder_name": encoder_name,
                "gpu_id": gpu_id
            })
            gpu_assignment_idx += 1

    total_experiments = len(experiment_groups) * args.num_runs
    print(f"Planning to run {total_experiments} experiments across {len(gpu_ids)} GPUs")
    print(f"Each model-encoder combination will be run {args.num_runs} times with different seeds")

    # Display planned experiments
    for group in experiment_groups:
        print(f"\n{group['model']} + {group['encoder_name']}: {args.num_runs} runs on GPU {group['gpu_id']}")

    # Check if user wants to proceed
    proceed = input("\nProceed with these experiments? (y/n): ")
    if proceed.lower() != 'y':
        print("Aborting.")
        return

    # Create a CSV file to track all runs
    all_runs_file = results_dir / "all_runs_summary.csv"
    with open(all_runs_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'encoder', 'run_id', 'seed', 'gpu', 'status', 'test_accuracy', 'test_auc'])

    # Define the missing variables
    config_dir = args.config_dir
    data_dir = args.data_dir

    # Group experiments by GPU so that each GPU runs sequentially
    gpu_groups = {}
    for group in experiment_groups:
        gpu_id = group['gpu_id']
        if gpu_id not in gpu_groups:
            gpu_groups[gpu_id] = []
        gpu_groups[gpu_id].append(group)

    # Launch one thread per GPU, each handling its assigned experiments sequentially
    threads = []
    for gpu_id, groups in gpu_groups.items():
        thread = threading.Thread(
            target=run_gpu_experiments_sequentially,
            args=(
                groups,
                args.num_runs,
                args.base_seed,
                args.save_learning_curves,
                results_dir,
                models_dir,
                config_dir,
                data_dir,
                log_dir,
                all_runs_file
            )
        )
        thread.start()
        threads.append(thread)
        time.sleep(1)

    # Wait for all threads to complete
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nInterrupted! Experiments will continue running in the background.")
        print("Press Ctrl+C again to force exit.")
        time.sleep(2)

    print("\nAll experiments completed!")

    # Generate summary statistics and plots if requested
    if args.generate_plots:
        print("\nGenerating summary statistics and plots...")
        generate_summary_stats_and_plots(results_dir, plots_dir, models, encoder_names)

    print("All done!")


def generate_summary_stats_and_plots(results_dir, plots_dir, models, encoder_names):
    """
    Generate summary statistics and plots for the completed experiments.
    """
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Create a summary table of all experiments
    all_results = []
    summary_stats = []

    for model in models:
        for encoder in encoder_names:
            # Load the summary results file for this model-encoder pair
            summary_file = results_dir / f"{model}_{encoder}_summary_results.csv"

            if summary_file.exists():
                try:
                    df = pd.read_csv(summary_file)
                    all_results.append(df)

                    # Calculate statistics
                    mean_accuracy = df['test_accuracy'].mean()
                    std_accuracy = df['test_accuracy'].std()

                    # Handle AUC which may be non-numeric
                    if 'test_auc' in df.columns:
                        auc_values = pd.to_numeric(df['test_auc'], errors='coerce').dropna()
                        if len(auc_values) > 0:
                            mean_auc = auc_values.mean()
                            std_auc = auc_values.std()
                        else:
                            mean_auc = np.nan
                            std_auc = np.nan
                    else:
                        mean_auc = np.nan
                        std_auc = np.nan

                    summary_stats.append({
                        'model': model,
                        'encoder': encoder,
                        'accuracy_mean': mean_accuracy,
                        'accuracy_std': std_accuracy,
                        'auc_mean': mean_auc,
                        'auc_std': std_auc,
                        'num_runs': len(df)
                    })
                except Exception as e:
                    print(f"Error processing {summary_file}: {e}")

    # Save the detailed statistics
    if summary_stats:
        stats_df = pd.DataFrame(summary_stats)
        stats_df.to_csv(results_dir / "experiments_summary_statistics.csv", index=False)

        # Create a human-readable summary table
        summary_path = plots_dir / "results_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=== Experiment Results Summary ===\n\n")
            for stat in summary_stats:
                f.write(f"{stat['model']} with {stat['encoder']}:\n")
                f.write(f"  Accuracy: {stat['accuracy_mean']:.4f} ± {stat['accuracy_std']:.4f}\n")
                if not np.isnan(stat['auc_mean']):
                    f.write(f"  AUC: {stat['auc_mean']:.4f} ± {stat['auc_std']:.4f}\n")
                else:
                    f.write(f"  AUC: N/A\n")
                f.write(f"  Number of runs: {stat['num_runs']}\n\n")

        print(f"Summary saved to {summary_path}")

    # Run the plotting script automatically
    try:
        plot_script = Path(__file__).parent / "make_plots.py"
        if plot_script.exists():
            print("Running make_plots.py...")
            plot_cmd = [
                "python", str(plot_script),
                "--results_dir", str(results_dir),
                "--output_dir", str(plots_dir)
            ]
            subprocess.run(plot_cmd, check=True)
            print("Plots generated successfully")
        else:
            print(f"Plot script not found at {plot_script}")
            print("Please run make_plots.py manually to generate visualizations")
    except Exception as e:
        print(f"Error running plot script: {e}")
        print("Please run make_plots.py manually to generate visualizations")


if __name__ == "__main__":
    main()

import subprocess
import argparse
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="Run multiple training jobs")

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

    args = parser.parse_args()

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

    # Create the experiment combinations
    experiments = []
    for model in models:
        for encoder, encoder_name in zip(encoders, encoder_names):
            experiments.append({
                "model": model,
                "use_clip": encoder,
                "encoder_name": encoder_name
            })

    print(f"Planning to run {len(experiments)} experiments across {len(gpu_ids)} GPUs")
    for i, exp in enumerate(experiments):
        print(f"Experiment {i + 1}: {exp['model']} with {exp['encoder_name']} encoder")

    # Check if user wants to proceed
    proceed = input("Proceed with these experiments? (y/n): ")
    if proceed.lower() != 'y':
        print("Aborting.")
        return

    # Create log directory
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True, parents=True)

    # Launch the experiments
    processes = []
    for i, exp in enumerate(experiments):
        # Select GPU in round-robin fashion
        gpu_id = gpu_ids[i % len(gpu_ids)]

        # Construct the command
        cmd = [
            "python", "../trainers/trainer.py",
            "--model_name", exp["model"],
            "--use_clip_for_text", str(exp["use_clip"]),
            "--gpu_device", gpu_id,
            "--results_dir", args.results_dir,
            "--models_dir", args.models_dir,
            "--config_dir", args.config_dir,
            "--data_dir", args.data_dir
        ]

        # Create a log file for this experiment
        log_file = log_dir / f"{exp['model']}_{exp['encoder_name']}_gpu{gpu_id}.log"

        print(f"Starting experiment: {exp['model']} with {exp['encoder_name']} on GPU {gpu_id}")
        print(f"Log file: {log_file}")

        # Launch the process
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            processes.append((process, exp, gpu_id))

        # Wait a bit between launching processes
        time.sleep(2)

    # Monitor the processes
    print("\nAll experiments launched. Monitoring progress...")
    try:
        while processes:
            for i, (process, exp, gpu_id) in enumerate(processes[:]):
                if process.poll() is not None:  # Process has finished
                    if process.returncode == 0:
                        print(
                            f"✓ Experiment {exp['model']} with {exp['encoder_name']} on GPU {gpu_id} completed successfully")
                    else:
                        print(
                            f"✗ Experiment {exp['model']} with {exp['encoder_name']} on GPU {gpu_id} failed with code {process.returncode}")
                    processes.pop(i)

            if processes:
                time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        print("\nInterrupted! Do you want to stop all running experiments?")
        stop_all = input("Stop all experiments? (y/n): ")
        if stop_all.lower() == 'y':
            for process, exp, gpu_id in processes:
                process.terminate()
            print("All experiments terminated.")
        else:
            print("Experiments will continue running in the background.")

    print("All done!")


if __name__ == "__main__":
    main()

import torch
import os
import json
import csv
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.my_datasets.vqa_dataset import create_vqa_dataloaders
from src.my_lightning_model.qformer_lightning import QFormerLightning
from src.my_lightning_model.cross_attention_lightning import CrossAttentionLightning
from src.my_lightning_model.concat_lightning import ConcatLightning
import argparse
from sklearn.metrics import roc_auc_score
import numpy as np
import random


def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model_name: str, use_clip_for_text: bool, gpu_device: int = 2,
          results_dir: str = "../results", models_dir: str = "../saved_models",
          config_dir: str = "../configs", data_dir: str = "../data/vqa",
          seed: int = 42, run_id: int = 0, save_learning_curves: bool = False):
    print(f"Starting run {run_id} with seed {seed}")
    print(f"use_clip_for_text: {use_clip_for_text}")
    print(f"gpu_device: {gpu_device}")

    # Set the seed for reproducibility
    set_seed(seed)

    """
    Train a model (QFormer, CrossAttention, or Concat) on the VQA dataset.
    """
    # Setup device configuration
    gpu_device = gpu_device if torch.cuda.is_available() else None
    device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")

    # Load the configuration file
    config_path = os.path.join(config_dir, f"config_{model_name.lower()}.json")
    with open(config_path, "r") as f:
        hyperparams = json.load(f)

    # Add use_clip_for_text to the hyperparams
    hyperparams['use_clip_for_text'] = use_clip_for_text

    # Create results directory if it doesn't exist
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create results file paths
    encoder_type = "clip" if use_clip_for_text else "bert"
    metrics_file = results_dir / f"{model_name}_{encoder_type}_metrics_run{run_id}.csv" if save_learning_curves else None

    # Create or update summary results file for all runs
    summary_results_file = results_dir / f"{model_name}_{encoder_type}_summary_results.csv"

    # Keep track of best checkpoint path for this run
    best_checkpoint_path = results_dir / f"{model_name}_{encoder_type}_best_run{run_id}.ckpt"

    # Set paths for the dataset
    train_file = os.path.join(data_dir, "vaq2.0.TrainImages.txt")
    val_file = os.path.join(data_dir, "vaq2.0.DevImages.txt")
    test_file = os.path.join(data_dir, "vaq2.0.TestImages.txt")
    images_dir = os.path.join(data_dir, "val2014-resised")

    # Ensure all files exist
    for file_path in [train_file, val_file, test_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_vqa_dataloaders(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        images_dir=images_dir,
        batch_size=hyperparams['batch_size'],
        device=device
    )

    # Initialize model
    if model_name.lower() == "qformer":
        model = QFormerLightning(hyperparams=hyperparams, device=device)
    elif model_name.lower() == "cross_attention":
        model = CrossAttentionLightning(hyperparams=hyperparams, device=device)
    elif model_name.lower() == "concat":
        model = ConcatLightning(hyperparams=hyperparams, device=device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Custom callback to save metrics after each epoch
    class MetricsCallback(pl.Callback):
        def __init__(self, metrics_file, model_name):
            super().__init__()
            self.metrics_file = metrics_file
            self.model_name = model_name

            # Skip if not saving learning curves
            if self.metrics_file is None:
                return

            # Create headers based on model type
            if model_name.lower() == "qformer":
                headers = ['epoch', 'train_loss', 'train_accuracy', 'train_loss_itc',
                           'train_loss_igt', 'train_loss_itm', 'train_loss_answer',
                           'val_loss', 'val_accuracy', 'val_loss_itc',
                           'val_loss_igt', 'val_loss_itm', 'val_loss_answer']
            else:  # cross_attention or concat (both have similar metrics)
                headers = ['epoch', 'train_loss', 'train_accuracy', 'train_loss_answer',
                           'val_loss', 'val_accuracy', 'val_loss_answer']

            # Create the CSV file with headers
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        def on_train_epoch_end(self, trainer, pl_module):
            # Skip if not saving learning curves
            if self.metrics_file is None:
                return

            # Get current metrics
            epoch = trainer.current_epoch

            # Common metrics
            try:
                train_acc = trainer.callback_metrics.get("train_answer_accuracy_epoch", 0.0)
                val_acc = trainer.callback_metrics.get("val_answer_accuracy_epoch", 0.0)
                train_loss_answer = trainer.callback_metrics.get("train_loss_answer_epoch", 0.0)
                val_loss_answer = trainer.callback_metrics.get("val_loss_answer_epoch", 0.0)

                # Convert tensors to float values
                train_acc_val = train_acc.item() if hasattr(train_acc, 'item') else float(train_acc)
                val_acc_val = val_acc.item() if hasattr(val_acc, 'item') else float(val_acc)
                train_loss_answer_val = train_loss_answer.item() if hasattr(train_loss_answer, 'item') else float(
                    train_loss_answer)
                val_loss_answer_val = val_loss_answer.item() if hasattr(val_loss_answer, 'item') else float(
                    val_loss_answer)

                # Save to CSV based on model type
                if self.model_name.lower() == "qformer":
                    # Get QFormer specific metrics
                    train_loss_itc = trainer.callback_metrics.get("train_loss_itc_epoch", 0.0)
                    train_loss_igt = trainer.callback_metrics.get("train_loss_igt_epoch", 0.0)
                    train_loss_itm = trainer.callback_metrics.get("train_loss_itm_epoch", 0.0)
                    val_loss_itc = trainer.callback_metrics.get("val_loss_itc_epoch", 0.0)
                    val_loss_igt = trainer.callback_metrics.get("val_loss_igt_epoch", 0.0)
                    val_loss_itm = trainer.callback_metrics.get("val_loss_itm_epoch", 0.0)

                    # Convert tensors to float values
                    train_loss_itc_val = train_loss_itc.item() if hasattr(train_loss_itc, 'item') else float(
                        train_loss_itc)
                    train_loss_igt_val = train_loss_igt.item() if hasattr(train_loss_igt, 'item') else float(
                        train_loss_igt)
                    train_loss_itm_val = train_loss_itm.item() if hasattr(train_loss_itm, 'item') else float(
                        train_loss_itm)
                    val_loss_itc_val = val_loss_itc.item() if hasattr(val_loss_itc, 'item') else float(val_loss_itc)
                    val_loss_igt_val = val_loss_igt.item() if hasattr(val_loss_igt, 'item') else float(val_loss_igt)
                    val_loss_itm_val = val_loss_itm.item() if hasattr(val_loss_itm, 'item') else float(val_loss_itm)

                    # Use total_loss or sum of components if available
                    train_loss = trainer.callback_metrics.get("train_loss_epoch",
                                                              sum([train_loss_itc_val, train_loss_igt_val,
                                                                   train_loss_itm_val, train_loss_answer_val]))
                    val_loss = trainer.callback_metrics.get("val_loss_epoch",
                                                            sum([val_loss_itc_val, val_loss_igt_val, val_loss_itm_val,
                                                                 val_loss_answer_val]))

                    train_loss_val = train_loss.item() if hasattr(train_loss, 'item') else float(train_loss)
                    val_loss_val = val_loss.item() if hasattr(val_loss, 'item') else float(val_loss)

                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, train_loss_val, train_acc_val,
                                         train_loss_itc_val, train_loss_igt_val, train_loss_itm_val,
                                         train_loss_answer_val,
                                         val_loss_val, val_acc_val,
                                         val_loss_itc_val, val_loss_igt_val, val_loss_itm_val, val_loss_answer_val])
                else:  # cross_attention or concat
                    # Use answer_loss directly
                    train_loss = train_loss_answer
                    val_loss = val_loss_answer

                    train_loss_val = train_loss.item() if hasattr(train_loss, 'item') else float(train_loss)
                    val_loss_val = val_loss.item() if hasattr(val_loss, 'item') else float(val_loss)

                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, train_loss_val, train_acc_val, train_loss_answer_val,
                                         val_loss_val, val_acc_val, val_loss_answer_val])
            except Exception as e:
                print(f"Error saving metrics: {str(e)}")

    # Setup model checkpoint callback to save best model based on validation accuracy
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(results_dir),
        filename=f"{model_name}_{encoder_type}_run{run_id}_best",
        monitor="val_answer_accuracy",
        mode="max",
        save_top_k=1,
        verbose=True
    )

    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor="val_answer_accuracy",
        patience=hyperparams.get('patience', 10),
        mode="max",
        verbose=True
    )

    metrics_callback = MetricsCallback(metrics_file, model_name)

    # Setup logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name=f"{model_name.lower()}_{encoder_type}_run{run_id}",
        default_hp_metric=False
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=hyperparams.get('num_epochs', 100),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[gpu_device] if gpu_device is not None else None,
        logger=logger,
        callbacks=[early_stopping, metrics_callback, checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Load the best model weights for testing
    if os.path.exists(checkpoint_callback.best_model_path):
        print(f"Loading best model from {checkpoint_callback.best_model_path}")
        best_model = type(model).load_from_checkpoint(
            checkpoint_callback.best_model_path,
            hyperparams=hyperparams,
            device=device
        )
        best_model.to(device)
        test_results = trainer.test(best_model, dataloaders=test_dataloader)
    else:
        print("Warning: Best model checkpoint not found, using current model state")
        test_results = trainer.test(model, dataloaders=test_dataloader)

    # Collect predictions and labels for AUC calculation
    all_preds = []
    all_labels = []

    try:
        # Use the best model for evaluation if available
        eval_model = best_model if 'best_model' in locals() else model
        eval_model.to(device)
        eval_model.eval()

        with torch.no_grad():
            for batch in test_dataloader:
                # Move tensors to the same device as model
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)

                # Get model output
                output = eval_model(batch)

                # Check if we have answer logits and labels to compute AUC
                if 'answer_predictions' in output and 'answer_labels' in output:
                    preds = output['answer_predictions']
                    labels = output['answer_labels'].float()

                    preds = preds.cpu().numpy()
                    labels = labels.cpu().numpy().flatten()

                    all_preds.append(preds)
                    all_labels.append(labels)

        # Concatenate all predictions and labels
        if all_preds and all_labels:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            # Calculate AUC for binary classification
            if len(all_preds) > 0 and len(np.unique(all_labels)) > 1:
                try:
                    # Binary case
                    auc = roc_auc_score(all_labels, all_preds)
                    print(f"Test AUC: {auc:.4f}")
                except ValueError as e:
                    print(f"Could not calculate AUC: {str(e)}")
                    auc = None
            else:
                print("Could not calculate AUC - insufficient class distribution")
                auc = None
        else:
            print("No predictions or labels available for AUC calculation")
            auc = None

    except Exception as e:
        print(f"Error during prediction or AUC calculation: {str(e)}")
        auc = None

    # Make sure test_results is a list with at least one dict
    if not test_results or not isinstance(test_results, list) or len(test_results) == 0:
        test_results = [{'test_answer_accuracy': 0.0}]

    # Extract test accuracy
    test_accuracy = test_results[0]['test_answer_accuracy_epoch']
    if isinstance(test_accuracy, torch.Tensor):
        test_accuracy = test_accuracy.item()

    print(f"Test accuracy: {test_accuracy:.4f}")

    # Create or update the summary results file
    summary_exists = os.path.exists(summary_results_file)
    run_results = {
        'run_id': run_id,
        'seed': seed,
        'test_accuracy': test_accuracy,
        'test_auc': auc if auc is not None else 'N/A'
    }

    # Extract other test metrics
    for key, value in test_results[0].items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        run_results[key] = value

    # Write to summary file
    mode = 'a' if summary_exists else 'w'
    with open(summary_results_file, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=run_results.keys())
        if not summary_exists:
            writer.writeheader()
        writer.writerow(run_results)

    print(f"Run {run_id} metrics saved to {summary_results_file}")
    if metrics_file:
        print(f"Training and validation metrics saved to {metrics_file}")

    # Get the best checkpoint path from the checkpoint callback
    best_model_path = checkpoint_callback.best_model_path

    # Save the best model with a standardized name
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Return the test metrics for this run
    return {
        'model': model_name,
        'encoder': encoder_type,
        'run_id': run_id,
        'seed': seed,
        'test_accuracy': test_accuracy,
        'test_auc': auc,
        'best_model_path': best_model_path
    }


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQC model")

    parser.add_argument("--model_name", type=str, required=True,
                        choices=["qformer", "cross_attention", "concat"],
                        help="Which model to train.")

    parser.add_argument(
        "--use_clip_for_text", type=str2bool, nargs='?', const=True, default=True,
        help="Use CLIP encoder for text instead of BERT. (default: True). Pass False to use BERT."
    )

    parser.add_argument("--gpu_device", type=int, default=0,
                        help="Index of the GPU to use (e.g., 0, 1, 2, ...). Defaults to 0.")

    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Directory to save results and metrics (default: '../results')")

    parser.add_argument("--models_dir", type=str, default="../saved_models",
                        help="Directory to save model checkpoints (default: '../saved_models')")

    parser.add_argument("--config_dir", type=str, default="../configs",
                        help="Directory containing model configs (default: '../configs')")

    parser.add_argument("--data_dir", type=str, default="../data/vqa",
                        help="Directory containing dataset files (default: '../data/vqa')")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    parser.add_argument("--run_id", type=int, default=0,
                        help="Run identifier for multiple runs (default: 0)")

    parser.add_argument("--save_learning_curves", type=str2bool, nargs='?', const=True, default=False,
                        help="Whether to save learning curves (default: False)")

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        use_clip_for_text=args.use_clip_for_text,
        gpu_device=args.gpu_device,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        config_dir=args.config_dir,
        data_dir=args.data_dir,
        seed=args.seed,
        run_id=args.run_id,
        save_learning_curves=args.save_learning_curves
    )

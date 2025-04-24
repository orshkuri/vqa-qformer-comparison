import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from model.qformer_model import QFormer


class QFormerLightning(pl.LightningModule):
    def __init__(self,
                 hyperparams,
                 device):
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparameters
        self.hyperparams = hyperparams

        # Initialize Q-Former model
        self.q_former = QFormer(
            sequence_size=self.hyperparams['sequence_size'],
            qformer_hidden_size=self.hyperparams['qformer_hidden_size'],
            blocks_num=self.hyperparams['blocks_num'],
            num_heads=self.hyperparams['num_heads'],
            num_queries=self.hyperparams['num_queries'],
            dropout_rate=self.hyperparams['dropout_rate'],
            use_clip_for_text=self.hyperparams['use_clip_for_text'],
            unfreeze_layers=self.hyperparams['unfreeze_layers'],
            device=device
        )

    def forward(self, samples):
        return self.q_former(samples)

    def _common_step(self, batch, batch_idx, task):
        output = self.forward(batch)

        # Log all metrics
        self.log(f"{task}_answer_accuracy", output['answer_accuracy'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_itc", output['loss_itc'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_igt", output['loss_igt'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_itm", output['loss_itm'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_answer", output['loss_answer'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_total_loss", output['total_loss'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])

        logs = ""
        logs += f"{task}_answer_accuracy: {output['answer_accuracy'].item():.4f}, "
        logs += f"{task}_loss_itc: {output['loss_itc'].item():.4f}, "
        logs += f"{task}_loss_igt: {output['loss_igt'].item():.4f}, "
        logs += f"{task}_loss_itm: {output['loss_itm'].item():.4f}, "
        logs += f"{task}_loss_answer: {output['loss_answer'].item():.4f}, "
        logs += f"{task}_total_loss: {output['total_loss'].item():.4f}"
        self.print(logs)

        return output

    def training_step(self, batch, batch_idx):
        output = self._common_step(batch, batch_idx, "train")
        return output["total_loss"]

    def validation_step(self, batch, batch_idx):
        output = self._common_step(batch, batch_idx, "val")
        return output

    def test_step(self, batch, batch_idx):
        output = self._common_step(batch, batch_idx, "test")
        return output

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hyperparams['lr'],
            betas=self.hyperparams['betas'],
            weight_decay=self.hyperparams['weight_decay'],
            eps=self.hyperparams['eps']
        )

        return optimizer

    def save_checkpoint(self, file_path: str):
        self.trainer.save_checkpoint(file_path)

    # def load_checkpoint(self, file_path: str):
    #     self.q_former = QFormerLightning.load_from_checkpoint(file_path)

import pytorch_lightning as pl
from torch.optim import AdamW
from src.model.cross_attention_model import CrossAttentionModel


class CrossAttentionLightning(pl.LightningModule):
    def __init__(self,
                 hyperparams,
                 device):
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparameters
        self.hyperparams = hyperparams

        # Initialize CrossAttention model
        self.network = CrossAttentionModel(
            sequence_size=self.hyperparams['sequence_size'],
            hidden_size=self.hyperparams['hidden_size'],
            blocks_num=self.hyperparams['blocks_num'],
            num_heads=self.hyperparams['num_heads'],
            dropout_rate=self.hyperparams['dropout_rate'],
            use_clip_for_text=self.hyperparams['use_clip_for_text'],
            unfreeze_layers=self.hyperparams['unfreeze_layers'],
            device=device
        )

    def forward(self, samples):
        return self.network(samples)

    def _common_step(self, batch, batch_idx, task):
        output = self.forward(batch)

        self.log(f"{task}_answer_accuracy", output['answer_accuracy'], prog_bar=True, on_step=True, on_epoch=True,
                 logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_answer", output['loss_answer'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])

        logs = ""
        # Log all losses
        logs += f"{task}_answer_accuracy: {output['answer_accuracy'].item():.4f}, "
        logs += f"{task}_loss_answer: {output['loss_answer'].item():.4f}, "
        self.print(logs)

        return output

    def training_step(self, batch, batch_idx):
        output = self._common_step(batch, batch_idx, "train")
        return output["loss_answer"]

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

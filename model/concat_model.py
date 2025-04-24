import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from torchmetrics import Accuracy
from model.clip_vit import VisionEncoder


class ConcatModel(nn.Module):
    def __init__(self,
                 sequence_size,
                 hidden_size,
                 device=None,
                 use_clip_for_text=True,
                 clip_model_name="openai/clip-vit-large-patch14",
                 dropout_rate=0.3,
                 unfreeze_layers=4):
        super(ConcatModel, self).__init__()

        self.vision_dim = 1024  # Default for ViT
        self.device = device
        self.max_text_len = sequence_size
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size

        # Text Encoder selection
        self.use_clip_for_text = use_clip_for_text
        if self.use_clip_for_text:
            self._setup_clip_model(clip_model_name, unfreeze_layers)
        else:
            self._setup_bert_model(unfreeze_layers)

        self.vision_encoder = VisionEncoder(model_name=clip_model_name,
                                            device=device,
                                            unfreeze_layers=unfreeze_layers)

        # Projections to normalize embedding dimensions
        self.vision_projection = nn.Linear(self.vision_dim, hidden_size)
        self.text_projection = nn.Linear(self.text_dim, hidden_size)

        # MLP for classification from concatenated embeddings
        layers = []
        input_size = hidden_size * 2  # Concatenated dim

        # Create hidden layers
        for i in range(2):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))

        # Final prediction layer
        layers.append(nn.Linear(hidden_size, 1))

        self.cat_mlp = nn.Sequential(*layers)

        # Metrics
        self.accuracy = Accuracy(task='binary')

        # Initialize weights
        self.init_weights()

        # Move to specified device
        self.to(device)

    def _setup_clip_model(self, clip_model_name, unfreeze_clip_layers):
        """Setup the CLIP model."""
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Freeze CLIP model by default
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze the specified number of layers (if requested)
        if unfreeze_clip_layers > 0:
            self._unfreeze_clip_layers(unfreeze_clip_layers)

        self.text_dim = self.clip_model.text_model.config.hidden_size

    def _setup_bert_model(self, unfreeze_bert_layers):
        """Setup the BERT model."""
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.bert.resize_token_embeddings(len(BertTokenizer.from_pretrained("bert-base-uncased")))

        for param in self.bert.parameters():
            param.requires_grad = False

        if unfreeze_bert_layers > 0:
            self._unfreeze_bert_layers(unfreeze_bert_layers)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.text_dim = self.bert.config.hidden_size

    def init_weights(self):
        """Initialize the weights using best practices."""
        # Vision projection
        nn.init.kaiming_normal_(self.vision_projection.weight, nonlinearity='relu')
        nn.init.zeros_(self.vision_projection.bias)

        # Text projection
        nn.init.kaiming_normal_(self.text_projection.weight, nonlinearity='relu')
        nn.init.zeros_(self.text_projection.bias)

        # MLP weights
        for layer in self.cat_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu', a=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Special initialization for final layer
        final_layer = None
        for layer in reversed(self.cat_mlp):
            if isinstance(layer, nn.Linear):
                final_layer = layer
                break

        if final_layer is not None:
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
            if final_layer.bias is not None:
                final_layer.bias.data.fill_(0.1)

        print("Model weights initialized successfully")

    def encode_text(self, questions):
        """Encode text using BERT or CLIP's text encoder"""
        if self.use_clip_for_text:
            # Process text through CLIP
            question_tokens = self.clip_processor(
                text=questions,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt"
            ).to(self.device)

            # Move each tensor manually to the same device as the model
            question_tokens = {k: v.to(self.device) for k, v in question_tokens.items()}

            text_output = self.clip_model.text_model(
                input_ids=question_tokens['input_ids'],
                attention_mask=question_tokens['attention_mask'],
                return_dict=True
            )

            # Use CLS token (first token)
            text_embedding = text_output.last_hidden_state[:, 0, :]

        else:
            # Process text through BERT
            question_tokens = self.tokenizer(
                questions,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            ).to(self.device)

            # Move each tensor manually to the same device as the model
            question_tokens = {k: v.to(self.device) for k, v in question_tokens.items()}

            text_output = self.bert(
                input_ids=question_tokens["input_ids"],
                attention_mask=question_tokens["attention_mask"],
                return_dict=True,
            )

            # Use CLS token (first token)
            text_embedding = text_output.last_hidden_state[:, 0, :]

        return text_embedding

    def forward(self, samples):
        """Forward pass of the ConcatModel."""
        # Get image features from vision encoder
        image_input = samples['image_input']
        image_features = self.vision_encoder.encode(image_input)

        # Get global image feature (mean pooling)
        if len(image_features.shape) == 3:  # Shape: (batch, seq_len, hidden_dim)
            image_features = image_features.mean(dim=1)  # Global average pooling

        # Project image features
        image_features = self.vision_projection(image_features)

        # Normalize image features
        image_features = F.normalize(image_features, dim=-1)

        # Get the question text
        questions = samples['question']

        # Encode and project text features
        text_embedding = self.encode_text(questions)
        text_embedding = self.text_projection(text_embedding)

        # Normalize text embedding
        text_embedding = F.normalize(text_embedding, dim=-1)

        # Concatenate image and text features
        concat_embeddings = torch.cat([text_embedding, image_features], dim=1)

        # Pass through MLP for prediction
        answer_logits = self.cat_mlp(concat_embeddings)

        # Get the ground truth labels
        answers = samples['answer']
        dic = {'yes': 1, 'no': 0}
        answer_labels = torch.tensor([dic[answer] for answer in answers], dtype=torch.float,
                                     device=self.device).unsqueeze(1)

        # Compute loss
        loss_answer = F.binary_cross_entropy_with_logits(answer_logits, answer_labels)

        # Compute accuracy
        with torch.no_grad():
            predictions = torch.sigmoid(answer_logits)
            answer_accuracy = self.accuracy(predictions, answer_labels.int())

            # Debug info during evaluation
            if not self.training:
                mean_pred = predictions.mean().item()
                std_pred = predictions.std().item()
                max_pred = predictions.max().item()
                min_pred = predictions.min().item()
                yes_pred = (predictions >= 0.5).sum().item()
                no_pred = (predictions < 0.5).sum().item()

                print(
                    f"Predictions - Mean: {mean_pred:.3f}, Std: {std_pred:.3f}, Range: [{min_pred:.3f}, {max_pred:.3f}]")
                print(f"Predicted: {yes_pred} yes, {no_pred} no")

        return {
            'answer_accuracy': answer_accuracy,
            'loss_answer': loss_answer,
            'answer_logits': answer_logits
        }

    def _unfreeze_clip_layers(self, num_layers=4):
        """Unfreeze the last n layers of the CLIP model for fine-tuning"""
        # Freeze all parameters first
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze text encoder layers
        for i, block in enumerate(reversed(self.clip_model.text_model.encoder.layers)):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = True

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable}")

    def _unfreeze_bert_layers(self, num_layers=4):
        """Unfreeze the last n layers of the BERT model for fine-tuning"""
        # Freeze all parameters first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last n encoder layers
        for i, layer in enumerate(reversed(self.bert.encoder.layer)):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = True

        # Optionally unfreeze the pooler
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable}")

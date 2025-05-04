import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from torchmetrics import Accuracy
from src.model.blocks import MultiHeadCrossAttention
from src.model.clip_vit import VisionEncoder


class CrossAttentionModel(nn.Module):
    def __init__(self,
                 sequence_size,
                 hidden_size,
                 blocks_num,
                 num_heads,
                 device,
                 use_clip_for_text=True,
                 clip_model_name="openai/clip-vit-large-patch14",
                 dropout_rate=0.3,
                 unfreeze_layers=4):
        super(CrossAttentionModel, self).__init__()

        self.vision_dim = 1024  # Default for ViT

        self.device = device
        self.max_text_len = sequence_size
        self.dropout_rate = dropout_rate

        # Regularization flags
        # self.use_token_dropout = True
        # self.use_feature_dropout = True
        # self.use_label_smoothing = True
        # self.label_smoothing = 0.1
        # self.mixup_alpha = 0.2  # For mixup augmentation

        # Check dimension compatibility
        assert hidden_size % num_heads == 0, \
            f"Hidden size ({hidden_size}) must be divisible by number of heads ({num_heads})"

        # Text Encoder selection
        self.use_clip_for_text = use_clip_for_text
        if self.use_clip_for_text:
            self._setup_clip_model(clip_model_name, unfreeze_layers)
        else:
            self._setup_bert_model(unfreeze_layers)

        self.vision_encoder = VisionEncoder(model_name=clip_model_name,
                                            device=device,
                                            unfreeze_layers=unfreeze_layers)

        # Load CLIP model for both vision and text encoding
        # self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        # self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Projections with layer normalization for stability
        self.vision_projection = nn.Sequential(
            nn.Linear(self.vision_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate)  # Increased dropout
        )

        self.text_projection = nn.Sequential(
            nn.Linear(self.text_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate)  # Increased dropout
        )

        # Multiple cross-attention blocks for better generalization
        self.cross_attn_blocks = nn.ModuleList([
            nn.Sequential(
                MultiHeadCrossAttention(hidden_size, num_heads),
                nn.Dropout(dropout_rate)
            ) for _ in range(blocks_num)  # Use k stacked blocks
        ])

        # Layer normalization for residual connection
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Feature dropout (applied to entire feature dimensions)
        # self.feature_dropout = nn.Dropout2d(0.1)  # Spatial dropout for features

        # Projection MLP with stronger regularization
        self.proj_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),  # Norm before linear
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(128),  # Add norm between layers
            nn.Linear(128, 10),
            nn.Dropout(dropout_rate * 0.5)  # Less dropout at the end
        )

        # Global average pooling to reduce sequence dimension
        self.use_global_pooling = True  # Flag to control whether to use global pooling

        # MLP head with better capacity and regularization
        if self.use_global_pooling:
            # If using global pooling, input dimension is just 10
            self.mlp_head = nn.Sequential(
                nn.Linear(10, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 1)
            )
        else:
            # Original approach using full sequence
            self.mlp_head = nn.Sequential(
                nn.Linear(10 * sequence_size, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 1)
            )

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
        """
        Initialize the weights using best practices for attention models.
        """
        # Vision projection - handle both single Linear layer or Sequential
        if isinstance(self.vision_projection, nn.Sequential):
            nn.init.kaiming_normal_(self.vision_projection[0].weight, nonlinearity='relu')
            nn.init.zeros_(self.vision_projection[0].bias)
        elif isinstance(self.vision_projection, nn.Linear):
            nn.init.kaiming_normal_(self.vision_projection.weight, nonlinearity='relu')
            nn.init.zeros_(self.vision_projection.bias)

        # Text projection - handle both single Linear layer or Sequential
        if isinstance(self.text_projection, nn.Sequential):
            nn.init.kaiming_normal_(self.text_projection[0].weight, nonlinearity='relu')
            nn.init.zeros_(self.text_projection[0].bias)
        elif isinstance(self.text_projection, nn.Linear):
            nn.init.kaiming_normal_(self.text_projection.weight, nonlinearity='relu')
            nn.init.zeros_(self.text_projection.bias)

        # Cross attention weights
        for block in self.cross_attn_blocks:
            # Handle both Sequential blocks and direct modules
            attn_module = block[0] if isinstance(block, nn.Sequential) else block

            # Handle different attention module implementations
            if hasattr(attn_module, 'lin_q'):
                nn.init.xavier_normal_(attn_module.lin_q.weight, gain=1.0)
                nn.init.zeros_(attn_module.lin_q.bias)
                nn.init.xavier_normal_(attn_module.lin_k.weight, gain=1.0)
                nn.init.zeros_(attn_module.lin_k.bias)
                nn.init.xavier_normal_(attn_module.lin_v.weight, gain=1.0)
                nn.init.zeros_(attn_module.lin_v.bias)
            elif hasattr(attn_module, 'q_proj'):
                # Alternative attention implementation (common in transformers)
                nn.init.xavier_normal_(attn_module.q_proj.weight, gain=1.0)
                if hasattr(attn_module.q_proj, 'bias') and attn_module.q_proj.bias is not None:
                    nn.init.zeros_(attn_module.q_proj.bias)
                nn.init.xavier_normal_(attn_module.k_proj.weight, gain=1.0)
                if hasattr(attn_module.k_proj, 'bias') and attn_module.k_proj.bias is not None:
                    nn.init.zeros_(attn_module.k_proj.bias)
                nn.init.xavier_normal_(attn_module.v_proj.weight, gain=1.0)
                if hasattr(attn_module.v_proj, 'bias') and attn_module.v_proj.bias is not None:
                    nn.init.zeros_(attn_module.v_proj.bias)

            if hasattr(attn_module, 'out_proj'):
                nn.init.xavier_normal_(attn_module.out_proj.weight, gain=1.0)
                if hasattr(attn_module.out_proj, 'bias') and attn_module.out_proj.bias is not None:
                    nn.init.zeros_(attn_module.out_proj.bias)

        # MLP weights - handle both iterables and single modules
        mlp_modules = [self.proj_mlp, self.mlp_head]
        for module in mlp_modules:
            if isinstance(module, nn.Sequential) or isinstance(module, list):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu', a=0.1)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu', a=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Special initialization for final layer
        # Be flexible in how we access the final layer
        final_layer = None
        if isinstance(self.mlp_head, nn.Sequential):
            final_layer = self.mlp_head[-1]
        elif isinstance(self.mlp_head, nn.Linear):
            final_layer = self.mlp_head

        if final_layer is not None and isinstance(final_layer, nn.Linear):
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
            if final_layer.bias is not None:
                final_layer.bias.data.fill_(0.1)

        print("Model weights initialized successfully")

    # def token_dropout(self, embeddings, attention_mask=None, drop_prob=0.1):
    #     """
    #     Randomly drop tokens from the sequence to improve generalization
    #     """
    #     if not self.training or not self.use_token_dropout:
    #         return embeddings, attention_mask
    #
    #     batch_size, seq_len, dim = embeddings.shape
    #
    #     # Don't drop the CLS token (first token)
    #     keep_prob = 1.0 - drop_prob
    #     drop_mask = torch.bernoulli(
    #         torch.ones(batch_size, seq_len - 1, device=embeddings.device) * keep_prob
    #     )
    #     drop_mask = torch.cat([torch.ones(batch_size, 1, device=drop_mask.device), drop_mask], dim=1)
    #     drop_mask = drop_mask.unsqueeze(-1)
    #
    #     # Apply the dropout mask
    #     embeddings = embeddings * drop_mask
    #
    #     # Also update attention mask if provided
    #     if attention_mask is not None:
    #         # Keep CLS token
    #         drop_mask = drop_mask.squeeze(-1)
    #         new_attention_mask = attention_mask * drop_mask.long()
    #         # Ensure CLS token is always attended to
    #         new_attention_mask[:, 0] = 1
    #         return embeddings, new_attention_mask
    #
    #     return embeddings, attention_mask

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

            question_output = self.clip_model.text_model(
                input_ids=question_tokens['input_ids'],
                attention_mask=question_tokens['attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
            text_embeddings = question_output.last_hidden_state
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

            question_output = self.bert(
                input_ids=question_tokens["input_ids"],
                attention_mask=question_tokens["attention_mask"],
                return_dict=True,
            )
            text_embeddings = question_output.last_hidden_state

        return text_embeddings, question_tokens["attention_mask"]

    # def mixup_features(self, features, labels, alpha=0.2):
    #     """
    #     Apply mixup augmentation to features and labels
    #     """
    #     if not self.training or alpha <= 0:
    #         return features, labels
    #
    #     batch_size = features.size(0)
    #
    #     # Sample mixup coefficient
    #     lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(self.device)
    #
    #     # Create shuffled indices
    #     indices = torch.randperm(batch_size).to(self.device)
    #
    #     # Mix the features
    #     mixed_features = lam * features + (1 - lam) * features[indices]
    #
    #     # Mix the labels
    #     mixed_labels = lam * labels + (1 - lam) * labels[indices]
    #
    #     return mixed_features, mixed_labels

    def forward(self, samples):
        """
        Forward pass of the CrossAttentionModel with enhanced regularization.
        """
        # Training mode specific operations
        is_training = self.training

        # Get image features from CLIP ViT
        image_input = samples['image_input']
        image_features = self.vision_encoder.encode(image_input)

        # Get the question text
        questions = samples['question']

        batch_size = image_features.shape[0]

        # Apply feature dropout to image features during training
        # if is_training and self.use_feature_dropout:
        #     # Reshape for 2D dropout then reshape back
        #     # This drops entire feature channels
        #     orig_shape = image_features.shape
        #     if len(orig_shape) == 3:  # (batch, seq, dim)
        #         image_features = image_features.permute(0, 2, 1)  # (batch, dim, seq)
        #         image_features = self.feature_dropout(image_features)
        #         image_features = image_features.permute(0, 2, 1)  # Back to (batch, seq, dim)

        # Project the image features
        image_features = self.vision_projection(image_features)

        # Encode text using text encoder
        text_embeddings, attention_mask = self.encode_text(questions)

        # Apply token dropout during training
        # if is_training:
        #     text_embeddings, attention_mask = self.token_dropout(
        #         text_embeddings, attention_mask, drop_prob=0.05
        #     )

        # Project text embeddings
        text_embeddings = self.text_projection(text_embeddings)

        # Process through multiple cross-attention blocks
        output = text_embeddings
        for block in self.cross_attn_blocks:
            attn_output = block[0](  # The attention module is the first in the sequential
                q=output,
                k=image_features,
                v=image_features,
                attention_mask=None  # Can use attention_mask here if MultiHeadCrossAttention supports it
            )
            # Apply dropout after attention
            attn_output = block[1](attn_output)  # Dropout is the second in the sequential

            # Residual connection with layer normalization
            output = self.layer_norm(attn_output + output)

        # Apply feature dropout during training
        # if is_training and self.use_feature_dropout:
        #     # Reshape for 2D dropout
        #     orig_shape = output.shape
        #     output = output.permute(0, 2, 1)  # (batch, dim, seq)
        #     output = self.feature_dropout(output)
        #     output = output.permute(0, 2, 1)  # Back to (batch, seq, dim)

        # Project through MLP
        output_projected = self.proj_mlp(output)

        # Apply global average pooling if enabled
        if self.use_global_pooling:
            # Use attention mask to get accurate pooling if available
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                # Sum and divide by number of actual tokens (not padding)
                output_pooled = (output_projected * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                # Simple mean pooling if no mask
                output_pooled = output_projected.mean(dim=1)

            # Pass through classification head
            logits = self.mlp_head(output_pooled)
        else:
            # Original approach - flatten and pass through MLP
            output_reshaped = output_projected.reshape(batch_size, -1)
            logits = self.mlp_head(output_reshaped)

        # Get the ground truth labels
        answers = samples['answer']
        dic = {'yes': 1, 'no': 0}
        answers_labels = torch.tensor([dic[answer] for answer in answers], dtype=torch.float,
                                      device=self.device).unsqueeze(1)

        # Apply mixup during training
        # if is_training and random.random() < 0.5:  # 50% chance of applying mixup
        #     logits, answers_labels = self.mixup_features(logits, answers_labels, alpha=self.mixup_alpha)

        # Compute loss with label smoothing if enabled
        # if is_training and self.use_label_smoothing:
        #     # Smooth targets for binary classification
        #     smooth_labels = answers_labels * (1.0 - self.label_smoothing) + self.label_smoothing * 0.5
        #     loss_answer = F.binary_cross_entropy_with_logits(logits, smooth_labels)
        # else:
        loss_answer = F.binary_cross_entropy_with_logits(logits, answers_labels)

        # Compute accuracy
        with torch.no_grad():
            predictions = torch.sigmoid(logits)
            answer_accuracy = self.accuracy(predictions, answers_labels.int())

            # Debug info - distribution of predictions
            mean_pred = predictions.mean().item()
            std_pred = predictions.std().item()
            max_pred = predictions.max().item()
            min_pred = predictions.min().item()

            # Count predictions above and below threshold
            yes_pred = (predictions >= 0.5).sum().item()
            no_pred = (predictions < 0.5).sum().item()

            if not is_training:  # Only print in evaluation to avoid cluttering logs
                print(
                    f"Predictions - Mean: {mean_pred:.3f}, Std: {std_pred:.3f}, Range: [{min_pred:.3f}, {max_pred:.3f}]")
                print(f"Predicted: {yes_pred} yes, {no_pred} no")

        return {
            'answer_accuracy': answer_accuracy,
            'loss_answer': loss_answer,
            'answer_predictions': predictions.detach(),  # For debugging
            'answer_labels': answers_labels.detach(),  # For debugging
        }

    def _unfreeze_clip_layers(self, num_layers=4):
        """
        Unfreeze the last n layers of the CLIP model for fine-tuning
        """
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
        """
        Unfreeze the last n layers of the BERT model for fine-tuning
        """
        # Freeze all parameters first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last n encoder layers
        for i, layer in enumerate(reversed(self.bert.encoder.layer)):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = True

        # Optionally, you might want to unfreeze the pooler as well
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable}")

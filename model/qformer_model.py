import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from model.blocks import CrossModalTransformer
from torchmetrics import Accuracy, AUROC
from model.clip_vit import VisionEncoder
import os
import random


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# # Optional: for even more detailed debugging
# os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Only works with certain PyTorch versions


class QFormer(nn.Module):
    """

    """

    def __init__(self,
                 sequence_size,
                 qformer_hidden_size,
                 blocks_num,
                 num_heads,
                 num_queries,
                 device,
                 use_clip_for_text=True,
                 clip_model_name="openai/clip-vit-large-patch14",
                 dropout_rate=0.3,
                 unfreeze_layers=4):
        super(QFormer, self).__init__()

        self.vision_dim = 1024  # Default for ViT

        self.device = device
        self.max_text_len = sequence_size
        self.use_clip_for_text = use_clip_for_text
        self.unfreeze_layers = unfreeze_layers

        # Text Encoder selection
        self.use_clip_for_text = use_clip_for_text
        if self.use_clip_for_text:
            self._setup_clip_model(clip_model_name, unfreeze_layers)
        else:
            self._setup_bert_model(unfreeze_layers)

        self.vision_encoder = VisionEncoder(model_name=clip_model_name,
                                            device=device,
                                            unfreeze_layers=unfreeze_layers)

        self.vision_projection = nn.Linear(self.vision_dim, qformer_hidden_size)
        self.text_projection = nn.Linear(self.text_dim, qformer_hidden_size)

        # learned queries for the cross attention
        self.learned_queries = nn.Parameter(torch.randn(1, num_queries, qformer_hidden_size))
        # nn.init.xavier_uniform_(self.learned_queries)

        self.max_text_len = sequence_size

        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # self.bert = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        # self.bert.resize_token_embeddings(len(self.tokenizer))
        # self.dec_token_id = self.tokenizer.convert_tokens_to_ids('[DEC]')
        # for parameter in self.bert.parameters():
        #     parameter.requires_grad = False

        self.cross_modal_transformer = CrossModalTransformer(
            qformer_hidden_size,
            num_heads,
            blocks_num,
            dropout=dropout_rate
        )

        self.temperature = nn.Parameter(
            torch.ones([]) * 0.07)  # Temperature parameter for scaling the similarity scores

        self.itm_head = nn.Linear(qformer_hidden_size, 2)  # ITM head

        self.lm_head = nn.Linear(qformer_hidden_size, self.tokenizer.vocab_size)  # LM head for text generation

        self.answer_head = nn.Linear(qformer_hidden_size, 1)  # Answer head

        self.accuracy = Accuracy(threshold=0.5, num_classes=2,
                                 task='binary')  # Accuracy metric for answer prediction

        # vision encoder

        # llm encoder

        # bert pretrained tokenizer

        # cross attention layer (transformer block)

        self.cat_mlp = nn.Sequential(
            nn.Linear(qformer_hidden_size * 2, qformer_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_hidden_size, 1)
        )

        self.init_weights()

        self.to(device)

    def _setup_clip_model(self, clip_model_name, unfreeze_clip_layers):
        """Setup the CLIP model and tokenizer."""
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Add a special token (like DEC token) to the CLIP tokenizer
        self.clip_tokenizer = self.clip_processor.tokenizer
        self.clip_tokenizer.add_special_tokens({"additional_special_tokens": ["[DEC]"]})

        # Get the new vocab size
        new_vocab_size = len(self.clip_tokenizer)
        old_vocab_size = self.clip_model.text_model.embeddings.token_embedding.weight.shape[0]

        # Resize token embeddings manually by replacing the embedding layer
        if new_vocab_size != old_vocab_size:
            old_embeddings = self.clip_model.text_model.embeddings.token_embedding
            new_embeddings = torch.nn.Embedding(new_vocab_size, old_embeddings.embedding_dim)

            # Initialize new embedding weights with the old ones
            new_embeddings.weight.data[:old_vocab_size] = old_embeddings.weight.data

            # Replace the old embedding layer with the new one
            self.clip_model.text_model.embeddings.token_embedding = new_embeddings

        # Convert the new token to an ID
        self.dec_token_id = self.clip_tokenizer.convert_tokens_to_ids('[DEC]')

        # Freeze CLIP model by default
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze the layers of the text encoder (if specified)
            # Unfreeze the specified number of layers (if requested)
        if unfreeze_clip_layers > 0:
            self._unfreeze_clip_layers(unfreeze_clip_layers)

        self.text_dim = self.clip_model.text_model.config.hidden_size

        # Use the updated tokenizer with special tokens
        self.tokenizer = self.clip_tokenizer

    def _setup_bert_model(self, unfreeze_bert_layers):
        """Setup the BERT model."""
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.dec_token_id = self.tokenizer.convert_tokens_to_ids('[DEC]')

        for param in self.bert.parameters():
            param.requires_grad = False

        if unfreeze_bert_layers > 0:
            self._unfreeze_bert_layers(unfreeze_bert_layers)

        self.text_dim = self.bert.config.hidden_size

    def init_weights(self):
        """
        Initialize the weights using best practices for attention models.
        """
        # Vision and text projections
        nn.init.kaiming_normal_(self.vision_projection.weight, nonlinearity='relu')
        nn.init.zeros_(self.vision_projection.bias)

        nn.init.kaiming_normal_(self.text_projection.weight, nonlinearity='relu')
        nn.init.zeros_(self.text_projection.bias)

        # Initialize learned queries with normalized values for stable training
        nn.init.normal_(self.learned_queries, mean=0.0, std=0.02)

        # The CrossModalTransformer already has its own initialization
        # in its _init_weights method, so we don't need to initialize it here

        # Initialize the temperature parameter
        nn.init.constant_(self.temperature, 0.07)

        # ITM head (Image-Text Matching)
        nn.init.normal_(self.itm_head.weight, std=0.02)
        nn.init.zeros_(self.itm_head.bias)

        # Language modeling head
        nn.init.normal_(self.lm_head.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

        # Answer prediction head
        nn.init.normal_(self.answer_head.weight, std=0.02)
        nn.init.zeros_(self.answer_head.bias)

        # Initialize cat_mlp (concatenation MLP)
        for layer in self.cat_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Final layer of cat_mlp with smaller weights
        final_layer = self.cat_mlp[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
            final_layer.bias.data.fill_(0.0)

        print("Model weights initialized successfully")

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

    def encode_text(self, questions):
        """Encode text using either BERT or CLIP's text encoder."""
        if self.use_clip_for_text:
            # Process text through CLIP tokenizer (with DEC token)
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
            # text_embeddings = text_outputs.last_hidden_state
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
            # text_embeddings = question_output.last_hidden_state

        return question_output, question_tokens

    def generate_attention_mask(self, task, query_len, pad_mask, device='cpu'):
        """
        Generates an attention mask based on the task (ITM, IGT, ITC) and padding mask.

        Args:
            task: Task type ("itm", "igt", "itc")
            query_len: Number of query tokens
            pad_mask: 1 - valid token, 0 - padding token, (batch_size, text_len)
            device: Device to place tensors on

        Returns:
            Attention mask tensor (batch_size, total_len, total_len)
            where 0 means "can attend" and -inf means "cannot attend"
        """
        # 1 - valid token, 0 - padding token
        batch_size, text_len = pad_mask.size()
        total_len = query_len + text_len  # Total length of the sequence (queries + text)

        # Initialize full attention mask - default is allowing attention (zeros)
        task_mask = torch.zeros((batch_size, total_len, total_len), device=device)

        if task == "itm":  # Works
            pass  # All tokens can attend to all other tokens (zeros)

        elif task == "igt":  # Works
            # 1. Create causal mask for text-to-text attention
            causal_indices = torch.triu_indices(text_len, text_len, offset=1, device=device)

            # Apply to text-to-text region for all samples
            for b in range(batch_size):
                task_mask[b,
                query_len + causal_indices[0],
                query_len + causal_indices[1]] = float('-inf')

            # 2. Queries should not attend to text tokens (set directly)
            task_mask[:, :query_len, query_len:] = float('-inf')

        elif task == "itc":  # Works
            # Block cross-modality attention (set directly)
            task_mask[:, :query_len, query_len:] = float('-inf')  # Queries cannot see text
            task_mask[:, query_len:, :query_len] = float('-inf')  # Text cannot see queries

        # Handle padding tokens
        # Create boolean masks for padding positions (True where padding exists)
        padding_positions = (pad_mask == 0)  # (batch_size, text_len)

        # For each batch item
        for b in range(batch_size):
            if padding_positions[b].any():  # Only process if there's padding
                # Get indices of padding tokens
                pad_indices = torch.nonzero(padding_positions[b], as_tuple=True)[0]

                # No token can attend to these padding tokens
                task_mask[b, :, query_len + pad_indices] = float('-inf')

                # These padding tokens cannot attend to any token
                task_mask[b, query_len + pad_indices, :] = float('-inf')

        return task_mask

    def forward(self, samples):
        # Get image features from CLIP ViT
        image_input = samples['image_input']
        image_features = self.vision_encoder.encode(image_input)  # (b, num_patches, hidden_dim)  (b, 256, 1024)
        # image_features = samples['image_features'].to(self.device)  # (b, num_patches, hidden_dim)  (b, 256, 1024)

        # Get the question text
        question = samples['question']

        batch_size = image_features.shape[0]

        # Project the image features to the same dimension as the text features and normalize
        image_features = F.normalize(self.vision_projection(image_features),
                                     dim=-1)  # Normalize the image features (b, num_patches, d)

        # Get the queries and expand them to the batch size (and clone for each task)
        queries = self.learned_queries.expand(image_features.shape[0], -1,
                                              -1).clone()  # (b, num_queries, hidden_dim)  (b, 32, 768)

        # Tokenize the question text
        question_output, question_tokens = self.encode_text(question)

        ############# Image Text Contrastive ######################
        cls_text_embedding = question_output['last_hidden_state'][:, 0, :]  # (b, d)

        cls_text_embedding = F.normalize(self.text_projection(cls_text_embedding),
                                         dim=-1)  # Normalize the text embedding  (b, d))

        attention_mask = self.generate_attention_mask(task='itm',
                                                      query_len=queries.shape[1],
                                                      pad_mask=question_tokens["attention_mask"],
                                                      device=self.device)  # (b, num_queries + text_len, num_queries + text_len)
        # text and queries should not attend to each other
        queries, _ = self.cross_modal_transformer(image_features,
                                                  queries,
                                                  text_embeddings=question_output['last_hidden_state'],
                                                  attention_mask=attention_mask)

        queries = F.normalize(queries, dim=-1)  # Normalize the queries (b, num_queries, d)

        # sim_matrix = torch.einsum("bqd, td -> btq", queries,
        #                           cls_text_embedding)  # (b, num_queries, d) x (b, d) -> (b, b, num_queries)

        # Image-to-text similarity calculation
        sim_i2t = torch.einsum("bqd, td -> btq", queries,
                               cls_text_embedding)  # (b, num_queries, d) x (t, d) -> (b, t, q)

        # sim_t2i = sim_i2t.permute(0, 2, 1)  # (b, q, t)
        #
        sim_i2t, _ = sim_i2t.max(-1)  # Max over queries: (b, t)
        sim_i2t = sim_i2t / self.temperature
        #
        # sim_t2i, _ = sim_t2i.max(-1)  # Max over clss: (b, t)
        # sim_t2i = sim_t2i / self.temperature
        sim_t2i = sim_i2t.T  # (b, q, t)

        targets = torch.arange(batch_size, device=image_features.device, dtype=int)

        loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) +
                    F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)) / 2

        ############# Image Text Matching ######################

        # Use image_features (from vision encoder) and text tokens (from bert tokenizer). Before the QFormer

        # Fill diagonal of sim matrix with really low value so that we don't sample positive pairs in the negative sampling
        with torch.no_grad():
            sim_i2t.fill_diagonal_(-10000)
            sim_t2i.fill_diagonal_(-10000)

        weights_t2i = torch.softmax(sim_t2i, dim=-1)  # (b, b)
        weights_i2t = torch.softmax(sim_i2t, dim=-1)

        image_embeddings_negative = []
        for b in range(batch_size):
            negative_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeddings_negative.append(image_features[negative_idx])

            # # Generate a random index that's different from the current one
            # available_indices = list(range(batch_size))
            # available_indices.remove(b)  # Remove current index
            # negative_idx = random.choice(available_indices)
            # image_embeddings_negative.append(image_features[negative_idx])

        image_embeddings_negative = torch.stack(image_embeddings_negative, dim=0)  # (b, num_patches, d)

        text_embeddings_negative = []
        attention_masks_negative = []

        for b in range(batch_size):
            negative_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeddings_negative.append(question_output['last_hidden_state'][negative_idx])
            attention_masks_negative.append(question_tokens['attention_mask'][negative_idx])

            # # Generate a random index that's different from the current one
            # available_indices = list(range(batch_size))
            # available_indices.remove(b)  # Remove current index
            # negative_idx = random.choice(available_indices)
            # text_embeddings_negative.append(question_output['last_hidden_state'][negative_idx])
            # attention_masks_negative.append(question_tokens['attention_mask'][negative_idx])

        text_embeddings_negative = torch.stack(text_embeddings_negative, dim=0)  # (b, max_len, d)
        attention_masks_negative = torch.stack(attention_masks_negative, dim=0)  # (b, max_len)
        attention_masks_negative = self.generate_attention_mask(task='itm',
                                                                query_len=queries.shape[1],
                                                                pad_mask=attention_masks_negative,
                                                                device=self.device)  # (b, num_queries + text_len, num_queries + text_len)

        # Concatenate the positive and negative samples (we get positive pairs, negative pairs, positive pairs)
        text_embeddings_all = torch.cat(
            [question_output['last_hidden_state'], question_output['last_hidden_state'], text_embeddings_negative],
            dim=0)  # (3b, max_len, d)

        image_embeddings_all = torch.cat([image_features, image_embeddings_negative, image_features],
                                         dim=0)  # (3b, num_patches, d)

        attention_masks_all = torch.cat(
            [self.generate_attention_mask(task='itm', query_len=self.learned_queries.shape[1], pad_mask=question_tokens["attention_mask"], device=self.device),
             self.generate_attention_mask(task='itm', query_len=self.learned_queries.shape[1], pad_mask=question_tokens["attention_mask"], device=self.device),
             attention_masks_negative],
            dim=0)

        queries_itm = self.learned_queries.expand(image_embeddings_all.shape[0], -1,
                                                  -1).clone()  # (3b, num_queries, d)

        # queries_itm = self.learned_queries.expand(image_features.shape[0], -1,
        #                                           -1).clone()  # (3b, num_queries, d)

        # TODO: concat the attention mask for here, and question output instead of ids all. Done, perform testing
        queries_itm, _ = self.cross_modal_transformer(image_embeddings_all,
                                                      queries_itm,
                                                      text_embeddings=text_embeddings_all,
                                                      attention_mask=attention_masks_all)  # (3b, num_queries, d)

        # attention_mask_itm = self.generate_attention_mask(
        #     task='itm',
        #     query_len=self.learned_queries.shape[1],
        #     pad_mask=question_tokens["attention_mask"],
        #     device=self.device
        # )

        # queries_itm, _ = self.cross_modal_transformer(image_features,
        #                                               queries_itm,
        #                                               text_embeddings=question_output['last_hidden_state'],
        #                                               attention_mask=attention_mask_itm)

        # Perform itm head
        itm_embeddings = self.itm_head(queries_itm)  # (3b, num_queries, 2)
        logits = torch.mean(itm_embeddings, dim=1)  # (3b, 2)

        itm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long),
                                torch.zeros(batch_size, dtype=torch.long),
                                torch.ones(batch_size, dtype=torch.long)], dim=0).to(self.device)

        loss_itm = F.cross_entropy(logits, itm_labels)

        ############# Image-Grounded Text Generation ######################
        # Clone the original input IDs and replace the first token with DEC token
        igt_input_ids = question_tokens["input_ids"].clone()
        igt_input_ids[:, 0] = self.dec_token_id

        labels = igt_input_ids.clone()
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)  # Mask padding tokens

        # Process through text encoder with the DEC token
        if self.use_clip_for_text:
            igt_text_output = self.clip_model.text_model(
                input_ids=igt_input_ids,
                attention_mask=question_tokens["attention_mask"],
                return_dict=True,
            )
        else:
            igt_text_output = self.bert(
                input_ids=igt_input_ids,
                attention_mask=question_tokens["attention_mask"],
                return_dict=True,
            )

        # Create IGT-specific attention mask (causal + queries can't see text)
        igt_attention_mask = self.generate_attention_mask(
            task='igt',
            query_len=queries.shape[1],
            pad_mask=question_tokens["attention_mask"],
            device=self.device
        )

        # Clone queries for IGT task
        queries_igt = self.learned_queries.expand(batch_size, -1, -1).clone()

        # Process through cross-modal transformer with IGT mask
        queries_igt, text_embeddings_igt = self.cross_modal_transformer(
            image_features,
            queries_igt,
            text_embeddings=igt_text_output['last_hidden_state'],
            attention_mask=igt_attention_mask
        )

        text_logits = self.lm_head(text_embeddings_igt)  # (batch_size, text_len, vocab_size)

        # Compute language modeling loss (shifted)
        # We want to predict the next token, so we shift the logits and labels
        shifted_logits = text_logits[:, :-1, :]  # Remove last token
        shifted_labels = labels[:, 1:]  # Remove first token (which was [DEC])

        # Calculate cross-entropy loss
        loss_igt = F.cross_entropy(
            shifted_logits.reshape(-1, self.tokenizer.vocab_size),
            shifted_labels.reshape(-1),
            ignore_index=-100  # Ignore padded positions
        )

        ############ Answer Prediction ######################
        # queries_itm  # b, 32, 768

        max_pooled_queries = torch.max(queries_itm[:batch_size], dim=1)[0]  # (b, d)
        #
        # # Perform answer head
        answer_logits = self.answer_head(max_pooled_queries)  # (b, 1)
        #
        # # Get the answers text
        answers = samples['answer']
        #
        # # Dictionary mapping for labels
        dic = {'yes': 1, 'no': 0}
        #
        # Convert labels to a tensor with the same device as logits, also unsqueeze to make it (B, 1)
        answers_labels = torch.tensor([dic[answer] for answer in answers], dtype=torch.float,
                                      device=answer_logits.device).unsqueeze(1)
        #
        # # Compute Binary Cross Entropy with logits
        loss_answer = F.binary_cross_entropy_with_logits(answer_logits, answers_labels)
        #
        p = torch.sigmoid(answer_logits)
        #
        answer_accuracy = self.accuracy(p, answers_labels.int())





        # CONCAT PART
        # cls_text_embedding = question_output['last_hidden_state'][:, 0, :]  # (b, d)
        # cls_text_embedding = F.normalize(self.text_projection(cls_text_embedding),
        #                        dim=-1)  # Normalize the text embedding  (b, d))
        #
        # # image_features  # b, d
        #
        # cat_image_text = torch.cat([cls_text_embedding, image_features],
        #                            dim=1)  # (b, 2d)
        #
        # answer_logits = self.cat_mlp(cat_image_text)  # (b, 1)
        #
        # # Get the answers text
        # answers = samples['answer']
        #
        # # Dictionary mapping for labels
        # dic = {'yes': 1, 'no': 0}
        #
        # # Convert labels to a tensor with the same device as logits, also unsqueeze to make it (B, 1)
        # answers_labels = torch.tensor([dic[answer] for answer in answers], dtype=torch.float,
        #                               device=answer_logits.device).unsqueeze(1)
        #
        # # Compute Binary Cross Entropy with logits
        # loss_answer = F.binary_cross_entropy_with_logits(answer_logits, answers_labels)
        #
        # p = torch.sigmoid(answer_logits)
        #
        # answer_accuracy = self.accuracy(p, answers_labels.int())



        return {
            'answer_accuracy': answer_accuracy,
            'loss_answer': loss_answer,
            'loss_itc': loss_itc,
            'loss_itm': loss_itm,
            'loss_igt': loss_igt,
            'total_loss': loss_itc + loss_itm + loss_igt + loss_answer,
            # 'total_loss': loss_itc + loss_igt + loss_answer,
            # 'queries_itm': queries_itm,
        }

        # # TODO:
        # # 3) Fix DEC token in bert
        # # 4) Implement each task
        # # 5) implement losses (account for PAD tokens in the cross entropy)
        #
        # queries, text_embeddings = self.cross_modal_transformer(image_features,
        #                                                         queries,
        #                                                         text_embeddings=question_output['last_hidden_state'],
        #                                                         attention_mask=None)
        #
        # queries = F.normalize(queries, dim=-1)

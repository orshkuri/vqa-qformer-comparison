import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union


class VisionEncoder:
    def __init__(self,
                 device,
                 model_name="openai/clip-vit-large-patch14",
                 unfreeze_layers=4,
                 only_use_processor=False):
        """
        Initialize the VisionEncoder with a CLIP model.
        """
        if not only_use_processor:
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self._unfreeze_clip_layers(num_layers=unfreeze_layers)

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device

    def path_to_tensor(self, image_paths: Union[str, List[str]]):
        """
        Convert image(s) to tensor(s) using the CLIP processor.
        Accepts a single image path or a list of image paths.
        Returns a BatchEncoding dict containing 'pixel_values'.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = [Image.open(path).convert("RGB") for path in image_paths]
        image_input = self.processor(images=images, return_tensors="pt").to(self.device)
        return image_input

    def encode(self, image_input) -> torch.Tensor:
        """
        Encode a batch of images using the CLIP model.
        """
        outputs = self.model.vision_model(**image_input)
        image_features = outputs.last_hidden_state
        image_features = image_features[:, 1:, :]  # Remove CLS token from each item in the batch
        return image_features  # shape: (batch_size, num_patches-1, hidden_dim)

    def _unfreeze_clip_layers(self, num_layers=4):
        """
        Unfreeze the last `num_layers` layers of the CLIP vision encoder for fine-tuning.
        All other parameters remain frozen.
        """
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last `num_layers` vision transformer blocks
        vision_layers = self.model.vision_model.encoder.layers
        for i, block in enumerate(reversed(vision_layers)):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = True

        # Count and print trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable}")


if __name__ == "__main__":
    model_name="openai/clip-vit-large-patch14"
    # # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Load and process image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Get final image embedding (pooled)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)  # shape: (1, hidden_dim)

    # Get patch-level embeddings (including CLS token)
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        patch_embeddings = vision_outputs.last_hidden_state  # shape: (1, num_patches+1, hidden_dim)

    # Print shapes
    print(f"Image embedding shape (pooled): {image_embedding.shape}")
    print(f"Patch embeddings shape: {patch_embeddings.shape}")

    # Optional: Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {total_params}")

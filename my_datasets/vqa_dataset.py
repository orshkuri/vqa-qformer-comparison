import os
from typing import Dict, List, Tuple, Optional, Union, Callable

import torch
from torch.utils.data import Dataset, DataLoader

from model.clip_vit import VisionEncoder


class VQADataset(Dataset):
    """
    A PyTorch Dataset for Visual Question Answering (VQA) tasks.
    Handles the new dataset format with image-question-answer triplets in text files.
    """

    def __init__(
            self,
            data_file_path: str,
            images_dir: str,
            image_model_name: str = "openai/clip-vit-large-patch14",
            transform: Optional[Callable] = None,
            max_length: int = 14,
            device='cpu'
    ):
        """
        Initialize the VQA dataset.

        Args:
            data_file_path: Path to the data file containing image names, questions, and answers
            images_dir: Directory containing the images
            image_model_name: CLIP model name for image processing
            transform: Optional custom transforms (used only if not using CLIP processor)
            max_length: Maximum length for tokenized text
            device: Device to run the model on ('cpu' or 'cuda')
        """

        self.images_dir = images_dir
        self.max_length = max_length
        self.device = device

        self.vision_encoder = VisionEncoder(device=device, model_name=image_model_name,
                                            only_use_processor=True)

        # Load data file and parse it
        self.samples = self._load_data_file(data_file_path)

    def _load_data_file(self, data_file_path: str) -> List[Dict]:
        """Load and parse the data file containing image names, questions, and answers."""
        samples = []

        with open(data_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Handle tab-separated format in TestImages.txt
            image, rest = line.split('\t', 1)
            image = image.split('#')[0]

            # Extract answer (last word) and make sure it's a valid answer
            words = rest.strip().split()
            if not words or words[-1].lower() not in ["yes", "no"]:
                continue  # Skip this line if no answer or invalid answer

            answer = words[-1].lower()

            # Extract question (everything except the last word and remove question marks)
            question_text = ' '.join(words[:-1]).strip()
            while question_text.endswith('?'):
                question_text = question_text[:-1].strip()

            image_path = os.path.join(self.images_dir, image)

            # Ensure image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                continue

            samples.append({
                "image_id": image,
                "question": question_text,
                "answer": answer,
                "image_path": image_path
            })

        return samples

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary with processed image features, question, and answer
        """
        sample = self.samples[idx]

        # Extract image features using the vision encoder
        image = self.vision_encoder.path_to_tensor(sample["image_path"])

        output = {
            "image": image["pixel_values"].squeeze(0),
            "question": sample["question"],
            "answer": sample["answer"],
            "image_id": sample["image_id"]
        }

        return output

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Custom collate function for DataLoader.
        """
        batch_dict = {}

        # Handle image features
        images = torch.stack([item["image"] for item in batch]).to(self.device)
        image_input = {"pixel_values": images.to(self.device)}
        batch_dict["image_input"] = image_input

        # Handle text and other fields
        batch_dict["question"] = [item["question"] for item in batch]
        batch_dict["answer"] = [item["answer"] for item in batch]
        batch_dict["image_id"] = [item["image_id"] for item in batch]

        return batch_dict


def create_vqa_dataloaders(
        train_file: str,
        val_file: str,
        test_file: str,
        images_dir: str,
        image_model_name: str = "openai/clip-vit-large-patch14",
        transform: Optional[Callable] = None,
        max_length: int = 14,
        batch_size: int = 32,
        device='cpu'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for VQA datasets (train, validation, and test).

    Args:
        train_file: Path to the training data file
        val_file: Path to the validation data file
        test_file: Path to the test data file
        images_dir: Directory containing the images
        image_model_name: CLIP model name for image processing
        transform: Optional custom transforms
        max_length: Maximum length for tokenized text
        batch_size: Batch size for the dataloaders
        device: Device to run the model on ('cpu' or 'cuda')

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Create train dataset
    train_dataset = VQADataset(
        data_file_path=train_file,
        images_dir=images_dir,
        image_model_name=image_model_name,
        device=device,
    )

    # Create validation dataset
    val_dataset = VQADataset(
        data_file_path=val_file,
        images_dir=images_dir,
        image_model_name=image_model_name,
        device=device,
    )

    # Create test dataset
    test_dataset = VQADataset(
        data_file_path=test_file,
        images_dir=images_dir,
        image_model_name=image_model_name,
        device=device,
    )

    # Print dataset sizes for debugging
    print(f"Train dataset created with {len(train_dataset)} samples")
    print(f"Validation dataset created with {len(val_dataset)} samples")
    print(f"Test dataset created with {len(test_dataset)} samples")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    return train_dataloader, val_dataloader, test_dataloader

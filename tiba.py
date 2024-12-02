import torch
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import cv2

class TiBA():
    def __init__(self, model, device):
        """
        TiBA (Token-based Importance Attribution) implementation.
        Args:
            model: Vision Transformer (ViT) model.
            processor: Image processor for preprocessing.
            device: Device to run computations (e.g., "cpu" or "cuda").
        """
        self.model = model.to(device)
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.device = device

    def tokenize_input(self, image):
        """
        Tokenize the input image and return the token embeddings and patch dimensions.
        Args:
            image: Input image (PIL Image).
        Returns:
            tokens: Input tokens (patch embeddings).
            patch_size: Patch size for the ViT.
        """
        inputs = self.processor(image, return_tensors="pt").to(self.device)
         # Add a batch dimension if not present
        if inputs['pixel_values'].dim() == 3:  # Shape [num_channels, height, width]
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        tokens = self.model.vit.embeddings(inputs['pixel_values'])  # Token embeddings
        patch_size = int(np.sqrt(tokens.shape[1]))  # Assuming square grid of patches
        return tokens, patch_size

    def compute_importance(self, image, target_class):
        """
        Compute token importance for the target class.
        Args:
            image: Input image (PIL Image).
            target_class: Target class index for importance attribution.
        Returns:
            token_importance: Importance scores for each token.
        """
        tokens, patch_size = self.tokenize_input(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device)
         # Add a batch dimension if not present
        if inputs['pixel_values'].dim() == 3:  # Shape [num_channels, height, width]
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        
        pixel_values = inputs['pixel_values']
        # Forward pass to get baseline logits
        baseline_logits = self.model(pixel_values).logits
        baseline_score = baseline_logits[0, target_class].item()
        
        # Patch size for the model
        patch_size = self.model.config.patch_size  # Usually 16 for ViT models
        num_patches = pixel_values.shape[-1] // patch_size  # Assuming square patches

        # Placeholder for importance scores
        token_importance = torch.zeros((num_patches, num_patches), device=self.device)


        # Perturb each patch (token) and measure the effect on the score
        for i in range(num_patches):
            for j in range(num_patches):
                # Create a copy of the input and mask the (i, j)-th patch
                perturbed_values = pixel_values.clone()
                x_start, x_end = i * patch_size, (i + 1) * patch_size
                y_start, y_end = j * patch_size, (j + 1) * patch_size
                perturbed_values[:, :, x_start:x_end, y_start:y_end] = 0  # Mask patch

                # Forward pass with perturbed input
                perturbed_logits = self.model(pixel_values=perturbed_values).logits
                perturbed_score = perturbed_logits[0, target_class].item()

                # Importance is the decrease in target class score
                token_importance[i, j] = baseline_score - perturbed_score

        # Normalize importance scores
        token_importance = token_importance / token_importance.sum()

        
        return token_importance.detach().cpu().numpy()

    def visualize_importance(self, image, importance, alpha=0.4):
        """
        Visualize token importance as a heatmap overlayed on the original image.
        Args:
            image: Original image (PIL Image).
            importance: Importance scores for each token.
            alpha: Transparency factor for the heatmap.
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(importance, (image.size[0], image.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Convert image to numpy
        image_np = np.array(image)
        overlay = cv2.addWeighted(image_np, alpha, heatmap_colored, 1 - alpha, 0)

        # Display the result
        plt.imshow(overlay)
        plt.axis("off")
        plt.show()

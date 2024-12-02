import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from torch.nn.functional import softmax
from torchvision.transforms.functional import to_pil_image
from PIL import Image

class LIMEExplainer:
    def __init__(self, model, device):
        """
        Initialize the LIME explainer.
        Args:
            model: PyTorch CNN model to be explained.
            device: Device to run the model (e.g., 'cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.device = device

    def predict(self, images):
        """
        Predict probabilities for a batch of images.
        Args:
            images: Batch of images as numpy arrays (H, W, C).
        Returns:
            Probabilities for each class.
        """
        if images.ndim == 3:  # If (batch, H, W)
            images = np.expand_dims(images, axis=-1)  # Add channel dimension to (batch, H, W, C)
        images = torch.tensor(images, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        with torch.no_grad():
            logits = self.model(images)
            probabilities = softmax(logits, dim=1).cpu().numpy()
        return probabilities

    def explain(self, image, transform, num_segments=50, target_class=0, num_samples=1000):
        """
        Explain the prediction for a single image using LIME.
        Args:
            image: Input image as a PIL image.
            transform: Transform function to preprocess the image.
            num_segments: Number of segments for superpixel segmentation.
            target_class: Target class index for explanation.
            num_samples: Number of perturbed samples to generate.
        Returns:
            Explanation weights and segments.
        """
        # Preprocess the image
        image_np = np.array(image)
        transformed_image = transform(image).unsqueeze(0).to(self.device)

        # Segment the image into superpixels
        segments = slic(image_np, n_segments=num_segments, compactness=5, sigma=1, channel_axis=None)

        # Generate perturbed samples
        samples = np.random.randint(0, 2, size=(num_samples, num_segments))
        perturbed_images = []

        for sample in samples:
            mask = np.zeros_like(image_np, dtype=np.float32)
            for i, active in enumerate(sample):
                if active:
                    mask[segments == i] = image_np[segments == i]
            perturbed_images.append(mask / 255.0)  # Normalize to [0, 1]

        perturbed_images = np.array(perturbed_images)

        # Get predictions for perturbed images
        predictions = self.predict(perturbed_images)
        target_probs = predictions[:, target_class]

        # Compute distances to the original image
        distances = pairwise_distances(samples, np.ones((1, num_segments)), metric="cosine").ravel()

        # Fit a linear model
        model = Ridge(alpha=1.0)
        weights = np.sqrt(np.exp(-(distances ** 2) / 0.25))  # Kernel function
        model.fit(samples, target_probs, sample_weight=weights)

        # Get explanation weights for each segment
        explanation_weights = model.coef_
        return explanation_weights, segments

    def visualize(self, explanation_weights, segments, image, alpha=0.5, save_path=None):
        """
        Visualize the explanation as a heatmap overlayed on the original image.
        Args:
            explanation_weights: Importance weights for each segment.
            segments: Segmented regions of the image.
            image: Original image (PIL Image or numpy array).
            alpha: Transparency factor for the heatmap overlay.
        """
        print("Explanation Weights:", explanation_weights)
        print("Min:", explanation_weights.min(), "Max:", explanation_weights.max())

        # Normalize explanation weights to [0, 1]
        explanation_weights = (explanation_weights - explanation_weights.min()) / (explanation_weights.max() - explanation_weights.min())

        # Map weights to the image using the segments
        heatmap = np.zeros_like(np.array(image), dtype=np.float32)
        for i, weight in enumerate(explanation_weights):
            heatmap[segments == i] = weight
        
        print("Heatmap Min:", heatmap.min(), "Heatmap Max:", heatmap.max())
        print("Heatmap Unique Values:", np.unique(heatmap))


        # If the image is grayscale, convert it to 3-channel RGB
        image_np = np.array(image)
        if len(image_np.shape) == 2:  # Grayscale check
            image_np = np.stack([image_np] * 3, axis=-1)  # Convert to (H, W, 3)

        # Apply colormap to the heatmap
        colored_heatmap = plt.cm.jet(heatmap)[:, :, :3]  # Convert to RGB heatmap

        # Overlay heatmap on the image
        overlay = (0.5 * image_np / 255.0 + 0.5 * colored_heatmap).clip(0, 1)
        # Display the result
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title("LIME Explanation Heatmap")
        plt.show()


        cv2.imwrite(save_path, overlay)
        print(f"Saved Grad-CAM overlay to {save_path}")

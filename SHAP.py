import shap
import numpy as np
import torch
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from skimage.segmentation import slic

class KernelSHAPExplainer:
    def __init__(self, model, device):
        """
        Initialize the Kernel SHAP explainer.
        
        Args:
            model: The CNN model to be explained.
            device: The device (CPU or GPU) on which the model is running.
        """
        self.model = model.to(device)
        self.device = device
    
    def preprocess_image(self, image, transform):
        """
        Preprocess the input image using the specified transform.
        
        Args:
            image: Input PIL image.
            transform: Preprocessing transform.
        
        Returns:
            Preprocessed image tensor.
        """
        return transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, images):
        """
        Predict probabilities for a batch of images.
        
        Args:
            images: Batch of images as numpy arrays.
        
        Returns:
            Predicted probabilities.
        """
        # Check if images are grayscale
        if len(images.shape) == 3:  # No channel dimension for grayscale
            images = images[:, np.newaxis, :, :]  # Add a channel dimension (batch, 1, H, W)
        
        # Convert numpy images to PyTorch tensors
        tensors = torch.tensor(images, dtype=torch.float32, device=self.device)

        # Permute dimensions from (batch, C, H, W)
        tensors = tensors.permute(0, 1, 2, 3)  # Already matches expected order for grayscale
        
        # Pass through the model
        with torch.no_grad():
            logits = self.model(tensors)
            probabilities = softmax(logits, dim=1).cpu().numpy()
        return probabilities

    def explain(self, image, transform, num_segments=50, target_class=None):
        """
        Explain the model's prediction using Kernel SHAP.
        
        Args:
            image: Input PIL image.
            transform: Preprocessing transform.
            num_segments: Number of segments for SLIC superpixel segmentation.
            target_class: Target class index to explain. If None, explain the predicted class.
        
        Returns:
            SHAP values and the segmented image.
        """
        # Convert image to numpy array
        image_np = np.array(image)
        
        # Segment the image using SLIC
        segments = slic(image_np, n_segments=num_segments, compactness=10, sigma=1, channel_axis=None)
        
        # Define a function to mask segments
        def mask_image(z, segments):
            out = np.zeros_like(image_np)
            for i in range(len(z)):
                if z[i]:
                    out[segments == i] = image_np[segments == i]
            return out

        # Define a function to predict probabilities for SHAP
        def f(z):
            masked_images = np.array([mask_image(z[i], segments) for i in range(z.shape[0])])
            return self.predict(masked_images)
        
        # Create a SHAP explainer
        explainer = shap.KernelExplainer(f, np.zeros((1, num_segments)))  # Baseline: All-zero mask
        
        # Predict the target class if not specified
        if target_class is None:
            image_tensor = self.preprocess_image(image, transform)
            logits = self.model(image_tensor)
            target_class = logits.argmax(dim=1).item()
        
        # Compute SHAP values for the target class
        shap_values = explainer.shap_values(np.ones((1, num_segments)), nsamples=500)
        #print(f"SHAP values type: {type(shap_values)}")
        #print(f"SHAP values content: {shap_values}")
        #print(f"SHAP values shape: {np.array(shap_values).shape}")
        return  shap_values[0, :, target_class], segments

    def visualize(self, shap_values, segments, image):
        """
        Visualize SHAP values as a heatmap overlayed on the original image.
        
        Args:
            shap_values: SHAP values for each segment.
            segments: Superpixel segments of the image.
            image: Original PIL image.
        """
        # Create a blank mask for the SHAP values
        mask = np.zeros_like(np.array(image), dtype=np.float32)
        for i, val in enumerate(shap_values):
            mask[segments == i] = val
        
        # Normalize mask values for visualization
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        
         # If the image is grayscale, convert it to RGB
        if len(np.array(image).shape) == 2:  # Grayscale image
            image_rgb = np.stack([np.array(image)] * 3, axis=-1)  # Convert to RGB
        else:
            image_rgb = np.array(image)

        # Overlay heatmap on the image
        heatmap = plt.cm.jet(mask)[:, :, :3]  # Apply colormap
        overlay = (0.5 * np.array(image_rgb) / 255.0 + 0.5 * heatmap).clip(0, 1)
        
        # Display the result
        plt.imshow(overlay)
        plt.axis("off")
        plt.title("Kernel SHAP Explanation")
        plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from utils import *


lablel_to_str = {0: "normal", 1: "pneumonia"}
img_size = (512, 512)


class GradCAM():

    def __init__(self, model, target_layer, device):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.img_transform = get_common_transform(img_size)

        self.hook_gradients()

    def hook_gradients(self):
        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activations(module, input, output):
            self.activations = output
        
        self.target_layer.register_forward_hook(save_activations)
        self.target_layer.register_backward_hook(save_gradients)


    def generate_heatmap(self, input_image, target_class):
        self.model.eval()

        input_tensor = self.img_transform(input_image).unsqueeze(0).to(self.device)

        output = self.model(input_tensor)
        class_score = output[:, target_class]

        self.model.zero_grad()
        class_score.backward(retain_graph=True)

        gradients = self.gradients.data.cpu().numpy()
        activations = self.activations.data.cpu().numpy()

        weights = np.mean(gradients, axis=(2,3))

        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        
        return cam
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):

        heatmap = cv2.resize(heatmap, img_size) # image dim: (b, c, w, h)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        transform = T.Compose([
                T.Grayscale(num_output_channels=1),  # Ensure grayscale
                T.ToTensor()  # Convert to PyTorch tensor
            ])
        image = transform(image)
        image = image.squeeze().cpu().numpy()
        image = np.uint8(255 * image)

        # Ensure dimensions match and print for debugging
        #print(f"Heatmap shape: {heatmap.shape}")  # Should be (H, W, 3)
        #print(f"Image shape: {cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).shape}")  # Should be (H, W, 3)

        overlay = cv2.addWeighted(heatmap, alpha, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 1 - alpha, 0)

        return overlay
    

    def visualize_gradcam(self, input_image, target_class, save_path=None):
       
        cam = self.generate_heatmap(input_image, target_class)
        overlay = self.overlay_heatmap(input_image, cam)

        cv2.imwrite(save_path, overlay)
        print(f"Saved Grad-CAM overlay to {save_path}")
    
        # Plot the result
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"Grad-CAM for Class {target_class}")
        plt.show()
        

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ViTGradCAM():
    def __init__(self, model, target_layer, device):
   
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


        self.hook_gradients()

    def hook_gradients(self):
        
        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]  # Gradients w.r.t the attention scores

        def save_activations(module, input, output):
            self.activations = output  # Activations (hidden states)

        self.target_layer.register_forward_hook(save_activations)
        self.target_layer.register_backward_hook(save_gradients)

    def generate_heatmap(self, input_tensor, target_class):

        self.model.eval()
        image = Image.open("example.jpg").convert("RGB")
        input_tensor = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        # Forward pass
        output = self.model(input_tensor)
        class_score = output.logits[:, target_class]

        # Backward pass
        self.model.zero_grad()
        class_score.backward(retain_graph=True)

        # Gradients and activations
        gradients = self.gradients.data.cpu().numpy()  # Gradients w.r.t. tokens
        activations = self.activations.data.cpu().numpy()  # Token activations

        # Compute weights (average gradients across tokens)
        weights = np.mean(gradients, axis=1)

        # Weighted sum of activations
        cam = np.zeros(activations.shape[2:], dtype=np.float32)  # Token grid shape
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        # ReLU activation to keep positive influence
        cam = np.maximum(cam, 0)

        # Normalize the CAM
        cam = cam / cam.max()

        # Reshape tokens to spatial grid
        patch_size = int(np.sqrt(activations.shape[1]))
        cam = cam.reshape((patch_size, patch_size))

        # Upsample to original image size
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))

        return cam

    def overlay_heatmap(self, original_image, heatmap, alpha=0.5):
        
        heatmap = cv2.resize(heatmap, img_size) # image dim: (b, c, w, h)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        transform = T.Compose([
                T.Grayscale(num_output_channels=1),  # Ensure grayscale
                T.ToTensor()  # Convert to PyTorch tensor
            ])
        image = transform(image)
        image = image.squeeze().cpu().numpy()
        image = np.uint8(255 * image)
        # Overlay heatmap on the original image
        overlay = cv2.addWeighted(original_image, alpha, heatmap, 1 - alpha, 0)
        return overlay

    def visualize(self, input_image, target_class, save_path=None):
       
        heatmap = self.generate_heatmap(input_image, target_class)
        overlay = self.overlay_heatmap(input_image, heatmap)

        # Save or display the image
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"Saved Grad-CAM overlay to {save_path}")
        else:
            plt.imshow(overlay)
            plt.axis("off")
            plt.show()


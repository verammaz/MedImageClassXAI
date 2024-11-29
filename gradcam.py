import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import torchvision.transforms as T
from PIL import Image

from cnn import SimpleCNN
from resnet import MyResNET 

lablel_to_str = {0: "normal", 1: "pneumonia"}

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class GradCAM():

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_gradients()

    def hook_gradients(self):
        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activations(module, input, output):
            self.activations = output
        
        self.target_layer.register_forward_hook(save_activations)
        self.target_layer.register_backward_hook(save_gradients)


    def generate_heatmap(self, input_tensor, target_class):
        self.model.eval()

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

        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3])) # image dim: (b, c, w, h)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        image = image.squeeze().cpu().numpy()
        image = np.uint8(255 * image)

         # Ensure dimensions match and print for debugging
        print(f"Heatmap shape: {heatmap.shape}")  # Should be (H, W, 3)
        print(f"Image shape: {cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).shape}")  # Should be (H, W, 3)

        overlay = cv2.addWeighted(heatmap, alpha, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 1 - alpha, 0)

        return overlay
    

def visualize_gradcam(input_image, model, target_layer, target_class):
    grad_cam = GradCAM(model, target_layer)

    cam = grad_cam.generate_heatmap(input_image, target_class)
    overlay = grad_cam.overlay_heatmap(input_image, cam)
    
    cv2.imwrite("gradcam_overlay.jpg", overlay)
    print("Saved Grad-CAM overlay to gradcam_overlay.jpg")
    
    if overlay.shape[-1] == 3:  # Check for 3-channel images
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    # Plot the result
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Grad-CAM for Class {target_class}")
    plt.show()
      

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', required=True, help='path to trained model.pth file with model weights')
    parser.add_argument('-model_type', required=True, help='SimpleCNN or ResNET')
    parser.add_argument('-img_path', required=True, help='path to image')
    parser.add_argument('-img_class', required=True, help='true label of image (0 - normal, 1 - pneumonia)')
    parser.add_argument('-img_size', default=(256,256), help="(width, height) size to resize input image to")

    args = parser.parse_args()

    # TODO: add check of model_type, img_class

    model = MyResNET() if args.model_type == 'ResNET' else SimpleCNN()
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    model.eval()

    image_transforms = T.Compose([
        T.Resize(tuple(args.img_size)),          
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])


    image = Image.open(args.img_path).convert("L")  # Ensure grayscale
    input_image = image_transforms(image).unsqueeze(0)  # Add batch dimension
    #print('Image Shape:', image.shape)
    print('Input Image Shape:', input_image.shape)

    # Target layer and class
    target_layer = model.model.layer4[-1] if args.model_type == 'ResNET' else model.conv_stack[-3]
    target_class = int(args.img_class)

    # Generate and visualize Grad-CAM
    visualize_gradcam(input_image, model, target_layer, target_class)


if __name__ == "__main__":
    main()
    




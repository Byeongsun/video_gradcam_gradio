import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import gradio as gr
import tempfile


def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

def load_labels():
    import urllib.request
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    return [line.strip().decode('utf-8') for line in urllib.request.urlopen(url).readlines()]

model = load_model()
labels = load_labels()

print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def analyze_video(video):
    cap = cv2.VideoCapture(video)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    results = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count < 50 or frame_count % 20 != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.framarray(rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output).item()
            class_name = labels[pred_class]
            
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        image_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
        cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        cam_pil = Image.fromarray(cam_image)

        results.append((cam_pil, f'Fram {frame_count} -> {class_name}'))
        
        if len(results) >= 5:
            break

    cap.release()
    return results

demo = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="동영상 업로드"),
    outputs=gr.Gallery(label="Grad-CAM 시각화 결과").style(grid=2)
)

if __name__ == "__main__":
    demo.launch()
    
import torch
from ultralytics import YOLO
import cv2
from config.settings import path_routing
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

# Print CUDA device details
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
def train():
    # Initialize the model
    model = YOLO(r"E:\1\Segmentation\app\yolo_pipeline\runs\segment\train3\weights\last.pt")

    # Train the model
    results = model.train(data=r'E:\1\Segmentation\config.yaml', epochs=100, imgsz=640, mode=train)

    # Save the trained model
    model.save("trained_yolo.pt")


if __name__ == '__main__':
    train()



def example_case(path_to_image, path_to_model, save_image_dir=None, to_mask=None):
    model_path = path_to_model.replace("\\", "/")  # Convert backslashes to forward slashes
    image_path = path_to_image.replace("\\", "/")  # Convert backslashes to forward slashes
    img = cv2.imread(image_path)
    H, W, _ = img.shape
    model = YOLO(model_path)
    results = model.predict(img, imgsz=640, box=True, show_labels=True, line_width=2)  # Ensure boxes are included

    detected_classes = []  # List to store detected class names and their bounding boxes
    bounding_boxes = []

    for result in results:  # Iterate over all results
        if result.boxes is not None and len(result.boxes) > 0:
            for j, box in enumerate(result.boxes):  # Extract boxes from results
                # Get bounding box coordinates
                x_min, y_min, x_max, y_max = map(int, box.xyxy.cpu().numpy()[0])
                bounding_boxes.append((x_min, y_min, x_max, y_max))  # Add bounding box coordinates
                class_index = int(box.cls.item())  # Convert tensor to integer index
                detected_classes.append((result.names[class_index],
                                         (x_min, y_min, x_max, y_max)))  # Add detected class name and bounding box

                # Save predictions
                if save_image_dir:
                    os.makedirs(save_image_dir, exist_ok=True)
                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    output_image_path = os.path.join(save_image_dir, f"{base_filename}_{result.names[class_index]}.png")

                    # Overlay masks on the image
                    overlay = np.zeros_like(img, dtype=np.uint8)
                    mask = None

                    if result.masks is not None and len(result.masks.data) > 0:
                        mask = result.masks.data[j].cpu().numpy() * 255  # Move tensor to CPU and convert to numpy array
                        mask = cv2.resize(mask, (W, H))
                        mask = mask.astype(np.uint8)  # Convert mask to uint8
                        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask, 0, 255)  # Add mask to overlay with clipping
                        overlay[:, :, 1] = np.clip(overlay[:, :, 1] + mask, 0, 255)  # Add mask to overlay with clipping
                        overlay[:, :, 2] = np.clip(overlay[:, :, 2] + mask, 0, 255)  # Add mask to overlay with clipping

                    res = cv2.addWeighted(img, 0.8, overlay, 0.5, 0)  # Overlay masks on the image

                    # Draw bounding box for the class
                    cv2.rectangle(res, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(res, result.names[class_index], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    cv2.imwrite(output_image_path, res)  # Save the result image
                    print(f"Saved result image to {output_image_path}")

                    # Save mask with class name appended
                    if result.masks is not None and len(result.masks.data) > 0 and to_mask:
                        os.makedirs(to_mask, exist_ok=True)
                        mask_filename = f"{base_filename}_{result.names[class_index]}.png"
                        mask_filepath = os.path.join(to_mask, mask_filename)
                        cv2.imwrite(mask_filepath, mask)  # Save the mask image
                        print(f"Saved mask image to {mask_filepath}")

    if len(detected_classes) == 0:
        print(f"No detections for image {image_path}")
        return None

    return img, overlay



# example_case(r'E:\Segmentation\database\processed_data\194\Yolo\images\slice_22_original.png', r'E:\1\Segmentation\best.pt')


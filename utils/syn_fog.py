import numpy as np
import cv2
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

def add_voc_fog(img, beta_range=(0.07, 0.12), A=0.5):
    img = img.astype(np.float32) / 255.0
    
    h, w = img.shape[:2]
    beta = np.random.uniform(beta_range[0], beta_range[1])
    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_wh = np.sqrt(max(w, h))  
    depth_map = -0.04 * distances + max_wh
    depth_map = np.maximum(depth_map, 0.1)
    transmission = np.exp(-beta * depth_map)
    if len(img.shape) == 3:
        transmission = np.stack([transmission] * 3, axis=2)
    hazy_img = img * transmission + A * (1 - transmission)
    hazy_img = np.clip(hazy_img, 0, 1)
    hazy_img = (hazy_img * 255).astype(np.uint8)
    return hazy_img, transmission, depth_map, beta

def save_hazy_image(clear_img_path, output_dir, beta_range=(0.07, 0.12), A=0.5, 
                   save_transmission=False, save_depth=False):
    clear_img = cv2.imread(clear_img_path)
    if clear_img is None:
        print(f"Failed: {clear_img_path}")
        return None
    clear_img_rgb = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
    hazy_img, transmission, depth_map, beta = add_voc_fog(clear_img_rgb, beta_range, A)
    hazy_img_bgr = cv2.cvtColor(hazy_img, cv2.COLOR_RGB2BGR)
    base_name = os.path.splitext(os.path.basename(clear_img_path))[0]
    hazy_output_path = os.path.join(output_dir, f"{base_name}.jpg")
    cv2.imwrite(hazy_output_path, hazy_img_bgr)
    params = {
        'original_image': clear_img_path,
        'hazy_image': hazy_output_path,
        'beta': float(beta),
        'A': float(A),
        'beta_range': [float(beta_range[0]), float(beta_range[1])],
        'image_size': list(clear_img.shape[:2]),
        'generation_time': datetime.now().isoformat()
    }
    if save_transmission:
        transmission_vis = (transmission[:,:,0] * 255).astype(np.uint8)
        transmission_path = os.path.join(output_dir, f"{base_name}_transmission.png")
        cv2.imwrite(transmission_path, transmission_vis)
        params['transmission_map'] = transmission_path
    if save_depth:
        depth_vis = create_depth_visualization(depth_map)
        depth_path = os.path.join(output_dir, f"{base_name}_depth.png")
        cv2.imwrite(depth_path, depth_vis)
        params['depth_map'] = depth_path
    params_path = os.path.join(output_dir, f"{base_name}_params.json")
    
    print(f"Down: {base_name}, β={beta:.3f}")
    return params

def create_depth_visualization(depth_map):
    depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_vis

def process_voc_fog_dataset(input_dir, output_dir, beta_range=(0.07, 0.12), A=0.5,
                          target_classes=None, max_images=None):
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for file in os.listdir(input_dir):
        if os.path.splitext(file)[1].lower() in valid_extensions:
            image_files.append(os.path.join(input_dir, file))
    if max_images is not None:
        image_files = image_files[:max_images]
    
    print(f"Total: {len(image_files)} images，processing..")
    
    all_params = []
    success_count = 0
    
    for i, img_path in enumerate(image_files):
        try:
            params = save_hazy_image(img_path, output_dir, beta_range, A)
            if params is not None:
                all_params.append(params)
                success_count += 1
        except Exception as e:
            print(f"Failed: {img_path}: {str(e)}")

        if (i + 1) % 100 == 0:
            print(f"processing {i + 1}/{len(image_files)}")

    summary = {
        'dataset_name': '',
        'total_images': len(image_files),
        'successful_images': success_count,
        'beta_range': [float(beta_range[0]), float(beta_range[1])],
        'A': float(A),
        'target_classes': target_classes if target_classes else ['car', 'bus', 'motorcycle', 'bicycle', 'person'],
        'generation_time': datetime.now().isoformat(),
        'individual_params': all_params
    }
    
    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    return summary


if __name__ == "__main__":
    input_directory = "" 
    output_directory = ""
    target_classes = ['car', 'bus', 'motorcycle', 'bicycle', 'person']
    summary = process_voc_fog_dataset(
        input_directory, 
        output_directory,
        beta_range=(0.07, 0.12),
        A=0.5,
        target_classes=target_classes,
        max_images=None 
    )

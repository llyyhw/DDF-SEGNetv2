import os
import cv2
import torch
import numpy as np
from collections import OrderedDict

from net.MultiFreqSegNetv2_lz import MultiFreqSegNetv2_lz

BASE_DIR = "your DIR PATH"
IMAGE_DIR = os.path.join(BASE_DIR, "visonimgs")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "visonimgs", "results")

MODEL_CONFIG = {
    "model_class": MultiFreqSegNetv2_lz,
    "weights_path": os.path.join(BASE_DIR, "NET_MODEL", "DDD-SEGNet_epoch_200.pth"),
    "model_args": {"num_classes": 3, "hidden_dim": 64, "num_freq_bands": 4}
}

INPUT_SIZE = 512
NUM_CLASSES = 3
CLASS_NAMES = ["background", "Moisture_condensed mulch film", "sub_main pipe"]

COLOR_MAP = {
    0: (26, 28, 44),
    1: (1, 175, 173),
    2: (2, 164, 221)
}

def load_model(device):
    print("Loading model...")
    
    model_class = MODEL_CONFIG["model_class"]
    model_args = MODEL_CONFIG["model_args"]
    model = model_class(**model_args)
    
    weights_path = MODEL_CONFIG["weights_path"]
    if not os.path.exists(weights_path):
        print(f"Error: Weight file not found: {weights_path}")
        return None
    
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if state_dict is None:
        state_dict = checkpoint
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {total_params:,}")
    
    return model

def inference(model, img_tensor, device):
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, tuple):
            output = output[0]
        
        probs = torch.softmax(output, dim=1)
        pred = probs.argmax(dim=1).squeeze(0).cpu().numpy()
    
    return pred

def process_image(model, img_path, output_dirs, device):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        return
    
    original_img = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img_rgb.shape[:2]
    
    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    pred = inference(model, img_tensor, device)
    
    pred_resized = cv2.resize(pred.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
    
    color_mask = np.zeros((h0, w0, 3), dtype=np.uint8)
    for cls_id, color in COLOR_MAP.items():
        color_mask[pred_resized == cls_id] = color
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    gray_path = os.path.join(output_dirs["masks"], f"{base_name}_mask.png")
    cv2.imwrite(gray_path, pred_resized)
    
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    color_path = os.path.join(output_dirs["color_masks"], f"{base_name}_mask_color.png")
    cv2.imwrite(color_path, color_mask_bgr)
    
    overlay = cv2.addWeighted(original_img, 0.6, color_mask_bgr, 0.4, 0)
    overlay_path = os.path.join(output_dirs["overlays"], f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    
    print(f"Processed: {base_name}")

def main():
    print("MultiFreqSegNet Inference Script")
    print("=" * 50)
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory not found: {IMAGE_DIR}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    output_dirs = {
        "masks": os.path.join(OUTPUT_BASE_DIR, "masks"),
        "color_masks": os.path.join(OUTPUT_BASE_DIR, "color_masks"),
        "overlays": os.path.join(OUTPUT_BASE_DIR, "overlays")
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    model = load_model(device)
    if model is None:
        return
    
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(extensions)]
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} images")
    print("Starting inference...")
    print("-" * 50)
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, img_file)
        process_image(model, img_path, output_dirs, device)
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(image_files)}")
    
    print("-" * 50)
    print(f"Inference completed!")
    print(f"Results saved to: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()
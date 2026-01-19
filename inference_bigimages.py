```python
import os
import cv2
import torch
import numpy as np
from collections import OrderedDict
from net.MultiFreqSegNet import MultiFreqSegNet

IMAGE_DIR = "/your DIR PATH/images"
OUTPUT_DIR = "/your DIR PATH/oraldata/predictions"
WEIGHT_PATH = "/ryour DIR PATH/1checkpoint/best_checkpoint_epoch_108.pth"
INPUT_SIZE = 512
STRIDE = 512
CLASS_NAMES = ["background", "Moisture_condensed mulch film", "sub_main pipe"]

COLOR_MAP = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255)
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiFreqSegNet()

checkpoint = torch.load(WEIGHT_PATH, map_location=device)
state_dict = checkpoint['model_state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '')
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

def predict_large_image(img, model, input_size=512, stride=512):
    h0, w0 = img.shape[:2]
    full_mask = np.zeros((h0, w0), dtype=np.uint8)
    
    for y in range(0, h0, stride):
        for x in range(0, w0, stride):
            patch = img[y:y+input_size, x:x+input_size]
            ph, pw = patch.shape[:2]
            
            if ph < input_size or pw < input_size:
                padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
                padded[:ph, :pw, :] = patch
                patch = padded
            
            img_tensor = torch.from_numpy(patch).float().permute(2,0,1)/255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                probs = torch.softmax(output, dim=1)
                pred = probs.argmax(dim=1).squeeze(0).cpu().numpy()
            
            pred = pred[:ph, :pw]
            full_mask[y:y+ph, x:x+pw] = pred
    
    return full_mask

def mask_to_color(mask, color_map):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in color_map.items():
        color_mask[mask==cls_id] = color
    return color_mask

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    full_mask = predict_large_image(img_rgb, model, input_size=INPUT_SIZE, stride=STRIDE)
    color_mask = mask_to_color(full_mask, COLOR_MAP)
    
    base_name = os.path.splitext(img_file)[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_mask.png"), full_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_mask_color.png"), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    
    color_mask_resized = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    if img.shape != color_mask_resized.shape:
        color_mask_resized = cv2.resize(color_mask_resized, (img.shape[1], img.shape[0]))
    comparison = np.concatenate((img, color_mask_resized), axis=1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png"), comparison)
    
    print(f"✅ Done: {base_name}")
```
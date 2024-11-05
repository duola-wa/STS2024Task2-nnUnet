import os
import json
import numpy as np
import cv2
from pathlib import Path

# Define paths
input_dir = Path(r"F:\dataset\miccaiSTS\task1pipeline\gptWrite\0923result0.85\FINAL_OUTPUT_DIR")
output_dir = Path(r"F:\dataset\miccaiSTS\task1pipeline\gptWrite\0923result0.85\FINAL_OUTPUT_DIR\output")
output_dir.mkdir(exist_ok=True)

def extract_contours(mask):
    contours = []
    unique_values = np.unique(mask)
    for value in unique_values:
        if value == 0:  # Skip background
            continue
        binary = np.uint8(mask == value)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:  # Collect all contours
            contours.append((value, cnt))
    return contours

def contour_to_points(contour):
    return [point[0].tolist() for point in contour]

def value_to_label(value):
    """Directly convert FDI number to string label."""
    return str(value)

def process_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    contours = extract_contours(mask)
    
    shapes = []
    for value, contour in contours:
        label = value_to_label(value)
        points = contour_to_points(contour)
        shapes.append({
            "label": label,
            "points": points
        })
    
    return shapes, mask.shape[0], mask.shape[1]

def main():
    png_files = list(input_dir.glob("*.png"))

    # Check if we need to remove "_0000" suffix
    remove_suffix = False
    if len(png_files) >= 2:
        first_file = png_files[0].name.rsplit('.', 1)[0]
        second_file = png_files[1].name.rsplit('.', 1)[0]
        if first_file.endswith("_0000") and second_file.endswith("_0000"):
            remove_suffix = True

    for mask_file in png_files:
        # Process filename
        base_name = mask_file.name.rsplit('.', 1)[0]
        if remove_suffix and base_name.endswith("_0000"):
            base_name = base_name[:-5]  # Remove "_0000"
        json_filename = f"{base_name}_Mask.json"
        json_path = output_dir / json_filename

        shapes, height, width = process_mask(mask_file)

        json_data = {
            "shapes": shapes,
            "imageHeight": height,
            "imageWidth": width
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed {mask_file.name} -> {json_filename}")

if __name__ == "__main__":
    main()

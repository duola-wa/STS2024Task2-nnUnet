import os
import numpy as np
from PIL import Image

def process_image_and_mask(image_path, mask_path, output_image_dir, output_mask_dir, padding=10):
    # Load image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    
    # Convert to numpy arrays
    image_array = np.array(image)
    mask_array = np.array(mask)

    # Get image dimensions
    height, width = mask_array.shape[:2]

    # Define pixel value mapping
    pixel_mapping = {
        # First quadrant
        1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1,
        33: 13, 34: 12, 35: 11, 36: 10, 37: 9,
        # Second quadrant
        9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7, 16: 8,
        38: 9, 39: 10, 40: 11, 41: 12, 42: 13,
        # Third quadrant
        17: 8, 18: 7, 19: 6, 20: 5, 21: 4, 22: 3, 23: 2, 24: 1,
        43: 13, 44: 12, 45: 11, 46: 10, 47: 9,
        # Fourth quadrant
        25: 1, 26: 2, 27: 3, 28: 4, 29: 5, 30: 6, 31: 7, 32: 8,
        48: 9, 49: 10, 50: 11, 51: 12, 52: 13
    }

    # Process each quadrant
    quadrants = [
        ((1, 8), (33, 37)),   # First quadrant
        ((9, 16), (38, 42)),  # Second quadrant
        ((17, 24), (43, 47)), # Third quadrant
        ((25, 32), (48, 52))  # Fourth quadrant
    ]

    for q, ((start1, end1), (start2, end2)) in enumerate(quadrants, 1):
        # Create mask for current quadrant
        quadrant_mask = np.logical_or(
            np.logical_and(mask_array >= start1, mask_array <= end1),
            np.logical_and(mask_array >= start2, mask_array <= end2)
        )
        
        if not np.any(quadrant_mask):
            continue  # Skip if quadrant is empty

        # Find the bounding box of the ROI
        rows, cols = np.where(quadrant_mask)
        if len(rows) == 0 or len(cols) == 0:
            continue  # Skip if no pixels in this quadrant

        # Expand the ROI by padding
        top = max(rows.min() - padding, 0)
        bottom = min(rows.max() + padding, height - 1)
        left = max(cols.min() - padding, 0)
        right = min(cols.max() + padding, width - 1)

        # Extract ROI from image and mask
        roi_image = image_array[top:bottom+1, left:right+1]
        roi_mask = mask_array[top:bottom+1, left:right+1]

        # Create a mask for the current quadrant's valid pixel range
        valid_mask = np.logical_or(
            np.logical_and(roi_mask >= start1, roi_mask <= end1),
            np.logical_and(roi_mask >= start2, roi_mask <= end2)
        )

        # Apply pixel mapping
        new_mask = np.zeros_like(roi_mask)
        for old_val, new_val in pixel_mapping.items():
            new_mask[roi_mask == old_val] = new_val
        
        # Set pixels from other quadrants to 14
        new_mask[~valid_mask & (roi_mask > 0)] = 14
        
        # Set remaining background pixels to 0
        new_mask[roi_mask == 0] = 0

        # Save new image and mask
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        Image.fromarray(roi_image).save(os.path.join(output_image_dir, f"{base_name}_q{q}.png"))
        Image.fromarray(new_mask.astype(np.uint8)).save(os.path.join(output_mask_dir, f"{base_name}_q{q}_mask.png"))

def main():
    image_dir = r"F:\dataset\miccaiSTS\Dataset430_QuaSegPre\trainforPreTrain_QuaSeg\train\images"
    mask_dir = r"F:\dataset\miccaiSTS\Dataset430_QuaSegPre\trainforPreTrain_QuaSeg - originData\train\mask"
    output_image_dir = r"F:\dataset\miccaiSTS\Dataset433\PreTrain_SeqInQua\images"
    output_mask_dir = r"F:\dataset\miccaiSTS\Dataset433\PreTrain_SeqInQua\masks"

    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace(".png", "_mask.png"))
            
            if os.path.exists(mask_path):
                process_image_and_mask(image_path, mask_path, output_image_dir, output_mask_dir)
            else:
                print(f"Mask not found for {filename}")

if __name__ == "__main__":
    main()
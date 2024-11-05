import os
import SimpleITK as sitk
import numpy as np

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def remap_mask(mask_array, quadrant):
    new_mask = np.zeros_like(mask_array)
    quadrant_ranges = [
        (1, 8),   # First quadrant
        (9, 16),  # Second quadrant
        (17, 24), # Third quadrant
        (25, 32)  # Fourth quadrant
    ]
    start, end = quadrant_ranges[quadrant - 1]
    
    # Set current quadrant teeth to their original values (shifted if necessary)
    new_mask[(mask_array >= start) & (mask_array <= end)] = mask_array[(mask_array >= start) & (mask_array <= end)] - start + 1
    
    # Set teeth from other quadrants to 9
    new_mask[(mask_array > 0) & ((mask_array < start) | (mask_array > end))] = 9
    
    return new_mask

def process_images(image_dir, mask_dir, output_image_dir, output_mask_dir):
    create_directory(output_image_dir)
    create_directory(output_mask_dir)

    for filename in os.listdir(image_dir):
        if filename.endswith('_0000.nii.gz'):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace('_0000', ''))

            if not os.path.exists(mask_path):
                print(f"Mask file not found for {filename}")
                continue

            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)

            image_array = sitk.GetArrayFromImage(image)
            mask_array = sitk.GetArrayFromImage(mask)

            quadrant_ranges = [
                (1, 8),   # First quadrant
                (9, 16),  # Second quadrant
                (17, 24), # Third quadrant
                (25, 32)  # Fourth quadrant
            ]

            for quadrant, (start, end) in enumerate(quadrant_ranges, 1):
                quadrant_mask = np.zeros_like(mask_array)
                quadrant_mask[mask_array > 0] = mask_array[mask_array > 0]
                
                # Find the bounding box of the quadrant
                non_zero = np.nonzero((mask_array >= start) & (mask_array <= end))
                if len(non_zero[0]) == 0:  # Skip if the quadrant is empty
                    continue
                
                bbox_min = np.min(non_zero, axis=1)
                bbox_max = np.max(non_zero, axis=1)

                # Add some padding to the bounding box
                padding = 2
                bbox_min = np.maximum(bbox_min - padding, 0)
                bbox_max = np.minimum(bbox_max + padding, np.array(mask_array.shape) - 1)

                # Extract ROI from image and mask
                roi_image = image_array[bbox_min[0]:bbox_max[0]+1, 
                                        bbox_min[1]:bbox_max[1]+1, 
                                        bbox_min[2]:bbox_max[2]+1]
                roi_mask = quadrant_mask[bbox_min[0]:bbox_max[0]+1, 
                                         bbox_min[1]:bbox_max[1]+1, 
                                         bbox_min[2]:bbox_max[2]+1]

                # Remap the mask values
                remapped_mask = remap_mask(roi_mask, quadrant)

                # Create SimpleITK images
                new_image = sitk.GetImageFromArray(roi_image)
                new_mask = sitk.GetImageFromArray(remapped_mask)

                # Set the same metadata as the original image
                new_image.SetSpacing(image.GetSpacing())
                new_image.SetDirection(image.GetDirection())
                new_origin = image.TransformContinuousIndexToPhysicalPoint(bbox_min.tolist())
                new_image.SetOrigin(new_origin)
                new_mask.CopyInformation(new_image)

                # Save new image and mask
                base_name = os.path.splitext(filename)[0]
                new_image_name = f"{base_name}_quadrant{quadrant}.nii.gz"
                new_mask_name = f"{base_name}_quadrant{quadrant}_mask.nii.gz"

                sitk.WriteImage(new_image, os.path.join(output_image_dir, new_image_name))
                sitk.WriteImage(new_mask, os.path.join(output_mask_dir, new_mask_name))

                print(f"Processed {new_image_name}")

    print("Processing complete.")

# Set your directories here
image_dir = r'F:\dataset\miccaiSTS\Task030\imagesTr'
mask_dir = r'F:\dataset\miccaiSTS\Task030\labelsTr'
output_image_dir = r'F:\dataset\miccaiSTS\Task031\quadrant_imagesTr_paddiing2'
output_mask_dir = r'F:\dataset\miccaiSTS\Task031\quadrant_labelsTr_paddiing2'

# Run the processing
process_images(image_dir, mask_dir, output_image_dir, output_mask_dir)
import os
import SimpleITK as sitk
import numpy as np

def check_mask_values(directory):
    print(f"Checking mask values in directory: {directory}")
    print("----------------------------------------")

    for filename in os.listdir(directory):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(directory, filename)
            
            try:
                # Read the mask file
                mask = sitk.ReadImage(file_path)
                mask_array = sitk.GetArrayFromImage(mask)

                # Get unique values
                unique_values = np.unique(mask_array)

                # Print the results
                print(f"File: {filename}")
                print(f"Unique mask values: {unique_values}")
                print("----------------------------------------")

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                print("----------------------------------------")

# Specify the directory containing the mask files
mask_directory = r'F:\dataset\miccaiSTS\Task018\labelsTr'

# Run the check
check_mask_values(mask_directory)

print("Check complete.")
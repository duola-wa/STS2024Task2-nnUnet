#!/bin/bash

# 设置路径
ORIGINAL_IMAGES_DIR="/inputs/"
NNUNET_INPUTS_DIR="/nnUNet_inputs/"
OUTPUT_QUADRANT_SEG_DIR="/opt/algorithm/output_quadrant_segmentation/"
OUTPUT_TOOTH_SEG_DIR="/opt/algorithm/output_tooth_segmentation/"
SAVE_QUADRANT_CROP_PATH="/opt/algorithm/quadrant_cropped_images/"
SAVE_FINAL_SEG_PATH="/outputs/"
QUADRANT_RESIZER_PATH="/opt/algorithm/quadrant_resizer/"
TASK_ID_QUADRANT="31"
TASK_ID_TOOTH="312"

# 函数：复制并重命名文件
copy_and_rename_files() {
    local src_dir="$1"
    local dst_dir="$2"

    for file in "$src_dir"/*.nii.gz; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            new_filename="${filename%.nii.gz}_0000.nii.gz"
            cp "$file" "$dst_dir/$new_filename"
            echo "Copied and renamed: $filename to $new_filename"
        fi
    done
}

# Step 0: 复制并重命名文件
echo "Step 0: Copying and renaming files..."
copy_and_rename_files "$ORIGINAL_IMAGES_DIR" "$NNUNET_INPUTS_DIR"

# Step 1: 使用nnUNet进行象限分割
echo "Step 1: Running nnUNet for quadrant segmentation..."
nnUNet_predict -i "$NNUNET_INPUTS_DIR" -o "$OUTPUT_QUADRANT_SEG_DIR" -t "$TASK_ID_QUADRANT" -m 3d_lowres -f all --disable_tta -chk model_best

# Step 2: 切割象限数据
echo "Step 2: Cutting quadrants based on segmentation results..."
python /opt/algorithm/quadrant_locate.py "$NNUNET_INPUTS_DIR" "$OUTPUT_QUADRANT_SEG_DIR" "$QUADRANT_RESIZER_PATH" "$SAVE_QUADRANT_CROP_PATH"
python /opt/algorithm/deleteLessThan50.py

# Step 3: 使用nnUNet进行象限内牙齿分割
echo "Step 3: Running nnUNet for tooth segmentation within quadrants..."
nnUNet_predict -i "$SAVE_QUADRANT_CROP_PATH" -o "$OUTPUT_TOOTH_SEG_DIR" -t "$TASK_ID_TOOTH" -m 3d_fullres -f all -chk model_best --disable_tta

# Step 4: 合并象限结果
echo "Step 4: Merging quadrant results into final segmentation..."
python /opt/algorithm/quadrant_merge.py "$NNUNET_INPUTS_DIR" "$NNUNET_INPUTS_DIR" "$OUTPUT_TOOTH_SEG_DIR" "$QUADRANT_RESIZER_PATH" "$SAVE_FINAL_SEG_PATH"
python /opt/algorithm/deleteLessThan30.py
echo "Pipeline completed successfully."

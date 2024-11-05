#!/bin/bash

# 定义路径
IMAGES_DIR="./imagesTs/"
QUADRANT_INFER_DIR="./quadrant_infer/"
ROI_DIR="./ROI_DIR/"
TEETH_INFER_DIR="./teeth_infer/"
FINAL_OUTPUT_DIR="./FINAL_OUTPUT_DIR/"

# 创建必要的目录
mkdir -p "$QUADRANT_INFER_DIR"
mkdir -p "$ROI_DIR"
mkdir -p "$TEETH_INFER_DIR"
mkdir -p "$FINAL_OUTPUT_DIR"

# 步骤1：运行象限分割模型
echo "正在运行象限分割模型..."
nnUNetv2_predict -i "$IMAGES_DIR" -o "$QUADRANT_INFER_DIR" -d 411 -c 2d -f all -tr nnUNetTrainerNoDA -chk checkpoint_best.pth

echo "run 1_deletepixelLessx.py"
python 1_deletepixelLessx.py
# 步骤2：根据象限分割的mask生成ROI，并记录映射关系
echo "正在根据象限分割生成ROI..."
python generate_roi.py --image_dir "$IMAGES_DIR" --mask_dir "$QUADRANT_INFER_DIR" --output_dir "$ROI_DIR" --padding 10

# 步骤3：在每个ROI上运行牙齿分割模型
echo "正在运行牙齿分割模型..."
nnUNetv2_predict -i "$ROI_DIR" -o "$TEETH_INFER_DIR" -d 412 -c 2d -f all -chk checkpoint_best.pth


# 步骤4：将牙齿分割结果拼接成完整的牙齿分割图
echo "正在拼接牙齿分割结果..."
echo "IMAGES_DIR: $IMAGES_DIR"
python merge_results.py --mapping_file "$ROI_DIR/roi_mapping.csv" --teeth_infer_dir "$TEETH_INFER_DIR" --output_dir "$FINAL_OUTPUT_DIR" --original_image_dir "$IMAGES_DIR"


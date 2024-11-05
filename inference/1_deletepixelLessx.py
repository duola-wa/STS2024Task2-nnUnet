import os
import numpy as np
from scipy import ndimage
import cv2
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_mask(mask_path, threshold=2000):
    # 读取灰度 mask 图像
    gray_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gray_mask is None:
        logging.error(f"Error: Unable to read the image at {mask_path}")
        return
    
    logging.info(f"Original mask shape (Grayscale): {gray_mask.shape}")
    
    height, width = gray_mask.shape[:2]
    
    # 计算5%的边界
    border_h = max(int(height * 0.05), 1)
    border_w = max(int(width * 0.05), 1)
    
    # 首先，将边界区域设置为黑色 (0)
    gray_mask[:border_h, :] = 0
    gray_mask[-border_h:, :] = 0
    gray_mask[:, :border_w] = 0
    gray_mask[:, -border_w:] = 0

    # 找到所有非零的像素值
    unique_values = np.unique(gray_mask)
    unique_values = unique_values[unique_values != 0]

    for value in unique_values:
        # 创建二值掩码
        binary_mask = (gray_mask == value)
        
        # 标记连通域
        labeled_array, num_features = ndimage.label(binary_mask)
        
        # 处理每个连通域
        for label in range(1, num_features + 1):
            component = (labeled_array == label)
            component_size = np.sum(component)
            
            
            if component_size < threshold:
                # 将小于阈值的连通域设置为0
                gray_mask[component] = 0

    logging.info(f"Processed mask shape (Grayscale): {gray_mask.shape}")

    # 直接覆盖原文件，确保保存为单通道
    cv2.imwrite(mask_path, gray_mask)

    logging.info(f"Mask processed and saved (in-place) to: {mask_path}")

# 设置输入目录
input_dir = "./quadrant_infer/"
# input_dir = r'F:\dataset\miccaiSTS\task1pipeline\gptWrite\quadrant_infer\infer_mask\1'
# 处理目录中的所有mask图像
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        mask_path = os.path.join(input_dir, filename)
        process_mask(mask_path)

logging.info("All masks have been processed in-place.")

import os
import numpy as np
from PIL import Image
import argparse
import csv
from scipy import ndimage

def process_image_and_mask(image_path, mask_path, output_image_dir, mapping_writer, tmp_dir, padding=10):
    # 加载图像和掩码
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path)

    # 转换为numpy数组
    image_array = np.array(image)
    mask_array = np.array(mask)

    # 获取图像尺寸
    height, width = mask_array.shape[:2]

    # **添加调试信息 - 开始**
    print(f"正在处理图像：{os.path.basename(image_path)}")
    print(f"图像尺寸：height={height}, width={width}")
    # **添加调试信息 - 结束**

    # 保存临时结果
    tmp_mask_path = os.path.join(tmp_dir, f"tmp_{os.path.basename(mask_path)}")
    Image.fromarray(mask_array.astype(np.uint8)).save(tmp_mask_path)
    print(f"临时掩码已保存至：{tmp_mask_path}")

    # 定义象限的标签值（根据你的实际标签进行调整）
    quadrants = [1, 2, 3, 4]

    for q in quadrants:
        # 创建当前象限的掩码
        quadrant_mask = (mask_array == q)

        if not np.any(quadrant_mask):
            continue  # 如果当前象限为空，跳过

        # 找到ROI的边界框
        rows, cols = np.where(quadrant_mask)
        top = max(rows.min() - padding, 0)
        bottom = min(rows.max() + padding, height - 1)
        left = max(cols.min() - padding, 0)
        right = min(cols.max() + padding, width - 1)

        # **添加调试信息 - 开始**
        print(f"象限 {q} 的 ROI 位置：top={top}, bottom={bottom}, left={left}, right={right}")
        print(f"象限 {q} 的 ROI 尺寸：height={bottom - top + 1}, width={right - left + 1}")
        # **添加调试信息 - 结束**

        # 提取ROI
        roi_image = image_array[top:bottom+1, left:right+1]

        # 保存ROI图像，命名为 caseid_qY_0000.png
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        roi_filename = f"{base_name}_q{q}_0000.png"
        roi_image_path = os.path.join(output_image_dir, roi_filename)
        Image.fromarray(roi_image).save(roi_image_path)

        # 记录映射关系
        mapping_writer.writerow([roi_filename, base_name, q, top, bottom, left, right])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True, help='原始图像目录')
    parser.add_argument('--mask_dir', required=True, help='象限分割结果目录')
    parser.add_argument('--output_dir', required=True, help='ROI输出目录')
    parser.add_argument('--padding', type=int, default=10, help='ROI外扩的像素数')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 创建临时目录
    tmp_dir = os.path.join(args.output_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # 创建映射文件
    mapping_file_path = os.path.join(args.output_dir, 'roi_mapping.csv')
    with open(mapping_file_path, mode='w', newline='') as mapping_file:
        mapping_writer = csv.writer(mapping_file)
        mapping_writer.writerow(['roi_filename', 'original_image', 'quadrant', 'top', 'bottom', 'left', 'right'])

        for filename in os.listdir(args.image_dir):
            if filename.endswith(".png"):
                image_path = os.path.join(args.image_dir, filename)
                base_name = filename.replace('_0000.png', '.png')  # 去除 '_0000' 后缀
                mask_path = os.path.join(args.mask_dir, base_name)

                if os.path.exists(mask_path):
                    process_image_and_mask(image_path, mask_path, args.output_dir, mapping_writer, tmp_dir, padding=args.padding)
                else:
                    print(f"未找到掩码文件：{mask_path}")

if __name__ == "__main__":
    main()
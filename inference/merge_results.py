import os
import numpy as np
from PIL import Image
import argparse
import csv

def pixel_mapping(quadrant, value):
    mappings = {
        1: {0: 0, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18, 9: 51, 10: 52, 11: 53, 12: 54, 13: 55},
        2: {0: 0, 1: 21, 2: 22, 3: 23, 4: 24, 5: 25, 6: 26, 7: 27, 8: 28, 9: 61, 10: 62, 11: 63, 12: 64, 13: 65},
        3: {0: 0, 1: 31, 2: 32, 3: 33, 4: 34, 5: 35, 6: 36, 7: 37, 8: 38, 9: 71, 10: 72, 11: 73, 12: 74, 13: 75},
        4: {0: 0, 1: 41, 2: 42, 3: 43, 4: 44, 5: 45, 6: 46, 7: 47, 8: 48, 9: 81, 10: 82, 11: 83, 12: 84, 13: 85}
    }
    return mappings[quadrant].get(value, 0)

def merge_segmentations(mapping_file_path, teeth_infer_dir, output_dir):
    with open(mapping_file_path, mode='r') as mapping_file:
        print("mapping_file_path", mapping_file_path)
        mapping_reader = csv.DictReader(mapping_file)
        image_dict = {}

        for row in mapping_reader:
            roi_filename = row['roi_filename']
            original_image = row['original_image']
            top = int(row['top'])
            bottom = int(row['bottom'])
            left = int(row['left'])
            right = int(row['right'])

            corrected_filename = roi_filename.replace('_0000.png', '.png')
            print("corrected_filename", corrected_filename)
            quadrant = int(corrected_filename[-5])
            seg_path = os.path.join(teeth_infer_dir, corrected_filename)
            if not os.path.exists(seg_path):
                print(f"未找到分割结果：{seg_path}")
                continue

            seg_image = Image.open(seg_path)
            seg_array = np.array(seg_image)

            unique_values_before = np.unique(seg_array)
            print(f"象限 {quadrant} 合并前像素值种类：{len(unique_values_before)}")
            print(unique_values_before)

            # 使用新的映射函数
            vectorized_mapping = np.vectorize(lambda x: pixel_mapping(quadrant, x))
            seg_array = vectorized_mapping(seg_array)

            if original_image not in image_dict:
                original_image_path = os.path.join(args.original_image_dir, original_image) + ".png"
                print("original_image_path", original_image_path)
                original_img = Image.open(original_image_path)
                width, height = original_img.size
                final_segmentation = np.zeros((height, width), dtype=np.uint8)
                image_dict[original_image] = final_segmentation
            else:
                final_segmentation = image_dict[original_image]

            existing_segmentation = final_segmentation[top:bottom+1, left:right+1]
            combined_segmentation = np.maximum(existing_segmentation, seg_array)
            final_segmentation[top:bottom+1, left:right+1] = combined_segmentation

            image_dict[original_image] = final_segmentation

        for original_image, final_segmentation in image_dict.items():
            base_name, ext = os.path.splitext(original_image)
            if ext == '':
                ext = '.png'
            output_filename = base_name + ext
            output_path = os.path.join(output_dir, output_filename)
            Image.fromarray(final_segmentation).save(output_path)

            unique_values_after = np.unique(final_segmentation)
            print(f"{original_image} 合并后像素值种类：{len(unique_values_after)}")
            print(unique_values_after)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_file', required=True, help='ROI映射文件路径')
    parser.add_argument('--teeth_infer_dir', required=True, help='牙齿分割结果目录')
    parser.add_argument('--output_dir', required=True, help='最终分割结果输出目录')
    parser.add_argument('--original_image_dir', required=True, help='原始图像目录')
    global args
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    merge_segmentations(args.mapping_file, args.teeth_infer_dir, args.output_dir)

if __name__ == "__main__":
    main()
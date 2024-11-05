import os
import numpy as np
from PIL import Image

def fdi_to_quadrant(fdi):
    if 11 <= fdi <= 18 or 51 <= fdi <= 55:
        return 1
    elif 21 <= fdi <= 28 or 61 <= fdi <= 65:
        return 2
    elif 31 <= fdi <= 38 or 71 <= fdi <= 75:
        return 3
    elif 41 <= fdi <= 48 or 81 <= fdi <= 85:
        return 4
    else:
        return 0  # 对于不在FDI范围内的值，返回0

def convert_fdi_to_quadrant(directory):
    converted_count = 0
    total_count = 0

    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            total_count += 1
            filepath = os.path.join(directory, filename)
            
            with Image.open(filepath) as img:
                # 确保图像是RGB模式
                if img.mode != 'RGB':
                    print(f"警告: {filename} 不是RGB模式，跳过")
                    continue

                # 将图像转换为numpy数组
                img_array = np.array(img)

                # 应用转换函数到每个像素
                vectorized_func = np.vectorize(fdi_to_quadrant)
                quadrant_array = vectorized_func(img_array[:,:,0])

                # 创建新的RGB图像，所有通道值相同
                new_img_array = np.stack((quadrant_array,)*3, axis=-1)

                # 创建新图像并保存
                new_img = Image.fromarray(new_img_array.astype('uint8'), 'RGB')
                new_img.save(filepath)
                converted_count += 1
                print(f"转换完成: {filename}")

    return converted_count, total_count

def main():
    directory = r'F:\dataset\miccaiSTS\Dataset445\labelsTr'

    if os.path.exists(directory):
        print(f"开始处理目录: {directory}")
        converted_count, total_count = convert_fdi_to_quadrant(directory)
        print(f"\n转换完成")
        print(f"总图像数: {total_count}")
        print(f"成功转换的图像数: {converted_count}")
    else:
        print(f"错误: 目录不存在 - {directory}")

if __name__ == "__main__":
    main()
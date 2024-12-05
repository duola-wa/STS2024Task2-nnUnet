import os
import SimpleITK as sitk

def remap_labels(file_path):
    # 读取NIfTI文件
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)

    # 重新映射标签
    array[(array >= 1) & (array <= 8)] = 1
    array[(array >= 9) & (array <= 16)] = 2
    array[(array >= 17) & (array <= 24)] = 3
    array[(array >= 25) & (array <= 32)] = 4

    # 创建新的SimpleITK图像
    new_image = sitk.GetImageFromArray(array)
    new_image.CopyInformation(image)  # 复制原始图像的元数据

    # 直接覆盖原文件
    sitk.WriteImage(new_image, file_path)

def process_directory(directory):
    # 处理目录中的所有文件
    for filename in os.listdir(directory):
        if filename.startswith("PreTrain_") and filename.endswith(".nii.gz"):
            file_path = os.path.join(directory, filename)
            remap_labels(file_path)
            print(f"Processed: {filename}")

# 设置目录
directory = r"F:\dataset\miccaiSTS\Task311\labelsTr"

# 处理文件
process_directory(directory)
print("All files have been processed.")

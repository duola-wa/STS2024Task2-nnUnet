import os
import numpy as np
import SimpleITK as sitk
import sys
sys.path.append('/opt/algorithm')
from other_utils import Resample  # 确保 utils 文件中有此函数

def adjust_size(source_array, target_shape):
    """
    调整 source_array 的大小以匹配 target_shape。
    如果形状不匹配，则进行裁剪或填充。
    """
    source_shape = source_array.shape
    result = np.zeros(target_shape)
    min_shape = [min(source_dim, target_dim) for source_dim, target_dim in zip(source_shape, target_shape)]
    slices = tuple(slice(0, dim) for dim in min_shape)
    result[slices] = source_array[slices]
    return result

def quadrant_merge(image_path, resample_path, quadrant_mask_path, save_test_resizer_path, mask_path):
    for case in os.listdir(save_test_resizer_path):
        # 读取切割信息文件
        directions = np.load(os.path.join(save_test_resizer_path, case), allow_pickle=True)
        result = dict(directions.tolist())
        image = sitk.ReadImage(os.path.join(image_path, case.replace('.npy', '.nii.gz')))
        image_resample = sitk.ReadImage(os.path.join(resample_path, case.replace('.npy', '.nii.gz')))
        Origin = image_resample.GetOrigin()
        Spacing = image_resample.GetSpacing()
        Direction = image_resample.GetDirection()
        image_array = sitk.GetArrayFromImage(image_resample)

        mask_copy_array = np.zeros_like(image_array)

        # 使用range(1, 5)来确保处理4个象限
        for idx in range(1, 5):
            # 查找 result 中对应象限的数据
            key = f'{case.replace(".npy", f"_quadrant_{idx}_0000.nii.gz")}'

            # 调整象限文件名
            quadrant_file = os.path.join(quadrant_mask_path, case.replace('.npy', f'_quadrant_{idx}.nii.gz'))
            print(f"Looking for quadrant file: {quadrant_file}")

            # 如果象限文件不存在，则跳过该象限
            if not os.path.exists(quadrant_file):
                print(f"Quadrant {idx} for case {case} does not exist. Skipping.")
                continue

            print(f"Processing quadrant {idx} for case {case}")

            mask_array = np.zeros_like(image_array)
            quadrant_mask = sitk.ReadImage(quadrant_file)
            quadrant_mask_array = sitk.GetArrayFromImage(quadrant_mask)

            # 输出当前象限的mask值
            print(f"Before merging, unique mask values in quadrant {idx}: {np.unique(quadrant_mask_array)}")

            # 将 'stuff' 类（9）处理为背景
            quadrant_mask_array[quadrant_mask_array == 9] = 0
            quadrant_mask_array[quadrant_mask_array == 15] = 0

            quadrant_mask_array_copy = np.zeros_like(quadrant_mask_array)

            # 定义FDI编号的映射关系
            fdi_mapping = {
                1: list(range(21, 29)),  # 原第2象限，现在是第1象限
                2: list(range(11, 19)),  # 原第1象限，现在是第2象限
                3: list(range(41, 49)),  # 原第4象限，现在是第3象限
                4: list(range(31, 39)),  # 原第3象限，现在是第4象限
            }

            # 替换原来的编号生成逻辑
            for i in np.unique(quadrant_mask_array):
                if i == 0:
                    continue
                # 使用映射表分配FDI编号
                quadrant_mask_array_copy[quadrant_mask_array == i] = fdi_mapping[idx][i - 1]

            # 确保 mask_array 的大小与 value 对应的大小一致
            if key not in result:
                print(f"Quadrant {idx} for case {case} does not exist in the result dictionary. Skipping.")
                continue

            value = result[key]
            quadrant_mask_array_copy = adjust_size(quadrant_mask_array_copy, mask_array[value].shape)

            # 输出调整后象限的mask值
            print(f"After FDI mapping and resizing, unique mask values in quadrant {idx}: {np.unique(quadrant_mask_array_copy)}")

            mask_array[value] = quadrant_mask_array_copy
            mask_copy_array += mask_array

        # 输出合并后的mask值
        mask_copy_array[mask_copy_array >= 49] = 0
        print(f"After merging, unique mask values in the entire case: {np.unique(mask_copy_array)}")

        mask = sitk.GetImageFromArray(mask_copy_array)

        # 处理不同的空间分辨率
        if image.GetSpacing() != image_resample.GetSpacing():
            mask.SetDirection(Direction)
            mask.SetOrigin(Origin)
            mask.SetSpacing(Spacing)
            mask = Resample(mask, image.GetSpacing(), True, image.GetSize())
        else:
            mask.SetDirection(Direction)
            mask.SetOrigin(Origin)
            mask.SetSpacing(Spacing)

        sitk.WriteImage(mask, os.path.join(mask_path, case.rsplit('.', 1)[0][:-5] + '_Mask.nii.gz'))



if __name__ == "__main__":
    # Example usage
    image_path = "/nnUNet_inputs/"  # 原始图像目录
    resample_path = "/nnUNet_inputs/"  # 重采样后的图像目录
    quadrant_mask_path = "/opt/algorithm/output_tooth_segmentation/"  # 象限内牙齿分割结果目录
    save_test_resizer_path = "/opt/algorithm/quadrant_resizer/"  # 切割信息存储目录
    mask_path = "/outputs/"  # 最终结果存储目录

    quadrant_merge(image_path, resample_path, quadrant_mask_path, save_test_resizer_path, mask_path)

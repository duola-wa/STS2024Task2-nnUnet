import SimpleITK as sitk
import numpy as np
import os
from skimage.morphology import remove_small_objects


def process_masks(mask_path, volume_threshold=50):
    for mask_file in os.listdir(mask_path):
        if mask_file.endswith('.nii.gz'):
            mask_file_path = os.path.join(mask_path, mask_file)
            mask = sitk.ReadImage(mask_file_path)
            mask_array = sitk.GetArrayFromImage(mask)
            spacing = mask.GetSpacing()
            voxel_volume = np.prod(spacing)

            # 计算最小连通域体素数量
            min_size_voxels = volume_threshold / voxel_volume
            min_size_voxels = int(np.round(min_size_voxels))
            print(f"Processing {mask_file}, min_size_voxels={min_size_voxels}")

            # 确保掩膜数组为整数类型
            if not np.issubdtype(mask_array.dtype, np.integer):
                mask_array = mask_array.astype(np.int32)

            # 使用 skimage 的 remove_small_objects 函数
            new_mask_array = remove_small_objects(mask_array, min_size=min_size_voxels)

            # 将结果转换回 SimpleITK 图像
            new_mask = sitk.GetImageFromArray(new_mask_array)
            new_mask.CopyInformation(mask)

            # 保存处理后的掩膜
            sitk.WriteImage(new_mask, mask_file_path)
            print(f"Processed and updated mask: {mask_file_path}")


if __name__ == '__main__':
    mask_path = "/opt/algorithm/output_quadrant_segmentation/"
    volume_threshold = 50  # 体积阈值（物理单位，例如 mm^3）

    process_masks(mask_path, volume_threshold)

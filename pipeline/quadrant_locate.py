import os
import numpy as np
import SimpleITK as sitk


def quadrant_locate(data_dir, quadrant_dir, save_resizer_path, save_quadrant_crop_path):
    if not os.path.exists(save_resizer_path):
        os.makedirs(save_resizer_path)
    if not os.path.exists(save_quadrant_crop_path):
        os.makedirs(save_quadrant_crop_path)

    for case in os.listdir(quadrant_dir):
        if not (case.endswith('.nii') or case.endswith('.nii.gz')):
            continue
        print(case)
        case_mask = case
        case_name_base = case.split('.nii')[0]
        case = f"{case_name_base}_0000.nii.gz"
        data = sitk.ReadImage(os.path.join(data_dir, case))
        data_array = sitk.GetArrayFromImage(data)

        mask_quadrant = sitk.ReadImage(os.path.join(quadrant_dir, case_mask))
        mask_quadrant_array = sitk.GetArrayFromImage(mask_quadrant)

        exclusion_threshold = int(mask_quadrant_array.shape[0] * 0.15)
        mask_quadrant_array[:exclusion_threshold] = 0

        if(mask_quadrant_array.shape[1] > 500):
            exclusion_threshold_y = int(mask_quadrant_array.shape[1] * 0.80)  # 使用80%
            mask_quadrant_array[:, exclusion_threshold_y:, :] = 0

        resizer_dict = dict()

        for i in range(1, 5):
            mask_voxel_coords = np.where(mask_quadrant_array == i)

            # 检查是否找到对应象限的牙齿区域
            if len(mask_voxel_coords[0]) == 0:
                print(f"No voxels found for quadrant {i} in case {case}. Skipping.")
                continue

            minzidx = int(np.min(mask_voxel_coords[0]))
            maxzidx = int(np.max(mask_voxel_coords[0])) + 1
            minxidx = int(np.min(mask_voxel_coords[1]))
            maxxidx = int(np.max(mask_voxel_coords[1])) + 1
            minyidx = int(np.min(mask_voxel_coords[2]))
            maxyidx = int(np.max(mask_voxel_coords[2])) + 1
            center_z = int((minzidx + maxzidx) / 2)
            center_x = int((minxidx + maxxidx) / 2)
            center_y = int((minyidx + maxyidx) / 2)
            diameter_z = maxzidx - minzidx
            diameter_x = maxxidx - minxidx
            diameter_y = maxyidx - minyidx

            for w in [100]:
                minzidx = int(center_z - 0.5 * diameter_z * w / 100) - 2
                maxzidx = int(center_z + 0.5 * diameter_z * w / 100) + 2
                minzidx = max(0, minzidx)
                maxzidx = min(mask_quadrant_array.shape[0] - 1, maxzidx)

                minxidx = int(center_x - 0.5 * diameter_x * w / 100) - 2
                maxxidx = int(center_x + 0.5 * diameter_x * w / 100) + 2
                minxidx = max(0, minxidx)
                maxxidx = min(mask_quadrant_array.shape[1] - 1, maxxidx)

                minyidx = int(center_y - 0.5 * diameter_y * w / 100) - 2
                maxyidx = int(center_y + 0.5 * diameter_y * w / 100) + 2
                minyidx = max(0, minyidx)
                maxyidx = min(mask_quadrant_array.shape[2] - 1, maxyidx)

                bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

                resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
                print(case.replace('.nii', f'_quadrant_{i}_0000.nii'), resizer)
                resizer_dict[case.replace('.nii', f'_quadrant_{i}_0000.nii')] = resizer

            data_crop_array = data_array[resizer]

            data_crop = sitk.GetImageFromArray(data_crop_array)
            data_crop.SetOrigin(data.GetOrigin())
            data_crop.SetSpacing(data.GetSpacing())
            data_crop.SetDirection(data.GetDirection())
            sitk.WriteImage(data_crop,
                            os.path.join(save_quadrant_crop_path, case.replace('.nii', f'_quadrant_{i}_0000.nii')))
        np.save(os.path.join(save_resizer_path, case.replace('.nii.gz', '.npy')), resizer_dict)


if __name__ == "__main__":
    # 设置路径
    data_dir = "/nnUNet_inputs/"  # 原始图像目录
    quadrant_dir = "/opt/algorithm/output_quadrant_segmentation/"  # 象限分割结果目录
    save_resizer_path = "/opt/algorithm/quadrant_resizer/"  # 存储切割信息的目录
    save_quadrant_crop_path = "/opt/algorithm/quadrant_cropped_images/"  # 存储切割后图像的目录

    quadrant_locate(data_dir, quadrant_dir, save_resizer_path, save_quadrant_crop_path)

import multiprocessing
import shutil
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes


def load_and_covnert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    seg = io.imread(input_seg)
    print(f"Shape of seg: {seg.shape}")
    #seg[seg == 255] = 1
    image = io.imread(input_image)
    image = image.sum(2)
    mask = image == (3 * 255)
    # the dataset has large white areas in which road segmentations can exist but no image information is available.
    # Remove the road label in these areas
    mask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
                                                                         sizes[j] > min_component_size])
    mask = binary_fill_holes(mask)
    #seg[mask] = 0
    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)


if __name__ == "__main__":
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = '/media/ps/PS10T/changkai/v2nnunet/nnUNet-master/nnUnetFrame/2dData/462'
    nnUNet_raw = '/media/ps/PS10T/changkai/v2nnunet/nnUNet-master/nnUnetFrame/nnUNet_raw'
    dataset_name = 'Dataset462_ToothSeg2-cn5'


    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    train_source = join(source, 'train')

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(train_source, 'mask'), join=False, suffix='png')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(train_source, 'images', v.replace('_mask', '')),
                         join(train_source, 'mask', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v),
                         50
                     ),)
                )
            )
        _ = [i.get() for i in r]

# 完整的标签映射：背景，恒牙1-32，乳牙A-T
    label_mapping = {
        'background': 0,    # 背景
        'Tooth_1': 1,       # 恒牙1
        'Tooth_2': 2,       # 恒牙2
        'Tooth_3': 3,       # 恒牙3
        'Tooth_4': 4,
        'Tooth_5': 5,       # 恒牙1
        'Tooth_6': 6,       # 恒牙2
        'Tooth_7': 7,       # 恒牙3
        'Tooth_8': 8,
        'Tooth_9': 9,       # 恒牙1
        'Tooth_10': 10,       # 恒牙2
        'Tooth_11': 11,       # 恒牙3
        'Tooth_12': 12,
        'Tooth_13': 13,       # 恒牙1
        'Tooth_14': 14       # 恒牙2
    }

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, label_mapping,
                          num_train, '.png', dataset_name=dataset_name)

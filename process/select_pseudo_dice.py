import os
import numpy as np
import nibabel as nib
from collections import defaultdict
import time

def compute_dice_coefficient(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in np.unique(gt):
        if i == 0:
            continue
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)

def process_file(gt_file, pred_files):
    sample_number = os.path.basename(gt_file).split('_')[1]
    results = {}
    
    print(f"Processing sample {sample_number}...")
    
    try:
        gt = nib.load(gt_file).get_fdata()
    except Exception as e:
        print(f"  ERROR: Failed to load ground truth file {gt_file}: {str(e)}")
        return sample_number, results
    
    for epoch, pred_file in pred_files.items():
        pred_number = os.path.basename(pred_file).split('_')[1]
        if sample_number != pred_number:
            print(f"  WARNING: Sample number mismatch for epoch {epoch}. GT: {sample_number}, Pred: {pred_number}")
            results[epoch] = 0  # Set Dice score to 0 for mismatches
            print(f"  Epoch {epoch}: Dice = 0.0000 (due to number mismatch)")
            continue
        
        try:
            pred = nib.load(pred_file).get_fdata()
            dice = compute_multi_class_dsc(gt, pred)
            results[epoch] = dice
            print(f"  Epoch {epoch}: Dice = {dice:.4f}")
        except Exception as e:
            print(f"  ERROR: Failed to process prediction file for epoch {epoch}: {str(e)}")
            results[epoch] = 0  # Set Dice score to 0 for any processing errors
            print(f"  Epoch {epoch}: Dice = 0.0000 (due to processing error)")
    
    print(f"Completed processing sample {sample_number}")
    print("-------------------------------")
    return sample_number, results

def main():
    start_time = time.time()
    base_path = "/media/ps/PS10T/changkai/nnUNet-nnunetv1/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task310_TooSeg1"
    epochs = ['100', '200', '299']
    
    gt_folder = os.path.join(base_path, 'best')
    gt_files = [os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('_Mask.nii.gz')]
    
    print(f"Total number of samples: {len(gt_files)}")
    
    epoch_folders = {epoch: os.path.join(base_path, epoch) for epoch in epochs}
    
    results = defaultdict(dict)
    for i, gt_file in enumerate(gt_files, 1):
        pred_files = {epoch: os.path.join(folder, os.path.basename(gt_file)) 
                      for epoch, folder in epoch_folders.items()}
        sample_number, sample_results = process_file(gt_file, pred_files)
        results[sample_number] = sample_results
        print(f"Processed {i}/{len(gt_files)} samples")
        print("==============================")
    
    # Calculate average Dice scores
    avg_scores = {sample: np.mean(list(epochs.values())) for sample, epochs in results.items() if epochs}
    
    # Select top 30 samples
    top_30 = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:30]
    
    # Print results
    print("\nTop 30 samples:")
    for i, (sample, score) in enumerate(top_30, 1):
        print(f"{i}. Sample {sample}: Average Dice = {score:.4f}")
    
    # Print average Dice for each epoch
    print("\nAverage Dice coefficients for each epoch:")
    for epoch in epochs:
        epoch_scores = [results[sample][epoch] for sample in results if epoch in results[sample]]
        if epoch_scores:
            print(f"Epoch {epoch}: {np.mean(epoch_scores):.4f}")
        else:
            print(f"Epoch {epoch}: No valid scores")

    # Select pseudo-labels (Dice > 0.9)
    pseudo_labels = [sample for sample, score in top_30 if score > 0.9]
    print(f"\nNumber of selected pseudo-labels: {len(pseudo_labels)}")
    print("Selected pseudo-labels:")
    for sample in pseudo_labels:
        print(f"Sample {sample}")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
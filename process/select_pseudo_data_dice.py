import os
import numpy as np
from PIL import Image
from collections import defaultdict
import time

def compute_dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def compute_instance_dice(gt, pred):
    dice_scores = []
    unique_labels = np.unique(gt)
    unique_labels = unique_labels[unique_labels != 0]  # 排除背景

    for label in unique_labels:
        gt_instance = (gt == label).astype(int)
        pred_instance = (pred == label).astype(int)
        dice = compute_dice_coefficient(gt_instance, pred_instance)
        dice_scores.append(dice)

    return np.mean(dice_scores) if dice_scores else 0

def load_image(file_path):
    return np.array(Image.open(file_path))

def has_deciduous_teeth(image):
    return np.any(image > 50)

def count_teeth(mask):
    return len(np.unique(mask)) - 1  # Subtract 1 to exclude background

def process_files(gt_folder, pred_folders):
    dice_scores = defaultdict(list)
    total_files = len([f for f in os.listdir(gt_folder) if f.endswith('_mask_0000.png')])
    processed_files = 0
    
    start_time = time.time()
    for file in sorted(os.listdir(gt_folder)):
        if file.endswith('_mask_0000.png'):
            gt_path = os.path.join(gt_folder, file)
            gt_mask = load_image(gt_path)
            
            sample_scores = {}
            for epoch, folder in pred_folders.items():
                pred_path = os.path.join(folder, file)
                if os.path.exists(pred_path):
                    pred_mask = load_image(pred_path)
                    dice = compute_instance_dice(gt_mask, pred_mask)
                    has_deciduous = has_deciduous_teeth(gt_mask)
                    dice_scores[epoch].append((file, dice, has_deciduous))
                    sample_scores[epoch] = dice
            
            avg_score = np.mean(list(sample_scores.values()))
            print(f"File: {file}")
            print(f"Has deciduous teeth: {has_deciduous}")
            for epoch, score in sample_scores.items():
                print(f"  Epoch {epoch}: Dice = {score:.4f}")
            print(f"  Average Dice: {avg_score:.4f}")
            print("--------------------")
            
            processed_files += 1
            if processed_files % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {processed_files}/{total_files} files. Elapsed time: {elapsed_time:.2f} seconds")
                print("====================")
    
    return dice_scores

def select_pseudo_labels(dice_scores):
    avg_scores = {epoch: np.mean([score for _, score, _ in scores]) for epoch, scores in dice_scores.items()}
    best_epoch = max(avg_scores, key=avg_scores.get)
    
    # Select samples with Dice score greater than 0.9
    pseudo_labels_deciduous = [(file, score) for file, score, has_deciduous in dice_scores[best_epoch] if has_deciduous and score > 0.94]
    pseudo_labels_permanent = [(file, score, count_teeth(load_image(os.path.join(gt_folder, file)))) 
                               for file, score, has_deciduous in dice_scores[best_epoch] 
                               if not has_deciduous and score > 0.9]
    
    # Sort permanent teeth samples by score
    pseudo_labels_permanent.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate the number of samples to keep for each category of permanent teeth
    n_deciduous = len(pseudo_labels_deciduous)
    n_permanent_large = int(n_deciduous * 0.8)
    n_permanent_medium = int(n_deciduous * 0.3)
    n_permanent_small = int(n_deciduous * 0.1)
    
    # Categorize and select permanent teeth samples
    permanent_large = [x for x in pseudo_labels_permanent if x[2] > 20][:n_permanent_large]
    permanent_medium = [x for x in pseudo_labels_permanent if 10 < x[2] <= 20][:n_permanent_medium]
    permanent_small = [x for x in pseudo_labels_permanent if x[2] <= 10][:n_permanent_small]
    
    # Combine all selected samples
    selected_permanent = permanent_large + permanent_medium + permanent_small
    
    return best_epoch, avg_scores, pseudo_labels_deciduous, selected_permanent

def main():
    base_path = "/home/customer/changkai/nnUNet-master/nnUnetFrame/nnUNet_raw/Dataset444_Sts"
    global gt_folder
    gt_folder = os.path.join(base_path, "best")
    pred_folders = {
        "50": os.path.join(base_path, "50"),
        "100": os.path.join(base_path, "100"),
        "150": os.path.join(base_path, "150")
    }
    
    print("Starting file processing...")
    dice_scores = process_files(gt_folder, pred_folders)
    print("File processing completed. Selecting pseudo-labels...")
    best_epoch, avg_scores, pseudo_labels_deciduous, pseudo_labels_permanent = select_pseudo_labels(dice_scores)
    
    print("\nAverage Dice scores for each epoch:")
    for epoch, avg_score in avg_scores.items():
        print(f"Epoch {epoch}: {avg_score:.4f}")

    # Write selected file names to selected_file.txt
    with open("./selected_file.txt", "w") as f:
        for file, _ in pseudo_labels_deciduous:
            f.write(f"{file}\n")
        for file, _, _ in pseudo_labels_permanent:
            f.write(f"{file}\n")
    
    print("\nSelected pseudo-labels (deciduous):")
    for file, score in pseudo_labels_deciduous:
        print(f"{file}: Instance Dice score = {score:.4f}")
    
    print("\nSelected pseudo-labels (permanent):")
    for file, score, tooth_count in pseudo_labels_permanent:
        print(f"{file}: Instance Dice score = {score:.4f}, Tooth count = {tooth_count}")

    print(f"\nBest performing model: {best_epoch} epochs")
    print(f"Number of selected pseudo-labels (deciduous): {len(pseudo_labels_deciduous)}")
    print(f"Number of selected pseudo-labels (permanent): {len(pseudo_labels_permanent)}")
    print(f"Total number of selected pseudo-labels: {len(pseudo_labels_deciduous) + len(pseudo_labels_permanent)}")
    print(f"Selected file names have been written to ./selected_file.txt")

if __name__ == "__main__":
    main()
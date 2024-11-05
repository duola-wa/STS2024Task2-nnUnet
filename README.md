# nnUnet retraining for MICCAI STS 2024 Challenge Task 1: Tooth Instance Segmentation in 2D Panoramic X-ray Images

## Environments and Requirements:
### 1. nnUNet Configuration
Install nnU-Net as below.   
For more details, please refer to https://github.com/MIC-DKFZ/nnUNet  
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
### 2. Pipeline of the Proposed Framework
#### 2.1. Dataset Load and Reconstruction
Load ToothFairy Dataset from https://www.codabench.org/competitions/3024/#/pages-tab

#### 2.2. Model Training from the Preprocessed Dataset
First of all, add codes in line 486 of nnUNet-master/nnunet/training/network_training/network_trainer.py to save the checkpoints over 1/3, 2/3, 3/3 total iterations during training
```
if (self.epoch + 1) in [int(1 / 3 * self.max_num_epochs), int(2 / 3 * self.max_num_epochs),int(self.max_num_epochs-1)]:
self.save_checkpoint(join(self.output_folder, str(self.epoch + 1) + '.model'))
```

Conduct automatic preprocessing using nnUNet and train the Teacher(or Student) Model.
```
nnUNet_plan_and_preprocess -t 100 --verify_dataset_integrity
nnUNet_train 3d_fullres nnUNetTrainerV2 100 all
```
#### 2.2. Selective Re-training Strategy
Do Inference with the model which owns saved checkpoint weights.
```
sh process/nnUNet_Pseudo_Generate.sh
```
Regard the top 100 images with the highest score and the number of connected domains as 2 with a meanDice score greater than 0.9 as reliable images, and the rest as unreliable images.
```
python process/select_pseudo_dice_image.py
python process/select_pseudo_dice_instance.py
```
After that, we use the original labeled data and pseudo-labeled data to jointly supervise the training of the student model, and update the student model to a new iteration of the teacher model.  
In the challenge, we first reconstruct the training set as 111 labeled images and 332
unlabeled images.   
Then, we performed two pseudo-label update iterations, training the final model with 111
labeled data and 200 pseudo-labeled data.


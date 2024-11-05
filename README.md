# 针对MICCAI STS 2024 Challenge Task 1: Tooth Instance Segmentation in 2D Panoramic X-ray Images的nnUnet再训练的方案

## Environments and Requirements:
### 1. nnUNet 配置
安装Install.   
For more details, please refer to https://github.com/MIC-DKFZ/nnUNet  
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
### 2. Pipeline of the Proposed Framework
#### 2.1. Dataset Load
Load STS Dataset from https://www.codabench.org/competitions/3024/#/pages-tab

#### 2.2. Model Training from the Preprocessed Dataset

第一步先训练象限分割模型
首先把FDI编号转化为象限编号
```
python process/FDI2Qua.py
```
把2d数据转换成nnUnet可以训练的形式
```
python process/Dataset411_Ruyastage0.py
```
训练模型
```
nnUNetv2_plan_and_preprocess -d 411 --verify_dataset_integrity
nnUNetv2_train 411 2d all --npz -tr nnUNetTrainerNoDA
```

第二步训练象限内牙齿分割模型
更改配置文件，保存不同阶段的权重，方便后续筛选伪标签。
Add codes in line 1142 of nnUNet/nnunetv2/training/nnUNetTrainer/nnUetTrainer.py to save the checkpoints over 1/3, 2/3, 3/3 total iterations during training 150 epochs
```
self.save_checkpoint(join(self.output_folder, str(current_epoch + 1) + '.pth'))
```
把数据处理成第二阶段训练的形式。
```
python process/dataPrepareForSegInTooth.py        
```
Conduct automatic preprocessing using nnUNet and train the Teacher(or Student) Model.
```
nnUNetv2_plan_and_preprocess -d 412 --verify_dataset_integrity
nnUNetv2_train 412 2d all --npz
```
#### 2.3. Selective Re-training Strategy
Do Inference with the model which owns saved checkpoint weights.
```
sh inference/pipeline.sh
```
Regard the top 100 images with the highest score and the number of connected domains as 2 with a meanDice score greater than 0.9 as reliable images, and the rest as unreliable images.
```
python process/select_pseudo_dice.py
```
After that, we use the original labeled data and pseudo-labeled data to jointly supervise the training of the student model, and update the student model to a new iteration of the teacher model.  
Then, we performed two pseudo-label update iterations.



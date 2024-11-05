# 针对MICCAI STS 2024 Challenge Task 2的nnUnet再训练方案

### 1. nnUNet 配置
安装nnUnet
对于更多细节，请参考https://github.com/MIC-DKFZ/nnUNet  
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
### 2. Pipeline 
#### 2.1. 下载数据
下载 STS 数据集 https://www.codabench.org/competitions/3025/

#### 2.2. 模型训练
第一步先训练象限分割模型

把FDI编号转化为象限编号
```
python process/FDI2Qua.py
```

训练模型
```
nnUNet_plan_and_preprocess -t 313 --verify_dataset_integrity
nnUNet_train 3d_lowres nnUNetTrainerV2 313 all --npz 
```

第二步训练象限内牙齿分割模型

更改配置文件，保存不同阶段的权重，方便后续筛选伪标签。
在 nnUNet-master/nnunet/training/network_training/network_trainer.py 的第 486 行添加代码，以保存训练 300 个epoch期间 1/3、2/3、3/3 总迭代次数的checkpoint
```
if (self.epoch + 1) in [int(1 / 3 * self.max_num_epochs), int(2 / 3 * self.max_num_epochs),int(self.max_num_epochs-1)]:
  self.save_checkpoint(join(self.output_folder, str(self.epoch + 1) + '.model'))
```
把数据处理成第二阶段训练的形式，即把每个原始数据切分成四个象限的数据。
```
python process/preparefor2.py        
```
使用 nnUNet 进行自动预处理并训练教师（或学生）模型。
```
nnUNet_plan_and_preprocess -t 312 --verify_dataset_integrity
nnUNet_train 3d_fullres nnUNetTrainerV2 312 all --npz 
```
#### 2.3. 选择性再训练策略
使用已保存的检查点权重的模型进行推理。
```
sh inference/pipeline.sh
```
筛选需要保留的伪标签。
```
python process/select_pseudo_dice.py
```
之后，我们使用原始标记数据和伪标记数据联合监督学生模型的训练，并将学生模型更新为教师模型的新迭代。
然后，我们进行了两次伪标签更新迭代。


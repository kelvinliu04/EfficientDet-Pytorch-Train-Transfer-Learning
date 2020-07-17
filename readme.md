# Training EfficientDet Pytorch (Transfer Learning)
## Step
1. Installation
2. Dataset Preparation
3. Project Config 
4. Download Pre-trained model Efficientdet
5. Run train.py

## 1. Installation
Pytorch with CUDA


## 2. Dataset Preparation

- This model must be train on **COCO** - dataset format. 
- If you have **VOC PASCAL** dataset format, you can convert it using my repo [here](??????????).
- Put the dataset on "datasets" folder with name of your project as the folder name

**PLEASE CHECK THIS FOLDER CONFIGURATION**
```bash
|
|--datasets
    |--head_local
        |--annotations
            |--instances_train.json
            |--instances_val.json
        |--train
            |-- {img_train_1.jpg}
            .
        |--val
            |-- {img_val_1.jpg}
            .
```
*NB:
a. Name of project is head_local (change with yours)
b. Please change your annotations filenames with "instances_train.json" and "instances_val.json"

## 3. Project Config
After prepare the dataset, you need to create a config (.yml) file on "projects" folder.
```yaml
project_name: head_local 
train_set: train
val_set: val
num_gpus: 1

mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]

anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

obj_list: ['person']
```
*NB:
a. Project_name must be the same name with dataset's name folder
b. Mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
c. The anchor is adapted to the dataset (you can also leave it)
d. Change the object list to your objects should be detected

## 4. Download Pre-trained model Efficientdet
- You need to download the pre-trained model. Put it on "weights" folder.

**This is the comparison of the model for each different coefficient**

| Coefficient | Pth_download | GPU Mem(MB) | FPS | Extreme FPS (Batchsize 32) | mAP 0.5:0.95(this repo) | mAP 0.5:0.95(paper) |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| D0 | [efficientdet-d0.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth) | 1049 | 36.20 | 163.14 | 33.1 | 33.8
| D1 | [efficientdet-d1.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth) | 1159 | 29.69 | 63.08 | 38.8 | 39.6
| D2 | [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth) | 1321 | 26.50 | 40.99 | 42.1 | 43.0
| D3 | [efficientdet-d3.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth) | 1647 | 22.73 | - | 45.6 | 45.8
| D4 | [efficientdet-d4.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth) | 1903 | 14.75 | - | 48.8 | 49.4
| D5 | [efficientdet-d5.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth) | 2255 | 7.11 | - | 50.2 | 50.7
| D6 | [efficientdet-d6.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth) | 2985 | 5.30 | - | 50.7 | 51.7
| D7 | [efficientdet-d7.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d7.pth) | 3819 | 3.73 | - | 51.2 | 52.2
## 5. Run train.py
All Configurations of train.py
```python
'-p', '--project', type=str, default='coco', help='project file that contains parameters'
'-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet'
'-n', '--num_workers', type=int, default=12, help='num_workers of dataloader'
'--batch_size', type=int, default=12, help='The number of images per batch among all devices'
'--head_only', type=boolean_string, default=False help='whether finetunes only the regressor and the classifier, useful in early stage convergence or small/easy dataset'
'--lr', type=float, default=1e-4
'--optim', type=str, default='adamw', help='select optimizer for training, suggest using \'admaw\' until the very final stage then switch to \'sgd\''
'--num_epochs', type=int, default=500
'--val_interval', type=int, default=1, help='Number of epoches between valing phases'
'--save_interval', type=int, default=500, help='Number of steps between saving'
'--es_min_delta', type=float, default=0.0, help='Early stopping\'s parameter: minimum change loss to qualify as an improvement'
'--es_patience', type=int, default=0, help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.'
'-w', '--load_weights', type=str, default=None,help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint'
'--saved_path', type=str, default='logs/'
'--debug', type=boolean_string, default=False, help='whether visualize the predicted boxes of training, the output images will be in test/'
```
**Example usage**
Project name is head_local
```
train.py 
--compound_coef 7 
--project head_local 
--batch_size 2 
--lr 1e-5 
--num_epochs 10 
--optim adamw
--load_weights weights/efficientdet-d7.pth 
--head_only True
```
*nb: 
a. use lower batch_size if you have small RAM
b. change the weights into "last" for continue the training
c. change the optim to sgd to improve the model after trained using adamw
d. use head_only = true if you have small dataset (only train the classifier)

**Result of model would be on "logs" folder**
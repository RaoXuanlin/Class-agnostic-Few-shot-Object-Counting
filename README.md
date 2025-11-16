# Class agnostic Few shot Object Counting

This repository is the non-official pytorch implementation of a WACV 2021 Paper "Class-agnostic Few-shot-Object-Counting". [Link](https://openaccess.thecvf.com/content/WACV2021/papers/Yang_Class-Agnostic_Few-Shot_Object_Counting_WACV_2021_paper.pdf)

In Proc. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021
Shuo-Diao Yang, Hung-Ting Su, Winston H. Hsu, Wen-Chin Chen<sup>*</sup>

![39_ref_good](https://user-images.githubusercontent.com/76461262/181033357-71dc9a34-7a78-4410-81d6-4dc7d74bedfd.png) </br>
<img src="https://user-images.githubusercontent.com/76461262/181033299-cda225d3-c964-4327-9d13-bdbdaa296af3.png" width="200" height="150" /> <img src="https://user-images.githubusercontent.com/76461262/181033407-cb571edc-cb2f-4f1a-9127-fb74cafc933c.png" width="200" height="150" /> <img src="https://user-images.githubusercontent.com/76461262/181033455-84efbbea-0656-4e47-b281-34e3eeb14482.png" width="200" height="150" /> </br>

![2_ref_good](https://user-images.githubusercontent.com/76461262/181036373-715da5ae-e150-4980-92e7-3434146e40e8.png) </br>
<img src="https://user-images.githubusercontent.com/76461262/181036450-ee30acc9-1521-4dd1-b5c3-562308dc7f8d.png" width="200" height="150" /> <img src="https://user-images.githubusercontent.com/76461262/181036669-6d0b78b4-8447-4f8c-9ac6-821351bc4f0b.png" width="200" height="150" /> <img src="https://user-images.githubusercontent.com/76461262/181036720-a7539696-bd5f-4886-aff6-4a343d390276.png" width="200" height="150" /> </br>

## Installation
Our code has been implemented on Python 3.8 and PyTorch 1.8.1+cu101. Please follow the instructions to setup your environment. See other required packages in `requirements.txt`.
````
conda create --name CFOCNet python=3.8
conda activate CFOCNet
pip install -r requirements.txt
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
````
If you have problem about installing **cocoapi**, come [here](https://github.com/philferriere/cocoapi) to find the official documentation.
## Getting Started
* [CFOCNet_demo.ipynb](CFOCNet_demo.ipynb) This notebook tests the detail implementations of CFOCNet, giving insights such as how the size of each tensor changes across each stage.
* [model](model) This directory contains all related modules of our CFOCNet implementation
* [Eval_Result](Eval_Result) This directory contains the ideal results during evaluation stage, where an example's predicted count and the density map aligns with the groundtruth.
## Data Preparation

### Option 1: FSC-147 Dataset (Recommended for Few-Shot Counting)
We now support FSC-147 dataset, which is specifically designed for few-shot object counting. </br>

1. Download FSC-147 dataset from [here](https://github.com/cvlab-stonybrook/LearningToCountEverything)
2. The dataset structure should be:
````
$PATH_TO_FSC147/
├──── images_384_VarV2/
│    └──── 6135 images (.jpg)
│
├──── annotation_FSC147_384.json
├──── Train_Test_Val_FSC_147.json
└──── ImageClasses_FSC147.txt
````
3. Update the `data_path` in [config_fsc147.yaml](configs/config_fsc147.yaml) to point to your FSC-147 dataset directory
4. No preprocessing is required - FSC-147 provides exemplar boxes directly in the annotations

### Option 2: COCO Dataset 2017 (Original Implementation)
We train and evaluate our methods on COCO dataset 2017. </br>
Please follow the instruction [here](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9) to download the COCO dataset 2017 </br>
structure used in our code will be like : </br>
````
$PATH_TO_DATASET/
├──── images
│    ├──── train2017
│             |──── 118287 images (.jpg)
│
│    ├──── test2017
│             |──── 40670 images (.jpg)
│
│    ├──── val2017
│             |──── 5000 images (.jpg)
│
├──── annotations
│    ├──── captions_train2017.json
│
│    ├──── captions_val2017.json
│
│    ├──── instances_train2017.json
|
│    ├──── instances_val2017.json
│
│    ├──── person_keypoints_train2017.json
│
│    ├──── person_keypoints_val2017.json
````
After downloading the data, please navigate to our repository. </br>
Then, modify the variable "coco_path" in line 8  in [crop.py](data/crop.py) to your COCO dataset path.
````
cd CODE_DIRECTORY
python data/crop.py
````
After performing the above instructions, the structure of your coco dataset will be like : </br>
````
$PATH_TO_DATASET/
├──── images
│    ├──── train2017
│             |──── 118287 images (.jpg)
│
│    ├──── test2017
│             |──── 40670 images (.jpg)
│
│    ├──── val2017
│             |──── 5000 images (.jpg)
│
│    ├──── crop
│             |──── 80 directories which store all categories 500 images in coco dataset 2017 (for references images)
│
├──── annotations
│    ├──── captions_train2017.json
│
│    ├──── captions_val2017.json
│
│    ├──── crop.json
│
│    ├──── instances_train2017.json
|
│    ├──── instances_val2017.json
│
│    ├──── person_keypoints_train2017.json
│
│    ├──── person_keypoints_val2017.json

````

## Training

### Training with FSC-147 Dataset
* Use [config_fsc147.yaml](configs/config_fsc147.yaml) for FSC-147 dataset configuration
* Update the `data_path` in the config file to point to your FSC-147 dataset directory
* Model configurations such as **epochs**, **batch_size**, and **result_path** can be tuned in config_fsc147.yaml
* Run training with:
````
cd CODE_DIRECTORY
python main.py --config=config_fsc147.yaml --doc=fsc147_training --train
````
* After running the code, you will find your training logs under CODE_DIRECTORY/exp/logs/fsc147_training

### Training with COCO Dataset (Original)
* Use [config.yaml](configs/config.yaml) for COCO dataset configuration
* Update the `data_path` in config.yaml to your COCO dataset path
* Model configurations such as **epochs**, **batch_size**, and **result_path** can be tuned in config.yaml
* Run training with:
````
cd CODE_DIRECTORY
python main.py --config=config.yaml --doc=coco_training --train
````
* After running the code, you will find your training logs under CODE_DIRECTORY/exp/logs/coco_training

## Testing

### Testing with FSC-147 Dataset
* Update the `checkpoint` path in [config_fsc147.yaml](configs/config_fsc147.yaml)
* The testing will use the pre-defined test split from FSC-147 dataset
* Run testing with:
````
cd CODE_DIRECTORY
python main.py --config=config_fsc147.yaml --doc=fsc147_testing --test
````

### Testing with COCO Dataset (Original)
* Update the `checkpoint` path in [config.yaml](configs/config.yaml)
* Run testing with:
````
cd CODE_DIRECTORY
python main.py --config=config.yaml --doc=coco_testing --test
````
* After running the code, you will find your testing logs under "CODE_DIRECTORY/exp/logs/doc_name".

## Implementation Details
* The runner architecture is from [NCSNv2](https://github.com/ermongroup/ncsnv2).
* For query image, instead of random crop, we resize it with aspect ratio and padding to 256 x 256.

### COCO Dataset Implementation
* We crop 500 reference images for each categories from COCO training set.
* The default setting is 5-shot learning, where each query image has 5 reference images to learn.

### FSC-147 Dataset Implementation
* FSC-147 provides 3 exemplar bounding boxes per image as reference objects.
* The default setting is 3-shot learning (matching FSC-147's annotation format).
* No preprocessing step needed - exemplars are extracted directly from annotations during training.
* Uses point annotations to generate density maps for supervision. 

## Acknowledgement
* Great thanks to the contributive discussions on the reproduction details with </br>
the author of [Class-Agnostic Few-Shot Object Counting](https://openaccess.thecvf.com/content/WACV2021/html/Yang_Class-Agnostic_Few-Shot_Object_Counting_WACV_2021_paper.html), Shuo-Diao Yang, </br>
the author of [Bilinear Matching Network](https://arxiv.org/abs/2203.08354), Min Shi, </br>
and the author of [Learning to Count Anything: Reference-less Class-agnostic Counting with Weak Supervision](https://arxiv.org/abs/2205.10203), Michael Hobley. </br> Their feedbacks have been really helpful, allowing us to replicate the model successfully. 

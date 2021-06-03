# Using a DCT-Driven Loss in Attention-Based Knowledge-Distillation for Scene Recognition
Official Pytorch Implementation of Using a DCT-Driven Loss in Attention-Based Knowledge-Distillation for Scene Recognition by *Anonymous Authors* (In revision for NeurIPS 2021 Conference).

FIGURE OF THE PROPOSED APPROACH

## Setup

### Requirements
The repository has been developed and tested in the following software versions.
 - Ubuntu 16.04
 - Python 3.6
 - Anaconda 4.9
 - PyTorch 1.7
 - CUDA 10.1
 
> **Important**: Support for different requirements than the ones stated above will not be provided in the issues section, specially for those regarding lower versions of libraries or CUDA.
 
 ### Clone Repository
Clone repository running the following command:

	$ git clone GitHubLINK

### Anaconda Enviroment
To create and setup the Anaconda Envirmorent run the following terminal command from the repository folder:

    $ conda env create -f Config/environment.yml
    $ conda activate DCTKnowledgeDistillation
   
## Datasets
We provide the following scripts to download the necesary datasets. These script are intended to ease the process but it is filled with Authors's original links, any query regarding the datasets or links should be asked to them:

 - **ADE20K**
 
	   $ ./Scripts/download_ADE20K.sh

 - **SUN397**
 
	 	$ ./Scripts/download_SUN397.sh
 - **MIT67**
 
	 	$ ./Scripts/download_MIT67.sh
	
## Training
  
This section provides a few examples on how to train the models from the paper. Training is divided into Teachers, Baselines and Proposed DCT-based Knowledge Distillation. 

### Teachers
To train a teacher model(s), e.g. using ResNet-50 backbone, run this command:

	$ python trainCNNs.py --Architecture ResNet50 --Dataset ADE20K --Distillation None
>This command will train a vanilla ResNet-50 for the ADE20K dataset using the default parameters used in the paper.
>**Note**: Models are saved in a unique folder named by the date and hour (pretty long name). We recommend changing the folder name once trained to ease their usage as Teachers.

### Baselines
To train a baseline model(s), e.g. using ResNet-18 backbone, run this command:

	$ python trainCNNs.py --Architecture ResNet18 --Dataset ADE20K --Distillation None
>This command will train a vanilla ResNet-50 for the ADE20K dataset using the default parameters used in the paper.

### Proposed DCT-based Knowledge Distillation
To train the proposed method run this command:

	$ python trainCNNs.py --Architecture ResNet18 --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet50 ADE20K'
	
>This command will train the proposed DCT-based approach using ResNet-18 as Student backbone and `Teacher ResNet50 ADE20K` folder path as the Teacher. ADE20K dataset is used with the default parameters used in the paper.

### Changing Hyper-Parameters
Training parameters are specified in each of the separate `Config.yaml` files from [`Config`](https://github.com/alexlopezcifuentes/Distillation-Attention/tree/main/Config) folder.  To change them, you can either change the specific `Config.yaml` file or you can override any set of hyper-parameters with the argument `--Options` like:
	
	$ python trainCNNs.py --Architecture X --Dataset X --Distillation DFT --Options TEACHER='X' BS_TRAIN=50 LR=0.01 COMMENTS='Testing argument override'
> Take a look at each configuration file to see the options. If you want to change an hyper-parameter it is likely that is coded there.

### Automatic Script
We provide a shell script that contains the set of necesary commands to train the models from the paper in [`Scripts/Run_Train.sh`](https://github.com/alexlopezcifuentes/Distillation-Attention/blob/main/Scripts/Run_Train.sh). You can either check the script to inspect how the models where trained or run it from `Scripts/` folder with:

	$ ./Run_Train.sh 

## Evaluation

### Model Evaluation
Once a model is trained you can simply evaluate it with the following command:

	$ python evaluateCNNs.py --Model "<ModelPath>"
>Evaluation results will be promt in the terminal but also saved in summary files in the model's path.

### Activation Maps Extraction
We also provide a Python script to automatically extract the activation maps for a given model with:

	$ python extractAMs.py --Model "<ModelPath>"

### Automatic Script
As in training, we provide and automatic shell script to ease the model evaluations in [`Scripts/Run_Val.sh`](https://github.com/alexlopezcifuentes/Distillation-Attention/blob/main/Scripts/Run_Val.sh) that can be run with:

	$ ./Run_Val.sh 

## Results & Model Zoo
This Section provides a table with the results from the paper and the links to the models from the used Teachers, Baselines and Students.

### ADE20K

| Method         | T: ResNet-50 <br> S: ResNet-18 | T: ResNet-152 <br> S: ResNet-34 | T: ResNet-50 <br> S: MobileNet-V2 |
|----------------|--------------------------------|---------------------------------|-----------------------------------|
| Teacher        | [58.34](LINK)                          | [60.07](LINK)                           | [58.34](LINK)                             |
| Baseline       | [40.97](LINK)                          | [41.63](LINK)                           | [44.29](LINK)                             |
| AT             | 45.43                          | 44.80                           | 46.65                             |
| PKT            | 44.59                          | 42.38                           | 46.42                             |
| VID            | 42.09                          | 39.31                           | 43.63                             |
| CRD            | 45.46                          | 43.06                           | 46.51                             |
| CKD            | 46.89                          | 45.01                           | 47.30                             |
| **DCT (Ours)** | [**47.35**](LINK)                      | [**45.63**](LINK)                       | [**47.39**](LINK)                         |
| KD             | 50.54                          | 48.91                           | 48.37                             |
| AT + KD        | 48.87                          | 50.70                           | 47.67                             |
| PKT  + KD          | 49.31                          | 49.70                           | 49.43                             |
| VID + KD           | 50.36                          | 48.99                           | 47.84                             |
| CRD  + KD          | 48.90                          | 49.68                           | 47.88                             |
| CKD   + KD         | 52.10                          | **53.54**                          | 49.15                             |
| **DCT (Ours) + KD** | [**54.25**](LINK)                      | [52.68](LINK)                           | [**50.75**](LINK)      

### SUN397

| Method         | T: ResNet-50 <br> S: ResNet-18 | T: ResNet-152 <br> S: ResNet-34 | T: ResNet-50 <br> S: MobileNet-V2 |
|----------------|--------------------------------|---------------------------------|-----------------------------------|
| Teacher        | [61.69](LINK)                          | [62.56](LINK)                           | [51.59](LINK)                             |
| Baseline       | [38.77](LINK)                          | [44.66](LINK)                           | [41.18](LINK)                             |
| AT             | 41.82                          | 40.99                           | 38.84                             |
| PKT            | 38.70                          | 37.70                           | 40.17                             |
| VID            | 41.16                          | 38.16                           | 41.06                             |
| CRD            | 44.89                          | 41.80                           | **43.56**                             |
| **DCT (Ours)** | [**45.75**](LINK)                      | [**43.50**](LINK)                       | [43.16](LINK)                         |
| KD             | 48.83                          | 48.26                           | 47.31                             |
| AT + KD        | 49.69                          | 49.05                           | 46.60                             |
| PKT  + KD          | 49.13                          | 48.08                           | 47.54                             |
| VID + KD           | 49.26                          | 47.08                           | 46.57                             |
| CRD  + KD          | 49.79                          | 48.39                           | 46.77                             |
| **DCT (Ours) + KD** | [**55.15**](LINK)                      | [**50.51**](LINK)                           | [**49.25**](LINK)

### MIT67

| Method         | T: ResNet-50 <br> S: ResNet-18 | T: ResNet-152 <br> S: ResNet-34 | T: ResNet-50 <br> S: MobileNet-V2 |
|----------------|--------------------------------|---------------------------------|-----------------------------------|
| Teacher        | [77.32](LINK)                          | [78.11](LINK)                           | [77.32](LINK)                             |
| Baseline       | [49.26](LINK)                          | [38.84](LINK)                           | [49.06](LINK)                             |
| AT             | 51.73                          | 49.66                           | 45.13                             |
| PKT            | 51.03                          | 46.32                           | 50.23                             |
| VID            | 48.57                          | 44.22                           | 47.87                             |
| CRD            | 52.19                          | 42.41                           | 50.10                             |
| **DCT (Ours)** | [**56.32**](LINK)                      | [**52.14**](LINK)                       | [**50.42**](LINK)                         |
| KD             | 54.87                          | 51.55                           | 56.14                             |
| AT + KD        | 57.10                          | 53.05                           | 52.17                             |
| PKT  + KD          | 53.83                          | 50.52                           | 53.05                             |
| VID + KD           | 54.15                          | 51.79                           | 54.05                             |
| CRD  + KD          | 56.11                          | 52.34                           | 55.00                             |
| **DCT (Ours) + KD** | [**60.11**](LINK)                      | [**55.18**](LINK)                           | [**57.35**](LINK)

## Citation
If you find this code and work useful, please consider citing:
```
Citation
```

## Acknowledgments
This study has been partially supported by the *Anonymous* project.

We want also to give a massive thank to  [Yonglong Tian (HobbitLong)](https://github.com/HobbitLong) for sharing his [RepDistiller Repository](https://github.com/HobbitLong/RepDistiller) from where we took the implementations for the state-of-the-art methods. Give them a star!
# 6DRepNet: Head Pose Estimation
![Vizualization](https://github.com/Shohruh72/HPENet/blob/main/output/Result.gif)

## Features

* **6D Rotation Matrix Representation**
* **Geodesic Loss Function.**
* **High Performance.**
* **Easy Integration**
* **Customizability**
  
*** _The project is structured into four main files:_**

- `main.py`: The entry point of the project, orchestrating the training, evaluation, and prediction processes.
- `nets.py`: Contains the definitions of neural network models or architectures.
- `datasets.py`: Manages dataset handling, including loading, preprocessing, and augmentations.
- `util.py`: Provides utility functions for data manipulation, visualization, logging, and other support tasks.

## Performance Metrics

The model achieved the following Mean Absolute Error (MAE) metrics across different pose angles:

### Results

| Pitch | Yaw  | Roll |                                                            Pretrained weights |
|:-----:|:----:|-----:|------------------------------------------------------------------------------:|
| 4.92  | 3.72 | 3.44 | [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best.pt) |

## Installation

1. Clone the repository
2. Create a Conda environment using the environment.yml file:

```bash 
conda env create -f environment.yml
```

3. Activate the Conda environment:

```bash
conda activate HPE
```

## Preparing the Dataset

Our project can work with different datasets like 300W_LP, AFLW2000.

1. Download the 300W-LP, AFLW2000 Datasets:
2. Download the dataset from the
   official [project page](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm).
3. Place the downloaded dataset into a directory named 'Datasets'.

        
## Training the Model

To initiate the training process, use the following command:

```bash
python main.py --train
```
                   
## Inference

For inference on new images, run:

```bash
python main.py --inference
```




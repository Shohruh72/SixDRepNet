# 6DRepNet: Head Pose Estimation
![Vizualization](https://github.com/Shohruh72/SixDRepNet/blob/master/weights/Result.gif)

## Features

* **6D Rotation Matrix Representation**
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

| Backbone  | Epochs | Pitch | Yaw | Roll | Params (M) | FLOPS (B) | Pretrained weights |
|:---------:|:------:|------:|----:|-----:|-----------:|----------:|-------------------:|
| RepVGG-A0 |  90    |  5.4  | 4.3 | 3.8  |     9.1    |    1.5    | [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_a0.pt)|
| RepVGG-A1 |  90    |  5.2  | 3.9 | 3.7  |     14     |    2.6    | [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_a1.pt)                   |
| **RepVGG-A2** |  **90**    |  **4.8**  | **3.7** | **3.4**  |    28.2    |    5.7    | [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_a2.pt)                    |
| RepVGG-B0 |  90    |  5.0  | 3.9 | 3.5  |    15.8    |    3.4    | [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_b0.pt)                    ||
| RepVGG-B1 |  90    |  5.0  | 3.9 | 3.5  |    57.4    |   13.1    | [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_b1.pt)                   |
| RepVGG-B1G2 |  90  |  4.9  | 3.6 | 3.4  |    45.7    |    9.8    | [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_b1g2.pt)                  |
| RepVGG-B1G4 |  90  |  4.8  | 3.6 | 3.4  |    39.9    |    8.1    | [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_b1g4.pt)                    |
| RepVGG-B2 |  90    |  4.78 | 3.4 | 3.3  |     89     |   20.4    | [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_b2.pt)                       |
| RepVGG-B2G4 |  90  | 4.8   | 3.6 | 3.4  |    61.7    |   12.6    | [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_b2g4.pt)                      |
| RepVGG-B3 |  90    |  4.9  | 3.6 | 3.4  |    123     |   29.2    | [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_b3.pt)                  |
| RepVGG-B3G4 |  90  |  4.8  | 3.7 | 3.4  |    83.8    |   17.9    |  [model](https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_b3g4.pt)                     |

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

1. Download the 300W-LP, AFLW2000 Datasets:
2. Download the dataset from the
   official [project page](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm).
3. Place the downloaded dataset into a directory named 'Datasets'.

        
## Training the Model

To initiate the training process, use the following command:
* Configure your dataset path in main.py for training
* Configure Model name (default A2) in main.py for training
* Run the below command for Single-GPU training
```bash
python main.py --train
```
* Run the below command for Multi-GPU training $ is number of GPUs 
```bash
bash main.sh $ --train
```

## Testing the Model
Configure your dataset path in main.py for testing
Run the below command:
```bash
bash main.sh $ --test
```
## Inference
* Configure your video path in main.py for visualizing the demo
* Run the below command:
```bash
python main.py --demo
```




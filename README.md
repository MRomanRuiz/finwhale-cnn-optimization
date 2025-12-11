# finwhale-cnn-optimization
Comparative analysis of convolutional neural networks for fin whale call detection and classification, focusing on architectural optimization and interpretability.

## Table of Contents
1. [Requirements](#requirements)
2. [Structure](#structure)
    - [Main files](#main-files)
    - [Scripts](#scripts)
    - [Extra files](#extra-files)
3. [Usage](#usage)
4. [Reference](#reference)

## Requirements

- MATLAB R2024a or newer
  - Required Toolboxes: Signal Processing, Image Processing, Deep Learning, Statistics and Machine Learning

## Structure
### Main functions
These MATLAB functions are part of a framework designed to perform **massive training and performance evaluation** of convolutional neural networks (CNNs).  
Each function varies specific parameters (e.g., convolution size, pooling size, or architecture) to analyze the impact on model performance metrics such as **accuracy, precision, recall, and F1-score**.

**Function `massiveTrainingConv.m`**  
This MATLAB function performs massive training of convolutional neural networks while varying **convolutional layer parameters** (filter size and number of filters). It is used to analyze the performance of different architectures.  

- **Input parameters**:  
  - `trainingData`: Image datastore used for training  
  - `testData`: Image datastore used for testing  
  - `ranges`: Vector defining convolution parameters `[minFilterSize, maxFilterSize, minRepetitions, maxRepetitions]`  
  - `num`: Number of training repetitions  

- **Output parameters**:  
  - `statistics`: 3D matrix of accuracy results for each configuration  
  - `results`: Struct containing `accuracy`, `precision`, `recall`, `F1`, and `confMatrix`  
  - `info`: String array describing filter size and repetitions for each test  

**Function `massiveTrainingMaxPool.m`**  
This MATLAB function performs massive training of convolutional neural networks while varying the **max pooling size**, allowing performance comparison across different pooling configurations.  

- **Input parameters**:  
  - `trainingData`: Image datastore used for training  
  - `testData`: Image datastore used for testing  
  - `ranges`: Vector defining pooling size range `[minSize, maxSize]`  
  - `num`: Number of training repetitions  

- **Output parameters**:  
  - `statistics`: Accuracy matrix for each pooling size and iteration  
  - `results`: Struct containing `accuracy`, `precision`, `recall`, `F1`, and `confMatrix`  
  - `info`: String array describing the pooling configuration for each test  

**Function `massiveTrainingReLUBatch.m`**  
This MATLAB function performs multiple trainings of a convolutional neural network using **ReLU activation** and **batch normalization**, in order to evaluate consistency and performance across several runs.  

- **Input parameters**:  
  - `trainingData`: Image datastore used for training  
  - `testData`: Image datastore used for testing  
  - `num`: Number of training repetitions  

- **Output parameters**:  
  - `statistics`: Vector with accuracy for each training iteration  
  - `results`: Struct with `accuracy`, `precision`, `recall`, `F1`, and `confMatrix`  

**Function `massiveTrainingNetwork.m`**  
This MATLAB function trains a **custom CNN architecture**. The model contains four convolutional blocks followed by ReLU and max pooling layers, and it evaluates network performance across multiple runs.  

- **Input parameters**:  
  - `trainingData`: Image datastore used for training  
  - `testData`: Image datastore used for testing  
  - `num`: Number of training repetitions  

- **Output parameters**:  
  - `statistics`: Vector containing the accuracy obtained in each iteration  
  - `results`: Struct including `accuracy`, `precision`, `recall`, `F1`, and `confMatrix`  

**Function `filterImage`**
This MATLAB function computes the filtered version of a given image by performing a pixel-wise intensity transformation. This sequence enhances features for subsequent processing steps.
The process involves:
- Inverting the image  
- Subtracting the original image  
- Inverting the result 

## Scripts
**Script `visualAnalysis.m`** 
This MATLAB script visualizes the **feature map activations** of a trained convolutional neural network (`trainedDetector`) across different layers and convolutional blocks. It is designed to help understand how the network processes an input image and which features are extracted at each stage.  

- **Process overview**:  
  1. Loads a pre-trained CNN containing the variable `trainedDetector`.  
  2. Reads and filters an input image using the function `filterImage`.  
  3. Sequentially computes and displays activations for key layers of the network (convolutional, ReLU, and pooling layers).

This tool is mainly used for **interpretability and qualitative analysis** of CNN behavior, providing insight into which spatial features are captured at different network depths.

**Script `massiveTests.m`** 
This MATLAB script performs **massive CNN training experiments** and subsequent **statistical analysis** to evaluate network performance across different architectures and hyperparameter configurations. It systematically compares several neural network setups, from simple convolutional models to a finalized architecture published in a research article.

The script is divided into multiple sections, each corresponding to a specific experiment:
1. Single convolutional layer – training and analysis  
2. Single convolutional layer with variable max pooling  
3. Convolutional layer with ReLU and/or batch normalization  
4. Published detection network architecture  
5. Reduced version of the published network (three convolutional blocks)  

For each configuration, the script:
- Loads and prepares training/testing image datasets.  
- Calls custom training functions (`massiveTrainingConv`, `massiveTrainingMaxPool`, `massiveTrainingReLUBatch`, `massiveTrainingNetwork`).  
- Computes and visualizes statistical results (accuracy, mean, standard deviation).  
- Identifies the **best-performing training iteration** and **optimal average configuration**.

## Extra files
Some extra files are added with the results of our experiments:
- `Network.mat`: network trained for visualization.
- `testsBatch.mat`: batch configuration tests.
- `testsConv.mat`: convolution layer configuration tests. 
- `testsMaxPool.mat`: max pooling configuration tests.
- `testsReLU.mat`: ReLU configuration tests.

## Usage
1. **Output visualization of a network:**  
Run the MATLAB script `visualAnalysis` to see the activations.

2. **Testing different configurations for training:**  
Use the MATLAB script `massiveTests.m` to train and compute performance metrics for different configurations.

## Reference

This code accompanies the study **Optimized CNN Architectures for Real-Time
Monitoring of Fin Whale Acoustic Activity**.

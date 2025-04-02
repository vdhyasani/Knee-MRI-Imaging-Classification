# Knee_MRI_ML

python version: 3.12

How to download the dataset: https://www.kaggle.com/datasets/sohaibanwaar1203/kneemridataset

- Do "pip install kaggle"
- Check if kaggle API is installed by typing "kaggle --version"
- Type "kaggle datasets download -d sohaibanwaar1203/kneemridataset -p ./datasets"
- This will download the dataset in volumes under a folder called "datasets", which is imperative to have to the code to work

How to run train_svm.py:

- If a venv doesn't exist yet, add a venv by typing "python -m venv <venv_name>"
- This will create a virtual environment where you can download the necessary dependencies
- Open the virtual environment by typing "source <venv_name>/bin/activate"
- Download the necessary dependencies by typing "pip install imbalanced-learn scikit-learn numpy pandas"
- You can check if the dependencies have installed by "<dependency> --version"
- Type "python train_svm.py" in the terminal to run

How to run train_cnn.py:

This project implements a Convolutional Neural Network (CNN) for classifying knee MRI scans. The model achieves 72.97% accuracy in distinguishing between two classes of knee conditions.

## Project Structure
```
knee_mri_project/
├── data/
│   ├── vol01/
│   ├── vol02/
│   ├── ...
│   ├── vol08/
│   └── labels.csv
├── main.py
```

## Environment Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv knee_mri_env
source knee_mri_env/bin/activate  # On Unix/macOS
# OR
knee_mri_env\Scripts\activate     # On Windows
```

2. Install required packages:
```bash
pip3 install torch torchvision numpy pandas scikit-learn opencv-python matplotlib
```

## Dependencies
The following versions were used in development:
```
torch==2.1.0
numpy==1.24.3
opencv-python==4.8.0
scikit-learn==1.3.0
matplotlib==3.7.2
pandas==2.0.3
```

## Dataset Structure
- MRI scans are stored in .pck files within volume folders (vol01 through vol08)
- Each scan has dimensions: (slices, 320, 320)
## Running the Code

1. Place your MRI data in the data directory following the structure above.

2. Run the model:
```bash
python3 train_cnn.py
```

## Model Architecture
The CNN architecture consists of:
- 4 convolutional blocks with ReLU activation
- Batch normalization after each conv layer
- MaxPooling for dimensionality reduction
- Global average pooling
- Fully connected layers with dropout

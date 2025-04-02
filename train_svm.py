import warnings
warnings.filterwarnings("ignore")

import random
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle
import numpy as np
import pandas as pd


def load_metadata(metadata_file):
    """Load metadata into a pandas DataFrame."""
    return pd.read_csv(metadata_file)


def gather_pck_files(input_dataset):
    """Gather all .pck file paths from the dataset."""
    all_pck_files = []
    for vol_folder in os.listdir(input_dataset):
        vol_folder_path = os.path.join(input_dataset, vol_folder)
        if os.path.isdir(vol_folder_path) and vol_folder.startswith('vol'):
            for filename in os.listdir(vol_folder_path):
                if filename.endswith('.pck'):
                    file_path = os.path.join(vol_folder_path, filename)
                    all_pck_files.append(file_path)
    return all_pck_files


def process_pck_files(random_pck_files, metadata):
    """Process each .pck file and extract data and metadata."""
    data_list = []
    metadata_features = []
    labels = []
    zero_count, one_or_two_count = 0, 0

    for file_path in random_pck_files:
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                data_list.append(data)
                print(f"Loaded {file_path} with shape: {np.shape(data)}")

                # Extract metadata
                filename = os.path.basename(file_path)
                exam_id = int(filename.split('-')[0])
                series_no = int(filename.split('-')[1].split('.')[0])
                row = metadata[(metadata['examId'] == exam_id) &
                               (metadata['seriesNo'] == series_no)]

                # Extract aclDiagnosis
                acl_diagnosis = row['aclDiagnosis'].values[0]

                # Binary classification: 0 (if aclDiagnosis == 0), 1 (if aclDiagnosis == 1 or 2)
                if acl_diagnosis == 0:
                    labels.append(0)
                    zero_count += 1
                else:
                    labels.append(1)
                    one_or_two_count += 1

                # Extract additional metadata features
                features = row[['kneeLR', 'roiX', 'roiY', 'roiZ',
                                'roiHeight', 'roiWidth', 'roiDepth']].values.flatten()
                metadata_features.append(features)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return data_list, metadata_features, labels, zero_count, one_or_two_count


def pad_data(data_list):
    """Pad each array to the maximum shape and stack them into one array."""
    max_shape = tuple(max(s) for s in zip(*[np.shape(data) for data in data_list]))
    data_list_padded = [
        np.pad(data, [(0, max_shape[i] - np.shape(data)[i]) for i in range(len(max_shape))], mode='constant')
        for data in data_list
    ]
    return np.stack(data_list_padded)


def prepare_data(data_array, metadata_features, labels):
    """Prepare the dataset for training."""
    X_image = data_array.reshape(data_array.shape[0], -1)
    metadata_array = np.array(metadata_features)
    X = np.hstack((X_image, metadata_array))
    y = np.array(labels)
    return X, y


def train_svm(X_train, y_train):
    """Train the SVM classifier with class weights."""
    # Specify higher weight for the minority class (class 1)
    clf = svm.SVC(kernel='rbf', gamma='scale', C=100.0, class_weight={0: 2, 1: 5})
    clf.fit(X_train, y_train)
    return clf


def main():
    # Paths
    input_dataset = './datasets'
    metadata_file = './datasets/metadata.csv'

    # Load metadata
    metadata = load_metadata(metadata_file)

    # Gather .pck files
    all_pck_files = gather_pck_files(input_dataset)

    # Randomly sample a subset of .pck files (Max is 736)
    sample_size = 200
    random_pck_files = random.sample(all_pck_files, min(sample_size, len(all_pck_files)))

    # Process .pck files
    data_list, metadata_features, labels, zero_count, one_or_two_count = process_pck_files(random_pck_files, metadata)

    if not data_list:
        raise ValueError("No data was loaded from the .pck files. Please check the file paths.")

    # Pad and prepare data
    data_array = pad_data(data_list)
    print('* Stored data into data_array')
    X, y = prepare_data(data_array, metadata_features, labels)
    print('* Combined image data with metadata features')

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('* Split the data into train/test')

    # Resample the training set

    under_sampler = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)
    print('* Resampled the training data to undersample majority class')

    print(y_train)
    print(y_train_resampled)

    # Train SVM
    clf = train_svm(X_train_resampled, y_train_resampled)
    print('* Trained the SVM model')

    # Make predictions and evaluate
    y_pred = clf.predict(X_test)
    print('* Completed predictions on the test data')
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print(f'ACL healthy count (class 0): {zero_count}')
    print(f'ACL damage count (class 1): {one_or_two_count}')


if __name__ == "__main__":
    main()

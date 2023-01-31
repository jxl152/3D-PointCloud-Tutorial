import os
import shutil
import sys

# the benchmarks consist of 7481 training data
data_dir = "../../data/KITTI-3D_Object_Detection_Evaluation/data_object_image_2/training/image_2/"
label_dir = "../../data/KITTI-3D_Object_Detection_Evaluation/data_object_label_2/training/label_2/"

# divide the KITTI 3D object detection training data into training set and validation set
train_idx_path = "train.txt"
training_dataset_dir = "../data/training_dataset/"
training_label_dir = "../data/training_label/"
val_idx_path = "val.txt"
val_dataset_dir = "../data/validation_dataset/"
val_label_dir = "../data/validation_label/"


def read_idx_file(file_path):
    idx_set = set()
    with open(file_path, 'r') as file:
        for idx in file.readlines():
            idx_set.add(idx.rstrip())
    return idx_set


def copy_files(source_dir, dest_dir_train, dest_dir_val, train_idx_set, val_idx_set):
    for file_name in os.listdir(source_dir):
        idx = file_name.split('.')[0]
        if idx in train_idx_set:
            shutil.copy2(source_dir+file_name, dest_dir_train)
        elif idx in val_idx_set:
            shutil.copy2(source_dir+file_name, dest_dir_val)


if __name__ == "__main__":
    train_idx_set = read_idx_file(train_idx_path)
    val_idx_set = read_idx_file(val_idx_path)
    print("The training dataset contains %d images." % len(train_idx_set))
    print("The validation dataset contains %d images." % len(val_idx_set))

    # check any of training_dataset, validation_dataset, training_label and validation_label has already had data
    training_dataset = os.listdir(training_dataset_dir)
    val_dataset = os.listdir(val_dataset_dir)
    training_label = os.listdir(training_label_dir)
    val_label = os.listdir(val_label_dir)
    if training_dataset or val_dataset or training_label or val_label:
        print("Any of training_dataset, validation_dataset, training_label and validation_label has already had data.")
        sys.exit(0)

    # split data
    copy_files(data_dir, training_dataset_dir, val_dataset_dir, train_idx_set, val_idx_set)
    # split label
    copy_files(label_dir, training_label_dir, val_label_dir, train_idx_set, val_idx_set)

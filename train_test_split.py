import numpy as np
import os

# Specify the parent folder containing multiple subfolders
parent_folder = "Event_frames/"

input_folder1 = os.path.join(parent_folder, 'banana')
input_folder2 = os.path.join(parent_folder, 'toy')

# List files in each folder
file_name1 = os.listdir(input_folder1)
file_name2 = os.listdir(input_folder2)

# Ensure equal representation by taking the minimum number of files from each
min_file_count = min(len(file_name1), len(file_name2))
file_name_fp1 = [os.path.join('banana', file) for file in file_name1[:min_file_count]]
file_name_fp2 = [os.path.join('toy', file) for file in file_name2[:min_file_count]]

# Shuffle each list individually
np.random.shuffle(file_name_fp1)
np.random.shuffle(file_name_fp2)

# Split each category into train, validation, and test sets
train_size = int(0.6 * min_file_count)
valid_size = int(0.2 * min_file_count)

# Split banana samples
train_banana = file_name_fp1[:train_size]
valid_banana = file_name_fp1[train_size:train_size + valid_size]
test_banana = file_name_fp1[train_size + valid_size:]

# Split toy samples
train_toy = file_name_fp2[:train_size]
valid_toy = file_name_fp2[train_size:train_size + valid_size]
test_toy = file_name_fp2[train_size + valid_size:]

# Combine banana and toy samples for each subset
train_data_idx = np.hstack((train_banana, train_toy))
valid_data_idx = np.hstack((valid_banana, valid_toy))
test_data_idx = np.hstack((test_banana, test_toy))

# Shuffle each combined dataset
np.random.shuffle(train_data_idx)
np.random.shuffle(valid_data_idx)
np.random.shuffle(test_data_idx)

# Print the dataset sizes to confirm balance
print("Train Data:", len(train_data_idx))
print("Validation Data:", len(valid_data_idx))
print("Test Data:", len(test_data_idx))

# Save each dataset
os.makedirs("dataset", exist_ok=True)
np.save("dataset/train_set", train_data_idx)
np.save("dataset/valid_set", valid_data_idx)
np.save("dataset/test_set", test_data_idx)

import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
# Folder containing the .mat files
folder_path = 'test_batches/'

# Function to visualize the 'image' field within 'affNISTdata'
def visualize_mat(file_path):
    try:
        # Load the .mat file
        data = loadmat(file_path)
        print(f"Keys in {file_path}: {list(data.keys())}")

        # Extract the 'affNISTdata' structure
        affNISTdata = data.get('affNISTdata', None)
        if affNISTdata is not None:
            print(f"Available fields in 'affNISTdata': {affNISTdata.dtype.names}")

            # Extract the 'image' field
            image_data = affNISTdata['image'][0, 0]  # Access the 'image' field
            label_data = affNISTdata['label_one_of_n'][0, 0]  # Access the 'label' field
            
            if isinstance(image_data, np.ndarray):
                print(f"Shape of image_data: {image_data.shape}")

                # Reshape each column of image_data into a 40x40 image
                num_images = image_data.shape[1]
                temp = random.randint(3,5)
                for idx in range(0, num_images, temp):  # Display up to 5 images
                    image_vector = image_data[:, idx]
                    image_matrix = image_vector.reshape(40, 40).T  # Reshape and transpose

                    # Get the corresponding label
                    label = np.argmax(label_data[:, idx]) if label_data.ndim > 1 else label_data[idx]

                    # Plot the image
                    plt.figure()
                    plt.title(f"Transformed Image (Label: {label})")
                    plt.imshow(image_matrix, cmap='gray', aspect='auto')
                    plt.axis('off')
                    plt.show()
            else:
                print(f"'image' field in {file_path} is not a valid ndarray.")
        else:
            print(f"No 'affNISTdata' key found in {file_path}.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Loop through files 1.mat to 32.mat
for i in range(1, 33):
    file_path = os.path.join(folder_path, f'{i}.mat')
    if os.path.exists(file_path):
        print(f"Processing file: {file_path}")
        visualize_mat(file_path)
    else:
        print(f"File {file_path} not found.")
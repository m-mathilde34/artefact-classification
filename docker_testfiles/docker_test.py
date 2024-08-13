import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        print(f"GPU Name: {gpu.name}, GPU Type: {gpu.device_type}")
else:
    print("No GPU(s) found.")

# Create two random matrices
matrix1 = tf.random.normal([1000, 1000])
matrix2 = tf.random.normal([1000, 1000])

# Perform matrix multiplication on the GPU
with tf.device('/device:GPU:0'):
    product = tf.matmul(matrix1, matrix2)

# Print the result
print(product)

# Define the path to the mounted volume
volume_path = '/app/data/9_american/'  # This path should match the mounted volume path in the container

# Check if the volume path exists
if os.path.exists(volume_path):
    print(f"Contents of {volume_path}:")
    contents = os.listdir(volume_path)
    for item in contents:
        print(item)
else:
    print(f"The volume path {volume_path} does not exist or is not accessible.")

# Specify the file path
file_path = "example.txt"

# Create an empty text file
with open(file_path, "w") as file:
    pass
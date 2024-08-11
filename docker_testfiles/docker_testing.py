import tensorflow as tf
import time

print(tf.test.is_gpu_available())

# Print GPU devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))


# Create two random matrices
matrix_a = tf.random.normal(shape=(1000, 1000))
matrix_b = tf.random.normal(shape=(1000, 1000))

# Perform matrix multiplication on the GPU
start_time = time.time()
result = tf.matmul(matrix_a, matrix_b)
end_time = time.time()

# Print the result and computation time
print("Matrix multiplication result:")
print(result)
print(f"Computation time on GPU: {end_time - start_time} seconds")
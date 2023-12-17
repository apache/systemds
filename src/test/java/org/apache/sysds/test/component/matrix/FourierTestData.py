import numpy as np
import csv

def generate_inputs(num_inputs, max_power):
    for _ in range(num_inputs):
        power = np.random.randint(1, max_power+1)
        length = 2 ** power
        yield np.random.rand(length)  # generate array of random floats

def compute_fft(inputs):
    for input_array in inputs:
        yield np.fft.fft(input_array)

def save_to_file(inputs, outputs, filename, mode='a'):
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)
        for input_array, output_array in zip(inputs, outputs):
            flattened_data = np.concatenate((input_array, output_array.real, output_array.imag))
            writer.writerow(flattened_data)

# Parameters
num_inputs = 100000
batch_size = 10000
max_power = 10  # example max power of 2 for input length
filename = "fft_data.csv"

# Process and save in batches
for i in range(0, num_inputs, batch_size):
    current_batch = min(batch_size, num_inputs - i)
    inputs = list(generate_inputs(current_batch, max_power))
    outputs = list(compute_fft(inputs))
    save_to_file(inputs, outputs, filename, mode='a' if i > 0 else 'w')
    print(f"Batch {i//batch_size + 1} out of {num_inputs/batch_size} saved to {filename}")

print("All data processed and saved.")
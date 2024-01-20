"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import numpy as np
import csv
import time

def generate_inputs(num_inputs, max_power):
	for _ in range(num_inputs):
		power = np.random.randint(1, max_power+1)
		length = 2 ** power
		yield np.random.rand(length)  # generate array of random floats

def generate_complex_inputs(num_inputs, max_power):
	for _ in range(num_inputs):
		power = np.random.randint(1, max_power+1)
		length = 2 ** power
		real_part = np.random.rand(length)
		imag_part = np.random.rand(length)
		complex_array = real_part + 1j * imag_part
		yield complex_array


def compute_fft(inputs):
	total_time = 0
	num_calculations = 0

	for input_array in inputs:
		start_time = time.time()
		result = np.fft.fft(input_array)
		end_time = time.time()

		total_time += end_time - start_time
		num_calculations += 1

		if num_calculations % 1000 == 0:
			average_time = total_time / num_calculations
			print(f"Average execution time after {num_calculations} calculations: {average_time:.6f} seconds")

		yield result

def compute_ifft(inputs):
	total_time = 0
	num_calculations = 0

	for input_array in inputs:
		start_time = time.time()
		result = np.fft.ifft(input_array)
		end_time = time.time()

		total_time += end_time - start_time
		num_calculations += 1

		if num_calculations % 1000 == 0:
			average_time = total_time / num_calculations
			print(f"Average execution time after {num_calculations} calculations: {average_time:.6f} seconds")

		yield result



def save_to_file(inputs, outputs, filename, mode='a'):
	with open(filename, mode, newline='') as file:
		writer = csv.writer(file)
		for input_array, output_array in zip(inputs, outputs):
			flattened_data = np.concatenate((input_array, output_array.real, output_array.imag))
			writer.writerow(flattened_data)


def save_to_file_complex(inputs, outputs, filename, mode='a'):
	with open(filename, mode, newline='') as file:
		writer = csv.writer(file)
		for input_array, output_array in zip(inputs, outputs):
			flattened_data = np.concatenate((input_array.real, input_array.imag, output_array.real, output_array.imag))
			writer.writerow(flattened_data)

def generate_complex_inputs_2d(num_inputs, max_power):
	for _ in range(num_inputs):
		power = np.random.randint(1, max_power+1)
		rows = 2 ** power
		cols = 2 ** power
		real_part = np.random.rand(rows, cols)
		imag_part = np.random.rand(rows, cols)
		complex_array = real_part + 1j * imag_part
		yield complex_array

def compute_fft_2d(inputs):

	total_time = 0
	num_calculations = 0

	for input_array in inputs:
		start_time = time.time()
		result = np.fft.fft2(input_array)
		end_time = time.time()

		total_time += end_time - start_time
		num_calculations += 1

		if num_calculations % 1000 == 0:
			average_time = total_time / num_calculations
			print(f"Average execution time after {num_calculations} calculations: {average_time:.6f} seconds")

		yield result

def compute_ifft_2d(inputs):

	total_time = 0
	num_calculations = 0

	for input_array in inputs:
		start_time = time.time()
		result = np.fft.ifft2(input_array)
		end_time = time.time()

		total_time += end_time - start_time
		num_calculations += 1

		if num_calculations % 1000 == 0:
			average_time = total_time / num_calculations
			print(f"Average execution time after {num_calculations} calculations: {average_time:.6f} seconds")

		yield result

def save_to_file_complex_2d(inputs, outputs, filename, mode='a'):
	with open(filename, mode, newline='') as file:
		writer = csv.writer(file)
		for input_array, output_array in zip(inputs, outputs):
			flattened_input = np.concatenate((input_array.real.flatten(), input_array.imag.flatten()))
			flattened_output = np.concatenate((output_array.real.flatten(), output_array.imag.flatten()))
			writer.writerow(np.concatenate((flattened_input, flattened_output)))


# Parameters
num_inputs = 100000
batch_size = 10000
max_power = 10  # example max power of 2 for input length


# Process and save in batches
for i in range(0, num_inputs, batch_size):
	current_batch = min(batch_size, num_inputs - i)
	inputs = list(generate_inputs(current_batch, max_power))
	outputs = list(compute_fft(inputs))
	save_to_file(inputs, outputs, "fft_data.csv", mode='a' if i > 0 else 'w')
	print(f"Batch {i//batch_size + 1} out of {num_inputs//batch_size} saved to fft_data.csv")

print("All data for fft processed and saved.")

# Process and save in batches 


for i in range(0, num_inputs, batch_size):
	current_batch = min(batch_size, num_inputs - i)
	inputs = list(generate_inputs(current_batch, max_power))
	outputs = list(compute_ifft(inputs))
	save_to_file(inputs, outputs, "ifft_data.csv", mode='a' if i > 0 else 'w')
	print(f"Batch {i//batch_size + 1} out of {num_inputs//batch_size} saved to ifft_data.csv")

print("All real iff data processed and saved.")

# Process and save in batches
for i in range(0, num_inputs, batch_size):
	current_batch = min(batch_size, num_inputs - i)
	inputs = list(generate_complex_inputs(current_batch, max_power))
	outputs = list(compute_ifft(inputs))
	save_to_file_complex(inputs, outputs, "complex_ifft_data.csv", mode='a' if i > 0 else 'w')
	print(f"Batch {i//batch_size + 1} out of {num_inputs//batch_size} saved to complex_ifft_data.csv")

print("All complex ifft data processed and saved.")


num_inputs = 100000
batch_size = 1000
max_power = 3

# Process and save 2D FFT data in batches
filename_2d = "complex_fft_2d_data.csv"
for i in range(0, num_inputs, batch_size):
	current_batch = min(batch_size, num_inputs - i)
	inputs_2d = list(generate_complex_inputs_2d(current_batch, max_power))
	outputs_2d = list(compute_fft_2d(inputs_2d))
	save_to_file_complex_2d(inputs_2d, outputs_2d, filename_2d, mode='a' if i > 0 else 'w')
	print(f"Batch {i//batch_size + 1} out of {num_inputs//batch_size} saved to {filename_2d}")

print("All complex 2D fft data processed and saved.")

# Process and save 2D IFFT data in batches
filename_2d = "complex_ifft_2d_data.csv"
for i in range(0, num_inputs, batch_size):
	current_batch = min(batch_size, num_inputs - i)
	inputs_2d = list(generate_complex_inputs_2d(current_batch, max_power))
	outputs_2d = list(compute_ifft_2d(inputs_2d))
	save_to_file_complex_2d(inputs_2d, outputs_2d, filename_2d, mode='a' if i > 0 else 'w')
	print(f"Batch {i//batch_size + 1} out of {num_inputs//batch_size} saved to {filename_2d}")

print("All complex 2D ifft data processed and saved.")


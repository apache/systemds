import unittest
import numpy as np
from systemds.context import SystemDSContext
from numpy.fft import fft2
from numpy.fft import fft as np_fft

class TestFFT2(unittest.TestCase):
    def setUp(self):
        # Initialize SystemDS context
        self.sds = SystemDSContext()

    def tearDown(self):
        # Clean up SystemDS context
        self.sds.close()

    def test_fft_basic(self):
        # Example input matrix
        input_matrix = np.array([[1, 2, 3, 4],
                                 [5, 6, 7, 8],
                                 [9, 10, 11, 12],
                                 [13, 14, 15, 16]])
        # Transfer input matrix to SystemDS
        sds_input = self.sds.from_numpy(input_matrix)
        # Perform FFT
        fft_result = self.sds.fft(sds_input).compute()

        # Extract real and imaginary parts
        real_part, imag_part = fft_result

        # Expected values for real and imaginary parts (example, adjust accordingly)
        np_fft_result = fft2(input_matrix)
        expected_real = np.real(np_fft_result)
        expected_imag = np.imag(np_fft_result) 

        # Verify real part
        np.testing.assert_array_almost_equal(real_part, expected_real, decimal=5)
        # Verify imaginary part
        np.testing.assert_array_almost_equal(imag_part, expected_imag, decimal=5)

    
    def test_fft_random_1d(self):
        # Generate a random 1D input matrix
        np.random.seed(123)  # For reproducibility
        input_matrix = np.random.rand(1, 16)  # 1D array of length 16
        
        # Transfer input matrix to SystemDS
        sds_input = self.sds.from_numpy(input_matrix)
        
        # Perform FFT on the 1D input
        fft_result = self.sds.fft(sds_input).compute()

        # Extract real and imaginary parts
        real_part, imag_part = fft_result

        # Compute expected FFT using NumPy's 1D FFT
        np_fft_result = np_fft(input_matrix[0])  # Use first row for 1D FFT
        expected_real = np.real(np_fft_result)
        expected_imag = np.imag(np_fft_result) 

        # Verify real part
        np.testing.assert_array_almost_equal(real_part.flatten(), expected_real, decimal=5)
        # Verify imaginary part
        np.testing.assert_array_almost_equal(imag_part.flatten(), expected_imag, decimal=5)
    
    def test_ifft_basic(self):
        print("starting test_ifft_basic")
        # Example input matrices for real and imaginary parts
        real_input_matrix = np.array([[1, 2, 3, 4],
                                       [5, 6, 7, 8],
                                       [9, 10, 11, 12],
                                       [13, 14, 15, 16]])
                                       
        imag_input_matrix = np.array([[1, 2, 3, 4],
                                       [5, 6, 7, 8],
                                       [9, 10, 11, 12],
                                       [13, 14, 15, 16]]) 
        
        # Transfer input matrices to SystemDS
        sds_real_input = self.sds.from_numpy(real_input_matrix)
        sds_imag_input = self.sds.from_numpy(imag_input_matrix)
        
        # Perform IFFT
        ifft_result = self.sds.ifft(sds_real_input, sds_imag_input).compute(verbose=True)

        # Extract real and imaginary parts
        real_part, imag_part = ifft_result

        # Compute expected IFFT using NumPy
        np_ifft_result = np.fft.ifft2(real_input_matrix + 1j * imag_input_matrix)
        expected_real = np.real(np_ifft_result)
        expected_imag = np.imag(np_ifft_result)

        # Verify real part
        np.testing.assert_array_almost_equal(real_part, expected_real, decimal=5)
        # Verify imaginary part
        np.testing.assert_array_almost_equal(imag_part, expected_imag, decimal=5)

    def test_ifft_random_1d(self):
        print("starting test_ifft_random_1d")
        # Generate a random complex 1D input matrix
        np.random.seed(123)  # For reproducibility
        real_part = np.random.rand(1, 16)  # Real part of the complex data
        imag_part = np.random.rand(1, 16)  # Imaginary part of the complex data
        complex_input = real_part + 1j * imag_part  # Form the complex data
        
        # Perform FFT first to get a transformed matrix
        np_fft_result = np.fft.fft(complex_input[0])

        # Transfer the FFT results (as separate real and imaginary matrices) to SystemDS
        sds_real_input = self.sds.from_numpy(np.real(np_fft_result).reshape(1, -1))
        sds_imag_input = self.sds.from_numpy(np.imag(np_fft_result).reshape(1, -1))

        # Perform IFFT in SystemDS
        ifft_result = self.sds.ifft(sds_real_input, sds_imag_input).compute()

        # Extract real and imaginary parts
        real_part_result, imag_part_result = ifft_result

        # Flatten the results for comparison
        real_part_result = real_part_result.flatten()
        imag_part_result = imag_part_result.flatten()

        # Compute expected IFFT using NumPy's 1D IFFT on the original complex input
        expected_ifft = np.fft.ifft(np_fft_result)
        expected_real = np.real(expected_ifft)
        expected_imag = np.imag(expected_ifft)

        # Verify real part
        np.testing.assert_array_almost_equal(real_part_result, expected_real, decimal=5)
        # Verify imaginary part
        np.testing.assert_array_almost_equal(imag_part_result, expected_imag, decimal=5)


    def test_ifft_real_only(self):
        print("starting test_ifft_real_only")
        # Generate a random real 1D input matrix
        np.random.seed(123)  # For reproducibility
        real_part = np.random.rand(1, 16)  # Real part of the data
        
        # Perform FFT on real data to get a transformed matrix suitable for IFFT
        np_fft_result = np.fft.fft(real_part[0])

        # Transfer the FFT result (real part) to SystemDS as input for IFFT
        sds_real_input = self.sds.from_numpy(np.real(np_fft_result).reshape(1, -1))
        
        # Perform IFFT in SystemDS with only real input
        ifft_result = self.sds.ifft(sds_real_input).compute()

        # Extract real and imaginary parts from the result
        real_part_result, imag_part_result = ifft_result

        # Flatten the results for comparison
        real_part_result = real_part_result.flatten()
        imag_part_result = imag_part_result.flatten()

        # Compute expected IFFT using NumPy's 1D IFFT
        expected_ifft = np.fft.ifft(np_fft_result)
        expected_real = np.real(expected_ifft)
        expected_imag = np.imag(expected_ifft)

        # Verify real part
        np.testing.assert_array_almost_equal(real_part_result, expected_real, decimal=5)
        # Verify imaginary part (should be close to zero)
        np.testing.assert_array_almost_equal(imag_part_result, expected_imag, decimal=5)


if __name__ == '__main__':
    unittest.main()

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


if __name__ == '__main__':
    unittest.main()

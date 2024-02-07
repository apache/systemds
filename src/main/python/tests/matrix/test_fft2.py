import unittest
import numpy as np
from systemds.context import SystemDSContext
from numpy.fft import fft2
from numpy.fft import fft as np_fft

class TestFFT2(unittest.TestCase):
    def setUp(self):
        self.sds = SystemDSContext()

    def tearDown(self):
        self.sds.close()

    def test_fft_basic(self):

        input_matrix = np.array([[1, 2, 3, 4],
                                 [5, 6, 7, 8],
                                 [9, 10, 11, 12],
                                 [13, 14, 15, 16]])

        sds_input = self.sds.from_numpy(input_matrix)
        fft_result = self.sds.fft(sds_input).compute()

        real_part, imag_part = fft_result

        np_fft_result = fft2(input_matrix)
        expected_real = np.real(np_fft_result)
        expected_imag = np.imag(np_fft_result) 

        np.testing.assert_array_almost_equal(real_part, expected_real, decimal=5)
        np.testing.assert_array_almost_equal(imag_part, expected_imag, decimal=5)

    
    def test_fft_random_1d(self):
        np.random.seed(123) 
        input_matrix = np.random.rand(1, 16)  
        
        sds_input = self.sds.from_numpy(input_matrix)
        
        fft_result = self.sds.fft(sds_input).compute()

        real_part, imag_part = fft_result

        np_fft_result = np_fft(input_matrix[0]) 
        expected_real = np.real(np_fft_result)
        expected_imag = np.imag(np_fft_result) 

        np.testing.assert_array_almost_equal(real_part.flatten(), expected_real, decimal=5)
        np.testing.assert_array_almost_equal(imag_part.flatten(), expected_imag, decimal=5)
    
    def test_ifft_basic(self):
        print("starting test_ifft_basic")

        real_input_matrix = np.array([[1, 2, 3, 4],
                                       [5, 6, 7, 8],
                                       [9, 10, 11, 12],
                                       [13, 14, 15, 16]])
                                       
        imag_input_matrix = np.array([[1, 2, 3, 4],
                                       [5, 6, 7, 8],
                                       [9, 10, 11, 12],
                                       [13, 14, 15, 16]]) 
        
        sds_real_input = self.sds.from_numpy(real_input_matrix)
        sds_imag_input = self.sds.from_numpy(imag_input_matrix)
        
        ifft_result = self.sds.ifft(sds_real_input, sds_imag_input).compute(verbose=True)

        real_part, imag_part = ifft_result

        np_ifft_result = np.fft.ifft2(real_input_matrix + 1j * imag_input_matrix)
        expected_real = np.real(np_ifft_result)
        expected_imag = np.imag(np_ifft_result)


        np.testing.assert_array_almost_equal(real_part, expected_real, decimal=5)
        np.testing.assert_array_almost_equal(imag_part, expected_imag, decimal=5)

    def test_ifft_random_1d(self):
        print("starting test_ifft_random_1d")
        np.random.seed(123) 
        real_part = np.random.rand(1, 16) 
        imag_part = np.random.rand(1, 16) 
        complex_input = real_part + 1j * imag_part  

        np_fft_result = np.fft.fft(complex_input[0])

        sds_real_input = self.sds.from_numpy(np.real(np_fft_result).reshape(1, -1))
        sds_imag_input = self.sds.from_numpy(np.imag(np_fft_result).reshape(1, -1))

        ifft_result = self.sds.ifft(sds_real_input, sds_imag_input).compute()

        real_part_result, imag_part_result = ifft_result

        real_part_result = real_part_result.flatten()
        imag_part_result = imag_part_result.flatten()

        expected_ifft = np.fft.ifft(np_fft_result)
        expected_real = np.real(expected_ifft)
        expected_imag = np.imag(expected_ifft)

        np.testing.assert_array_almost_equal(real_part_result, expected_real, decimal=5)
        np.testing.assert_array_almost_equal(imag_part_result, expected_imag, decimal=5)


    def test_ifft_real_only(self):
        print("starting test_ifft_real_only")
        np.random.seed(123)  
        real_part = np.random.rand(1, 16)  
        
        np_fft_result = np.fft.fft(real_part[0])

        sds_real_input = self.sds.from_numpy(np.real(np_fft_result).reshape(1, -1))
        
        ifft_result = self.sds.ifft(sds_real_input).compute()

        real_part_result, imag_part_result = ifft_result

        real_part_result = real_part_result.flatten()
        imag_part_result = imag_part_result.flatten()

        expected_ifft = np.fft.ifft(np_fft_result)
        expected_real = np.real(expected_ifft)
        expected_imag = np.imag(expected_ifft)

        np.testing.assert_array_almost_equal(real_part_result, expected_real, decimal=5)
        np.testing.assert_array_almost_equal(imag_part_result, expected_imag, decimal=5)


if __name__ == '__main__':
    unittest.main()

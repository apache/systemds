import unittest
import numpy as np
from systemds.context import SystemDSContext

class TestFFT(unittest.TestCase):
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

        np_fft_result = np.fft.fft2(input_matrix)
        expected_real = np.real(np_fft_result)
        expected_imag = np.imag(np_fft_result) 

        np.testing.assert_array_almost_equal(real_part, expected_real, decimal=5)
        np.testing.assert_array_almost_equal(imag_part, expected_imag, decimal=5)

    
    def test_fft_random_1d(self):
        np.random.seed(123) 
        for _ in range(10):
            input_matrix = np.random.rand(1, 16)  
            
            sds_input = self.sds.from_numpy(input_matrix)
            
            fft_result = self.sds.fft(sds_input).compute()

            real_part, imag_part = fft_result

            np_fft_result = np.fft.fft(input_matrix[0]) 
            expected_real = np.real(np_fft_result)
            expected_imag = np.imag(np_fft_result) 

            np.testing.assert_array_almost_equal(real_part.flatten(), expected_real, decimal=5)
            np.testing.assert_array_almost_equal(imag_part.flatten(), expected_imag, decimal=5)

    def test_fft_2d(self):
        np.random.seed(123) 
        for _ in range(10):
            input_matrix = np.random.rand(8, 8) 

            sds_input = self.sds.from_numpy(input_matrix)
            
            fft_result = self.sds.fft(sds_input).compute()

            real_part, imag_part = fft_result

            np_fft_result = np.fft.fft2(input_matrix)
            expected_real = np.real(np_fft_result)
            expected_imag = np.imag(np_fft_result)

            np.testing.assert_array_almost_equal(real_part, expected_real, decimal=5)
            np.testing.assert_array_almost_equal(imag_part, expected_imag, decimal=5)

    
    def test_ifft_basic(self):
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
        
        ifft_result = self.sds.ifft(sds_real_input, sds_imag_input).compute()

        real_part, imag_part = ifft_result

        np_ifft_result = np.fft.ifft2(real_input_matrix + 1j * imag_input_matrix)
        expected_real = np.real(np_ifft_result)
        expected_imag = np.imag(np_ifft_result)


        np.testing.assert_array_almost_equal(real_part, expected_real, decimal=5)
        np.testing.assert_array_almost_equal(imag_part, expected_imag, decimal=5)

    def test_ifft_random_1d(self):
        np.random.seed(123) 
        for _ in range(10):
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

    def test_ifft_real_only_basic(self):
        np.random.seed(123)  
        real = np.array([1, 2, 3, 4, 4,  3, 2, 1])

        sds_real_input = self.sds.from_numpy(real)
        
        ifft_result = self.sds.ifft(sds_real_input).compute()

        real_part_result, imag_part_result = ifft_result

        real_part_result = real_part_result.flatten()
        imag_part_result = imag_part_result.flatten()

        expected_ifft = np.fft.ifft(real)
        expected_real = np.real(expected_ifft)
        expected_imag = np.imag(expected_ifft)

        np.testing.assert_array_almost_equal(real_part_result, expected_real, decimal=5)
        np.testing.assert_array_almost_equal(imag_part_result, expected_imag, decimal=5)

    def test_ifft_real_only_random(self):
        np.random.seed(123) 
        for _ in range(10):
            input_matrix = np.random.rand(1, 16)  
            
            sds_input = self.sds.from_numpy(input_matrix)
            
            ifft_result = self.sds.ifft(sds_input).compute()

            real_part, imag_part = ifft_result

            np_ifft_result = np.fft.ifft(input_matrix[0]) 
            expected_real = np.real(np_ifft_result)
            expected_imag = np.imag(np_ifft_result) 

            np.testing.assert_array_almost_equal(real_part.flatten(), expected_real, decimal=5)
            np.testing.assert_array_almost_equal(imag_part.flatten(), expected_imag, decimal=5)
    

    def test_ifft_2d(self):
        np.random.seed(123) 
        for _ in range(10):
            input_matrix = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)

            fft_result = np.fft.fft2(input_matrix)

            sds_real_input = self.sds.from_numpy(np.real(fft_result))
            sds_imag_input = self.sds.from_numpy(np.imag(fft_result))

            ifft_result = self.sds.ifft(sds_real_input, sds_imag_input).compute()

            real_part, imag_part = ifft_result

            expected_ifft_result = np.fft.ifft2(fft_result)
            expected_real = np.real(expected_ifft_result)
            expected_imag = np.imag(expected_ifft_result)

            np.testing.assert_array_almost_equal(real_part, expected_real, decimal=5)
            np.testing.assert_array_almost_equal(imag_part, expected_imag, decimal=5)

    def test_fft_empty_matrix(self):
        input_matrix = np.array([])
        sds_input = self.sds.from_numpy(input_matrix)
        
        with self.assertRaisesRegex(RuntimeError, "The first argument to FFT cannot be an empty matrix."):
            _ = self.sds.fft(sds_input).compute()

    def test_ifft_empty_matrix(self):
        input_matrix = np.array([])
        sds_input = self.sds.from_numpy(input_matrix)
        
        with self.assertRaisesRegex(RuntimeError, "The first argument to IFFT cannot be an empty matrix."):
            _ = self.sds.ifft(sds_input).compute()

    def test_fft_single_element(self):
        input_matrix = np.array([[5]])
        sds_input = self.sds.from_numpy(input_matrix)
        fft_result = self.sds.fft(sds_input).compute()
        
        real_part, imag_part = fft_result
        np.testing.assert_array_almost_equal(real_part, [[5]], decimal=5)
        np.testing.assert_array_almost_equal(imag_part, [[0]], decimal=5)

    def test_ifft_single_element(self):
        input_matrix = np.array([[5]])
        sds_input = self.sds.from_numpy(input_matrix)
        ifft_result = self.sds.ifft(sds_input).compute()
        
        real_part, imag_part = ifft_result
        np.testing.assert_array_almost_equal(real_part, [[5]], decimal=5)
        np.testing.assert_array_almost_equal(imag_part, [[0]], decimal=5)

    def test_fft_zeros_matrix(self):
        input_matrix = np.zeros((4, 4))
        sds_input = self.sds.from_numpy(input_matrix)
        fft_result = self.sds.fft(sds_input).compute()
        
        real_part, imag_part = fft_result
        np.testing.assert_array_almost_equal(real_part, np.zeros((4, 4)), decimal=5)
        np.testing.assert_array_almost_equal(imag_part, np.zeros((4, 4)), decimal=5)

    def test_ifft_zeros_matrix(self):
        input_matrix = np.zeros((4, 4))
        sds_input = self.sds.from_numpy(input_matrix)
        ifft_result = self.sds.ifft(sds_input).compute()
        
        real_part, imag_part = ifft_result
        np.testing.assert_array_almost_equal(real_part, np.zeros((4, 4)), decimal=5)
        np.testing.assert_array_almost_equal(imag_part, np.zeros((4, 4)), decimal=5)

    def test_ifft_real_and_imaginary_dimensions_check(self):
        real_part = np.random.rand(1, 16) 
        imag_part = np.random.rand(1, 14) 

        sds_real_input = self.sds.from_numpy(real_part)
        sds_imag_input = self.sds.from_numpy(imag_part)

        with self.assertRaisesRegex(RuntimeError, "The real and imaginary part of the provided matrix are of different dimensions."):
            self.sds.ifft(sds_real_input, sds_imag_input).compute()

    


if __name__ == '__main__':
    unittest.main()

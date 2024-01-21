import unittest
import numpy as np
from systemds.context import SystemDSContext

class TestFFT(unittest.TestCase):

    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_fft_1d(self):
        # Generate a random 1D array
        np.random.seed(7)
        A = np.random.rand(8)  # Using a power of 2 for dimension
        A_sds = self.sds.from_numpy(A)

        # Compute FFT using SystemDS
        Re, Im = A_sds.fft().compute()  # Assuming fft method returns a tuple of real and imaginary parts

        # Compute FFT using NumPy
        A_fft = np.fft.fft(A)
        Re_np, Im_np = A_fft.real, A_fft.imag

        # Compare the results
        self.assertTrue(np.allclose(Re, Re_np))
        self.assertTrue(np.allclose(Im, Im_np))

    def test_fft_2d(self):
        # Generate a random 2D matrix
        np.random.seed(7)
        A = np.random.rand(8, 8)  # Using a power of 2 for dimensions
        A_sds = self.sds.from_numpy(A)

        # Compute FFT using SystemDS
        Re, Im = A_sds.fft().compute()  # Assuming fft method returns a tuple of real and imaginary parts

        # Compute FFT using NumPy
        A_fft = np.fft.fft2(A)
        Re_np, Im_np = A_fft.real, A_fft.imag

        # Compare the results
        self.assertTrue(np.allclose(Re, Re_np))
        self.assertTrue(np.allclose(Im, Im_np))

    def test_ifft_1d(self):
        np.random.seed(7)
        A = np.random.rand(8)
        A_sds = self.sds.from_numpy(A)

        # Compute IFFT using SystemDS
        Re, Im = A_sds.ifft().compute()  # Assuming ifft method returns a tuple of real and imaginary parts

        # Compute IFFT using NumPy
        A_ifft = np.fft.ifft(A)
        Re_np, Im_np = A_ifft.real, A_ifft.imag

        # Compare the results
        self.assertTrue(np.allclose(Re, Re_np, atol=1e-6))
        self.assertTrue(np.allclose(Im, Im_np, atol=1e-6))

    def test_ifft_2d(self):
        np.random.seed(7)
        A = np.random.rand(8, 8)
        A_sds = self.sds.from_numpy(A)

        # Compute IFFT using SystemDS
        Re, Im = A_sds.ifft().compute()  # Assuming ifft method returns a tuple of real and imaginary parts

        # Compute IFFT using NumPy
        A_ifft = np.fft.ifft2(A)
        Re_np, Im_np = A_ifft.real, A_ifft.imag

        # Compare the results
        self.assertTrue(np.allclose(Re, Re_np, atol=1e-6))
        self.assertTrue(np.allclose(Im, Im_np, atol=1e-6))

if __name__ == "__main__":
    unittest.main()

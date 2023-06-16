import java.util.Random;
import org.apache.sysds.utils.NativeHelper;

public class NativeBindingTestWithDgemm {
    static {
        System.loadLibrary("systemds_mkl-Darwin-x86_64");
    }
    // Helper method to flatten a 2D matrix into a 1D array
    private double[] flattenMatrix(double[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        double[] flattened = new double[rows * columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                flattened[i * columns + j] = matrix[i][j];
            }
        }

        return flattened;
    }

    // Helper method to generate a random matrix
    private double[][] generateRandomMatrix(int rows, int columns) {
        Random random = new Random();
        double[][] matrix = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = random.nextDouble();
            }
        }

        return matrix;
    }

    // Method to test the dgemm function
    public void testDgemm() {
        char transa = 'N';
        char transb = 'N';
        int m = 2000;
        int n = 200;
        int k = 1000;
        double alpha = 1.0;
        double beta = 0.0;

        // Generate random input matrices A, B, and C
        double[][] A = generateRandomMatrix(m, k);
        double[][] B = generateRandomMatrix(k, n);
        double[][] C = new double[m][n];

        // Convert matrices to 1D arrays
        double[] flatA = flattenMatrix(A);
        double[] flatB = flattenMatrix(B);
        double[] flatC = flattenMatrix(C);

        // Call the native dgemm method

        NativeHelper.testNativeBindingWithDgemm(transa, transb, m, n, k, alpha, flatA, k, flatB, n, beta, flatC, n);

        // Print the result matrix C
        System.out.println("Result Matrix C:");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(flatC[i * n + j] + " ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        new NativeBindingTestWithDgemm().testDgemm();
    }
}

package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.*;
import static org.junit.Assert.assertTrue;

public class FourierTestWithFiles {
    int progressInterval = 5000;

    // prior to executing the following tests it is necessary to run the Numpy Script in FourierTestData.py 
    // and add the generated files to the root of the project.
    @Test
    public void testFftWithNumpyData() throws IOException {
        String filename = "fft_data.csv"; // Path to your CSV file
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        int lineNumber = 0;
        long totalTime = 0; // Total time for all FFT computations
        int numCalculations = 0; // Number of FFT computations

        while ((line = reader.readLine()) != null) {
            lineNumber++;

            String[] values = line.split(",");
            int n = values.length / 3;
            double[][][] input = new double[2][1][n];
            double[][] expected = new double[2][n]; // First row for real, second row for imaginary parts

            for (int i = 0; i < n; i++) {
                input[0][0][i] = Double.parseDouble(values[i]);
                expected[0][i] = Double.parseDouble(values[n + i]); // Real part
                expected[1][i] = Double.parseDouble(values[n * 2 + i]); // Imaginary part
            }

            long startTime = System.nanoTime();
            MatrixBlock[] actualBlocks = fft(input);
            long endTime = System.nanoTime();

            if(lineNumber > 1000){
                totalTime += (endTime - startTime);
                numCalculations++;

                if (numCalculations % progressInterval == 0) {
                    double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
                    System.out.println("fft(double[][][] in): Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000) + " s");
                }
            }

            // Validate the FFT results
            validateFftResults(expected, actualBlocks, lineNumber);
        }

        reader.close();
        
    }

    private void validateFftResults(double[][] expected, MatrixBlock[] actualBlocks, int lineNumber) {
        int length = expected[0].length;
        for (int i = 0; i < length; i++) {
            double realActual = actualBlocks[0].getValueDenseUnsafe(0, i);
            double imagActual = actualBlocks[1].getValueDenseUnsafe(0, i);
            assertEquals("Mismatch in real part at index " + i + " in line " + lineNumber, expected[0][i], realActual, 1e-9);
            assertEquals("Mismatch in imaginary part at index " + i + " in line " + lineNumber, expected[1][i], imagActual, 1e-9);
        }
        if(lineNumber % progressInterval == 0){
            System.out.println("fft(double[][][] in): Finished processing line " + lineNumber);
        }
        
    }

    @Test
    public void testFftExecutionTime() throws IOException {
        String filename = "fft_data.csv"; // Path to your CSV file
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        int lineNumber = 0;
        long totalTime = 0; // Total time for all FFT computations
        int numCalculations = 0; // Number of FFT computations

        while ((line = reader.readLine()) != null) {
            lineNumber++;
            String[] values = line.split(",");
            int n = values.length / 3;
            double[][][] input = new double[2][1][n];

            for (int i = 0; i < n; i++) {
                input[0][0][i] = Double.parseDouble(values[i]); // Real part
                input[1][0][i] = Double.parseDouble(values[n + i]); // Imaginary part
            }

            long startTime = System.nanoTime();
            fft(input, false);
            long endTime = System.nanoTime();
            if(lineNumber > 1000){
                totalTime += (endTime - startTime);
                numCalculations++;

                if (numCalculations % progressInterval == 0) {
                    double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
                    System.out.println("fft(double[][][] in, boolean calcInv) Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000) + " s");
                }
            }
        }

        reader.close();
    }

    @Test
    public void testFftExecutionTimeOfOneDimFFT() throws IOException {
        String filename = "fft_data.csv"; // Path to your CSV file
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        int lineNumber = 0;
        long totalTime = 0; // Total time for all FFT computations
        int numCalculations = 0; // Number of FFT computations

        while ((line = reader.readLine()) != null) {
            lineNumber++;
            String[] values = line.split(",");
            int n = values.length / 2;
            double[][] input = new double[2][n]; // First row for real, second row for imaginary parts

            for (int i = 0; i < n; i++) {
                input[0][i] = Double.parseDouble(values[i]); // Real part
                input[1][i] = Double.parseDouble(values[n + i]); // Imaginary part
            }

            long startTime = System.nanoTime();
            fft_one_dim(input);
            long endTime = System.nanoTime();
            if(lineNumber > 1000){
                totalTime += (endTime - startTime);
                numCalculations++;

                if (numCalculations % progressInterval == 0) {
                    double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
                    System.out.println("fft_one_dim: Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000) + " s ");
                }
            }
        }

        reader.close();
    }


    // prior to executing this test it is necessary to run the Numpy Script in FourierTestData.py and add the generated file to the root of the project.
    @Test
    public void testIfftWithRealNumpyData() throws IOException {
        String filename = "ifft_data.csv"; // Path to your CSV file
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        int lineNumber = 0;
    
        while ((line = reader.readLine()) != null) {
            lineNumber++;
            String[] values = line.split(",");
            int n = values.length / 3;
            double[][][] input = new double[2][1][n];
            double[][] expected = new double[2][n]; // First row for real, second row for imaginary parts
    
            for (int i = 0; i < n; i++) {
                input[0][0][i] = Double.parseDouble(values[i]); // Real part of input
                // Imaginary part of input is assumed to be 0
                expected[0][i] = Double.parseDouble(values[n + i]); // Real part of expected output
                expected[1][i] = Double.parseDouble(values[n * 2 + i]); // Imaginary part of expected output
            }
    
            double[][][] actualResult = fft(input, true); // Perform IFFT
        
            // Validate the IFFT results
            validateFftResults(expected, actualResult, lineNumber);
        }
    
        reader.close();
    }
    
    private void validateFftResults(double[][] expected, double[][][] actualResult, int lineNumber) {
        int length = expected[0].length;
        for (int i = 0; i < length; i++) {
            double realActual = actualResult[0][0][i];
            double imagActual = actualResult[1][0][i];
            assertEquals("Mismatch in real part at index " + i + " in line " + lineNumber, expected[0][i], realActual, 1e-9);
            assertEquals("Mismatch in imaginary part at index " + i + " in line " + lineNumber, expected[1][i], imagActual, 1e-9);
        }
        if(lineNumber % progressInterval == 0){
            System.out.println("ifft(real input): Finished processing line " + lineNumber);
        }
        
    }
    


    @Test
public void testIfftWithComplexNumpyData() throws IOException {
    String filename = "complex_ifft_data.csv"; // Adjusted path to your IFFT data file with complex inputs
    BufferedReader reader = new BufferedReader(new FileReader(filename));
    String line;
    int lineNumber = 0;
    long totalTime = 0; // Total time for all IFFT computations
    int numCalculations = 0; // Number of IFFT computations

    while ((line = reader.readLine()) != null) {
        lineNumber++;
        String[] values = line.split(",");
        int n = values.length / 4; // Adjusted for complex numbers
        double[][][] input = new double[2][1][n]; // Real and imaginary parts
        double[][] expected = new double[2][n]; // Expected real and imaginary parts

        for (int i = 0; i < n; i++) {
            input[0][0][i] = Double.parseDouble(values[i]); // Real part of input
            input[1][0][i] = Double.parseDouble(values[i + n]); // Imaginary part of input
            expected[0][i] = Double.parseDouble(values[i + 2 * n]); // Expected real part
            expected[1][i] = Double.parseDouble(values[i + 3 * n]); // Expected imaginary part
        }

        long startTime = System.nanoTime();
        double[][][] actualResult = fft(input, true); // Perform IFFT
        long endTime = System.nanoTime();

        if (lineNumber > 1000) {
            totalTime += (endTime - startTime);
            numCalculations++;
        }

        // Validate the IFFT results
        validateComplexIFftResults(expected, actualResult, lineNumber);

        if (lineNumber % progressInterval == 0) {
            double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
            System.out.println("ifft: Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime / 1000) + " s");
        }
    }

    reader.close();
}

private void validateComplexIFftResults(double[][] expected, double[][][] actualResult, int lineNumber) {
    int length = expected[0].length;
    for (int i = 0; i < length; i++) {
        double realActual = actualResult[0][0][i];
        double imagActual = actualResult[1][0][i];
        assertEquals("Mismatch in real part at index " + i + " in line " + lineNumber, expected[0][i], realActual, 1e-9);
        assertEquals("Mismatch in imaginary part at index " + i + " in line " + lineNumber, expected[1][i], imagActual, 1e-9);
    }
    if (lineNumber % progressInterval == 0) {
    System.out.println("ifft(complex input): Finished processing line " + lineNumber);
    }
}


@Test
public void testFft2dWithNumpyData() throws IOException {
    String filename = "complex_fft_2d_data.csv"; // path to your 2D FFT data file
    BufferedReader reader = new BufferedReader(new FileReader(filename));
    String line;
    int lineNumber = 0;
    long totalTime = 0; // Total time for all FFT 2D computations
    int numCalculations = 0; // Number of FFT 2D computations

    while ((line = reader.readLine()) != null) {
        lineNumber++;
        String[] values = line.split(",");
        int halfLength = values.length / 4;
        int sideLength = (int) Math.sqrt(halfLength); // Assuming square matrix

        double[][][] input = new double[2][sideLength][sideLength];
        double[][][] expected = new double[2][sideLength][sideLength];

        for (int i = 0; i < halfLength; i++) {
            int row = i / sideLength;
            int col = i % sideLength;
            input[0][row][col] = Double.parseDouble(values[i]);
            input[1][row][col] = Double.parseDouble(values[i + halfLength]);
            expected[0][row][col] = Double.parseDouble(values[i + 2 * halfLength]);
            expected[1][row][col] = Double.parseDouble(values[i + 3 * halfLength]);
        }

        long startTime = System.nanoTime();
        double[][][] javaFftResult = fft(input, false); // Use your fft2d implementation
        long endTime = System.nanoTime();
        totalTime += (endTime - startTime);
        numCalculations++;

        for (int i = 0; i < sideLength; i++) {
            for (int j = 0; j < sideLength; j++) {
                assertComplexEquals("Mismatch at [" + i + "][" + j + "] in line " + lineNumber, 
                                    expected[0][i][j], expected[1][i][j], 
                                    javaFftResult[0][i][j], javaFftResult[1][i][j]);
            }
        }

        if (lineNumber % progressInterval == 0) {
            double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
            System.out.println("fft2d: Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000) + " s");
        }
    }

    reader.close();
    System.out.println("fft2d: Finished processing " + lineNumber + " lines.\n Average execution time: " + String.format("%.8f", (totalTime / 1e6 / numCalculations)/1000) + " s");
}


@Test
public void testIfft2dWithNumpyData() throws IOException {
    String filename = "complex_ifft_2d_data.csv"; // path to your 2D IFFT data file
    BufferedReader reader = new BufferedReader(new FileReader(filename));
    String line;
    int lineNumber = 0;
    long totalTime = 0; // Total time for all IFFT 2D computations
    int numCalculations = 0; // Number of IFFT 2D computations

    while ((line = reader.readLine()) != null) {
        lineNumber++;
        String[] values = line.split(",");
        int halfLength = values.length / 4;
        int sideLength = (int) Math.sqrt(halfLength); // Assuming square matrix

        double[][][] input = new double[2][sideLength][sideLength];
        double[][][] expected = new double[2][sideLength][sideLength];

        for (int i = 0; i < halfLength; i++) {
            int row = i / sideLength;
            int col = i % sideLength;
            input[0][row][col] = Double.parseDouble(values[i]);
            input[1][row][col] = Double.parseDouble(values[i + halfLength]);
            expected[0][row][col] = Double.parseDouble(values[i + 2 * halfLength]);
            expected[1][row][col] = Double.parseDouble(values[i + 3 * halfLength]);
        }

        long startTime = System.nanoTime();
        double[][][] javaIfftResult = fft(input, true); // Use your ifft2d implementation
        long endTime = System.nanoTime();
        if(lineNumber > 1000){
            totalTime += (endTime - startTime);
            numCalculations++;
        }

        for (int i = 0; i < sideLength; i++) {
            for (int j = 0; j < sideLength; j++) {
                assertComplexEquals("Mismatch at [" + i + "][" + j + "] in line " + lineNumber, 
                                    expected[0][i][j], expected[1][i][j], 
                                    javaIfftResult[0][i][j], javaIfftResult[1][i][j]);
            }
        }

        if (lineNumber % progressInterval == 0) {
            System.out.println("fft2d/ifft2d: Finished processing line " + lineNumber);
            double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
            System.out.println("Ifft2d Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000)  + " s");
        }
    }

    reader.close();
    System.out.println("ifft2d: Finished processing " + lineNumber + " lines.\n Average execution time: " + String.format("%.8f", (totalTime / 1e6 / numCalculations)/1000) + " ms");
}

    // Helper method for asserting equality with a tolerance
    private static void assertEquals(String message, double expected, double actual, double tolerance) {
        assertTrue(message + " - Expected: " + expected + ", Actual: " + actual, Math.abs(expected - actual) <= tolerance);
    }

    private void assertComplexEquals(String message, double expectedReal, double expectedImag, double actualReal, double actualImag) {
        final double EPSILON = 1e-9;
        assertTrue(message + " - Mismatch in real part. Expected: " + expectedReal + ", Actual: " + actualReal, 
                   Math.abs(expectedReal - actualReal) <= EPSILON);
        assertTrue(message + " - Mismatch in imaginary part. Expected: " + expectedImag + ", Actual: " + actualImag, 
                   Math.abs(expectedImag - actualImag) <= EPSILON);
    }
    

}
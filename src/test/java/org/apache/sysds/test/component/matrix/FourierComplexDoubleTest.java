package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.ComplexDouble;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import static org.apache.sysds.runtime.matrix.data.LibMatrixFourierComplexDouble.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

public class FourierComplexDoubleTest {

    // prior to executing this test it is necessary to run the Numpy Script in FourierTestData.py and add the generated file to the root of the project.
    @Test
    public void testFftWithNumpyData() throws IOException {
        String filename = "fft_data.csv"; // path to your CSV file
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        int lineNumber = 0;

        while ((line = reader.readLine()) != null) {
            lineNumber++;
            if (lineNumber % 1000 == 0) { // Print progress every 1000 lines
                System.out.println("Processing line: " + lineNumber);
            }

            String[] values = line.split(",");
            int n = values.length / 3;
            double[] input = new double[n];
            ComplexDouble[] expected = new ComplexDouble[n];
            ComplexDouble[] actual;

            for (int i = 0; i < n; i++) {
                input[i] = Double.parseDouble(values[i]);
                double real = Double.parseDouble(values[n + i]);
                double imag = Double.parseDouble(values[n * 2 + i]);
                expected[i] = new ComplexDouble(real, imag);
            }

            actual = fft_old(input);

            for (int i = 0; i < n; i++) {
                assertComplexEquals("Mismatch at index " + i + " in line " + lineNumber, expected[i], actual[i]);
            }
        }

        reader.close();
        System.out.println("Finished processing " + lineNumber + " lines.");
    }

    // prior to executing this test it is necessary to run the Numpy Script in FourierTestData.py and add the generated file to the root of the project.
    @Test
    public void testFftExecutionTime() throws IOException {
        String filename = "fft_data.csv"; // path to your CSV file
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        int lineNumber = 0;
        long totalTime = 0; // Total time for all FFT computations
        int numCalculations = 0; // Number of FFT computations

        while ((line = reader.readLine()) != null) {
            lineNumber++;
            String[] values = line.split(",");
            int n = values.length / 3;
            double[] input = new double[n];
            ComplexDouble[] expected = new ComplexDouble[n];

            for (int i = 0; i < n; i++) {
                input[i] = Double.parseDouble(values[i]);
                double real = Double.parseDouble(values[n + i]);
                double imag = Double.parseDouble(values[n * 2 + i]);
                expected[i] = new ComplexDouble(real, imag);
            }

            long startTime = System.nanoTime();
            fft_old(input);
            long endTime = System.nanoTime();
            if(lineNumber > 1000){
                totalTime += (endTime - startTime);
                numCalculations++;



                if (numCalculations % 5000 == 0) {
                    double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
                    System.out.println("\nAverage execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000) + " s \n");
                }
            }
        }

        reader.close();
        System.out.println("Finished processing " + lineNumber + " lines.");
    }



    // prior to executing this test it is necessary to run the Numpy Script in FourierTestData.py and add the generated file to the root of the project.
    @Test
    public void testIfftWithRealNumpyData() throws IOException {
        String filename = "ifft_data.csv"; // path to your IFFT data file
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        int lineNumber = 0;
        long totalTime = 0; // Total time for all IFFT computations
        int numCalculations = 0; // Number of IFFT computations

        while ((line = reader.readLine()) != null) {
            lineNumber++;
            String[] values = line.split(",");
            int n = values.length / 3;
            ComplexDouble[] originalInput = new ComplexDouble[n];
            ComplexDouble[] numpyIfftResult = new ComplexDouble[n];
            ComplexDouble[] javaIfftResult;

            for (int i = 0; i < n; i++) {
                originalInput[i] = new ComplexDouble(Double.parseDouble(values[i]), 0); // Original data
                double realPart = Double.parseDouble(values[n + i]);
                double imagPart = Double.parseDouble(values[n * 2 + i]);
                numpyIfftResult[i] = new ComplexDouble(realPart, imagPart); // NumPy IFFT result
            }

            long startTime = System.nanoTime();
            javaIfftResult = ifft(originalInput);
            long endTime = System.nanoTime();
            if(lineNumber > 1000){
                totalTime += (endTime - startTime);
                numCalculations++;
            }
            for (int i = 0; i < n; i++) {
                assertComplexEquals("Mismatch at index " + i + " in line " + lineNumber, numpyIfftResult[i], javaIfftResult[i]);
            }

            if (numCalculations % 5000 == 0) {
                double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
                System.out.println("Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000)  + " s");
                // System.out.println("input: ");
                // for(int i = 0; i < originalInput.length; i++ ){
                //     System.out.println(originalInput[i].toString());
                // }
                // System.out.println("output: " );
                // for(int i = 0; i < javaIfftResult.length; i++ ){
                //     System.out.println(javaIfftResult[i].toString());
                // }
            }
        }

        reader.close();
        System.out.println("Finished processing " + lineNumber + " lines.");
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
            ComplexDouble[] originalInput = new ComplexDouble[n];
            ComplexDouble[] numpyIfftResult = new ComplexDouble[n];
            ComplexDouble[] javaIfftResult;

            for (int i = 0; i < n; i++) {
                double realPartOriginal = Double.parseDouble(values[i]);
                double imagPartOriginal = Double.parseDouble(values[i + n]);
                originalInput[i] = new ComplexDouble(realPartOriginal, imagPartOriginal); // Original complex data

                double realPartIfft = Double.parseDouble(values[i + 2 * n]);
                double imagPartIfft = Double.parseDouble(values[i + 3 * n]);
                numpyIfftResult[i] = new ComplexDouble(realPartIfft, imagPartIfft); // NumPy IFFT result
            }

            long startTime = System.nanoTime();
            javaIfftResult = ifft(originalInput);
            long endTime = System.nanoTime();

            if(lineNumber > 1000){
                totalTime += (endTime - startTime);
                numCalculations++;
            }
            for (int i = 0; i < n; i++) {
                assertComplexEquals("Mismatch at index " + i + " in line " + lineNumber, numpyIfftResult[i], javaIfftResult[i]);
            }

            if (numCalculations % 5000 == 0) {
                double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
                System.out.println("Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000)  + " s");
                // System.out.println("input: ");
                //     for(int i = 0; i < originalInput.length; i++ ){
                //         System.out.println(originalInput[i].toString());
                //     }
                //     System.out.println("output: " );
                //     for(int i = 0; i < javaIfftResult.length; i++ ){
                //         System.out.println(javaIfftResult[i].toString());
                //     }

            }
        }

        reader.close();
        System.out.println("Finished processing " + lineNumber + " lines.");
    }



    // prior to executing this test it is necessary to run the Numpy Script in FourierTestData.py and add the generated file to the root of the project.
    @Test
    public void testIfftExecutionTime() throws IOException {
        String filename = "ifft_data.csv"; // path to your IFFT data file
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        int lineNumber = 0;
        long totalTime = 0; // Total time for all IFFT computations
        int numCalculations = 0; // Number of IFFT computations

        while ((line = reader.readLine()) != null) {
            lineNumber++;
            String[] values = line.split(",");
            int n = values.length / 3;
            ComplexDouble[] original = new ComplexDouble[n];
            ComplexDouble[] numpyResult = new ComplexDouble[n];

            for (int i = 0; i < n; i++) {
                original[i] = new ComplexDouble(Double.parseDouble(values[i]), 0);
                double realPart = Double.parseDouble(values[n + i]);
                double imagPart = Double.parseDouble(values[n * 2 + i]);
                numpyResult[i] = new ComplexDouble(realPart, imagPart);
            }

            long startTime = System.nanoTime();
            ifft(numpyResult);
            long endTime = System.nanoTime();
            if(lineNumber > 5000){
                totalTime += (endTime - startTime);
                numCalculations++;


                if (numCalculations % 1000 == 0) {
                    double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
                    System.out.println("Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000)  + " s");
                }
            }
        }

        reader.close();
        System.out.println("Finished processing " + lineNumber + " lines.");
    }


    @Test
    public void testSimpleIfft2d() {
        double[][] original = {{1, -2}, {3, -4}};
        ComplexDouble[][] complexInput = new ComplexDouble[original.length][original[0].length];
        for (int i = 0; i < original.length; i++) {
            for (int j = 0; j < original[0].length; j++) {
                complexInput[i][j] = new ComplexDouble(original[i][j], 0);
            }
        }

        ComplexDouble[][] fftResult = fft2d(complexInput);
        ComplexDouble[][] ifftResult = ifft2d(fftResult);

        for (int i = 0; i < original.length; i++) {
            for (int j = 0; j < original[0].length; j++) {
                assertEquals("Mismatch at [" + i + "][" + j + "]", original[i][j], ifftResult[i][j].re, 0.000001);
                assertEquals("Non-zero imaginary part at [" + i + "][" + j + "]", 0, ifftResult[i][j].im, 0.000001);
            }
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
        int progressInterval = 1000; // Print progress every 1000 lines

        while ((line = reader.readLine()) != null) {
            lineNumber++;
            String[] values = line.split(",");
            int halfLength = values.length / 4;
            int sideLength = (int) Math.sqrt(halfLength); // Assuming square matrix

            ComplexDouble[][] originalInput = new ComplexDouble[sideLength][sideLength];
            ComplexDouble[][] numpyFftResult = new ComplexDouble[sideLength][sideLength];

            for (int i = 0; i < halfLength; i++) {
                int row = i / sideLength;
                int col = i % sideLength;
                double realPartOriginal = Double.parseDouble(values[i]);
                double imagPartOriginal = Double.parseDouble(values[i + halfLength]);
                originalInput[row][col] = new ComplexDouble(realPartOriginal, imagPartOriginal);

                double realPartFft = Double.parseDouble(values[i + 2 * halfLength]);
                double imagPartFft = Double.parseDouble(values[i + 3 * halfLength]);
                numpyFftResult[row][col] = new ComplexDouble(realPartFft, imagPartFft);
            }

            long startTime = System.nanoTime();
            ComplexDouble[][] javaFftResult = fft2d(originalInput);
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
            numCalculations++;

            for (int i = 0; i < sideLength; i++) {
                for (int j = 0; j < sideLength; j++) {
                    assertComplexEquals("Mismatch at [" + i + "][" + j + "] in line " + lineNumber, numpyFftResult[i][j], javaFftResult[i][j]);
                }
            }

            if (lineNumber % progressInterval == 0) { // Print progress
                System.out.println("Processing line: " + lineNumber);
                if (numCalculations > 0) {
                    double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
                    System.out.println("Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000) + " s");
                }
            }
        }

        reader.close();
        System.out.println("Finished processing " + lineNumber + " lines. Average execution time: " + String.format("%.8f", (totalTime / 1e6 / numCalculations)/1000) + " s");
    }

    @Test
    public void testIfft2dWithNumpyData() throws IOException {
        String filename = "complex_ifft_2d_data.csv"; // path to your 2D IFFT data file
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        int lineNumber = 0;
        long totalTime = 0; // Total time for all IFFT 2D computations
        int numCalculations = 0; // Number of IFFT 2D computations
        int progressInterval = 10000; // Print progress every 1000 lines

        while ((line = reader.readLine()) != null) {
            lineNumber++;
            String[] values = line.split(",");
            int halfLength = values.length / 4;
            int sideLength = (int) Math.sqrt(halfLength); // Assuming square matrix

            ComplexDouble[][] originalInput = new ComplexDouble[sideLength][sideLength];
            ComplexDouble[][] numpyIfftResult = new ComplexDouble[sideLength][sideLength];

            for (int i = 0; i < halfLength; i++) {
                int row = i / sideLength;
                int col = i % sideLength;
                double realPartOriginal = Double.parseDouble(values[i]);
                double imagPartOriginal = Double.parseDouble(values[i + halfLength]);
                originalInput[row][col] = new ComplexDouble(realPartOriginal, imagPartOriginal);

                double realPartIfft = Double.parseDouble(values[i + 2 * halfLength]);
                double imagPartIfft = Double.parseDouble(values[i + 3 * halfLength]);
                numpyIfftResult[row][col] = new ComplexDouble(realPartIfft, imagPartIfft);
            }

            long startTime = System.nanoTime();
            ComplexDouble[][] javaIfftResult = ifft2d(originalInput);
            long endTime = System.nanoTime();
            if(lineNumber > 1000){
                totalTime += (endTime - startTime);
                numCalculations++;
            }

            for (int i = 0; i < sideLength; i++) {
                for (int j = 0; j < sideLength; j++) {
                    assertComplexEquals("Mismatch at [" + i + "][" + j + "] in line " + lineNumber, numpyIfftResult[i][j], javaIfftResult[i][j]);
                }
            }

            if (lineNumber % progressInterval == 0) { // Print progress
                System.out.println("Processing line: " + lineNumber);
                if (numCalculations > 0) {
                    double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
                    System.out.println("Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000)  + " s");
                }
            }
        }

        reader.close();
        System.out.println("Finished processing " + lineNumber + " lines. Average execution time: " + String.format("%.8f", (totalTime / 1e6 / numCalculations)/1000) + " ms");
    }

    // Helper method for asserting equality with a tolerance
    private static void assertEquals(String message, double expected, double actual, double tolerance) {
        assertTrue(message + " - Expected: " + expected + ", Actual: " + actual, Math.abs(expected - actual) <= tolerance);
    }

    private void assertComplexEquals(String message, ComplexDouble expected, ComplexDouble actual) {
        final double EPSILON = 0.000001;
        boolean realMatch = Math.abs(Math.abs(expected.re) - Math.abs(actual.re)) < EPSILON;
        boolean imagMatch = Math.abs(Math.abs(expected.im) - Math.abs(actual.im)) < EPSILON;

        if (realMatch && imagMatch) {
            if (Math.signum(expected.re) != Math.signum(actual.re)) {
                System.out.println(message + " - Real part is of opposite sign but otherwise correct");
            }
            if (Math.signum(expected.im) != Math.signum(actual.im)) {
                System.out.println(message + " - Imaginary part is of opposite sign but otherwise correct");
            }
        } else {
            assertTrue(message + " - Incorrect values", false);
        }
    }

    @Test
    public void simpleTest() {
        // tested with numpy
        double[] in = {0, 18, -15, 3};
        ComplexDouble[] expected = new ComplexDouble[4];
        expected[0] = new ComplexDouble(6, 0);
        expected[1] =  new ComplexDouble(15, -15);
        expected[2] = new ComplexDouble(-36, 0);
        expected[3] = new ComplexDouble(15, 15);

        ComplexDouble[] res = fft_old(in);
        for(ComplexDouble elem : res){
            System.out.println(elem);
        }

        assertArrayEquals(expected, res);
    }

    @Test
    public void notPowerOfTwoTest() {
        double[] in = {1, 2, 3, 4, 5};

        // see https://de.mathworks.com/help/matlab/ref/ifft.html
        ComplexDouble[] expected = new ComplexDouble[5];
        expected[0] = new ComplexDouble(15,0);
        expected[1] = new ComplexDouble(-2.5000,3.4410);
        expected[2] = new ComplexDouble(-2.5000,0.8123);
        expected[3] = new ComplexDouble(-2.5000, - 0.8123);
        expected[4] = new ComplexDouble(-2.5000, - 3.4410);

        ComplexDouble[] res = fft_old(in);
        for(ComplexDouble elem : res){
            System.out.println(elem);
        }
        assertArrayEquals(expected, res);
    }

    @Test
    public void simple2dTest() {
        // tested with matlab
        double[][] in = {{0, 18}, {-15, 3}};
        ComplexDouble[][] expected = new ComplexDouble[2][2];
        expected[0][0] = new ComplexDouble(6, 0);
        expected[0][1] = new ComplexDouble(-36, 0);
        expected[1][0] =  new ComplexDouble(30, 0);
        expected[1][1] =  new ComplexDouble(0, 0);

        ComplexDouble[][] res = fft2d(in);
        for(ComplexDouble[] row : res){
            for(ComplexDouble elem : row){
                System.out.println(elem);
            }
        }

        assertArrayEquals(expected, res);
    }

    @Test
    public void simple2dTest2() {
        double[][] in = {{0, 18, -15, 3}, {0, 18, -15, 3}};
        ComplexDouble[][] expected = new ComplexDouble[2][4];
        expected[0][0] = new ComplexDouble(12, 0);
        expected[0][1] = new ComplexDouble(30, -30);
        expected[0][2] = new ComplexDouble(-72, 0);
        expected[0][3] = new ComplexDouble(30, 30);
        expected[1][0] =  new ComplexDouble(0, 0);
        expected[1][1] =  new ComplexDouble(0, 0);
        expected[1][2] = new ComplexDouble(0, 0);
        expected[1][3] = new ComplexDouble(0, 0);

        ComplexDouble[][] res = fft2d(in);
        for(ComplexDouble[] row : res){
            for(ComplexDouble elem : row){
                System.out.println(elem);
            }
        }

        assertArrayEquals(expected, res);
    }

    @Test
    public void simple2dTest2SecondPart() {
        // ComplexDouble(15, -15) is the second (expected[1]) entry of simpleTest
        // this tests the col computation in fft2d for simple2dTest2

        ComplexDouble[] in = new ComplexDouble[2];
        in[0] = new ComplexDouble(15, -15);
        in[1] = new ComplexDouble(15, -15);

        ComplexDouble[] expected = new ComplexDouble[2];
        expected[0] = new ComplexDouble(30, -30);
        expected[1] = new ComplexDouble(0, 0);

        ComplexDouble[] res = fft(in);
        for (ComplexDouble elem : res) {
            System.out.println(elem);
        }

        assertArrayEquals(expected, res);
    }



    @Test
    public void testSimpleIfft1d() {
        double[] original = {1, -2, 3, -4};
        ComplexDouble[] complexInput = new ComplexDouble[original.length];
        for (int i = 0; i < original.length; i++) {
            complexInput[i] = new ComplexDouble(original[i], 0);
        }

        ComplexDouble[] fftResult = fft(complexInput);
        ComplexDouble[] ifftResult = ifft(fftResult);

        for (int i = 0; i < original.length; i++) {
            assertEquals("Mismatch at index " + i, original[i], ifftResult[i].re, 0.000001);
            assertEquals("Non-zero imaginary part at index " + i, 0, ifftResult[i].im, 0.000001);
        }
    }

}

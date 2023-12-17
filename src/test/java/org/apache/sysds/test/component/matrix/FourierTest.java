package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.ComplexDouble;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft2d;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.ifft;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.ifft2d;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class FourierTest {

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
    
            actual = fft(input);
    
            for (int i = 0; i < n; i++) {
                assertComplexEquals("Mismatch at index " + i + " in line " + lineNumber, expected[i], actual[i]);
            }
        }
    
        reader.close();
        System.out.println("Finished processing " + lineNumber + " lines.");
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
        double[] in = {0, 18, -15, 3};
        ComplexDouble[] expected = new ComplexDouble[4];
        expected[0] = new ComplexDouble(6, 0);
        expected[1] =  new ComplexDouble(15, 15);
        expected[2] = new ComplexDouble(-36, 0);
        expected[3] = new ComplexDouble(15, -15);

        ComplexDouble[] res = fft(in);
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

        ComplexDouble[] res = fft(in);
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

    // Helper method for asserting equality with a tolerance
    private static void assertEquals(String message, double expected, double actual, double tolerance) {
        assertTrue(message + " - Expected: " + expected + ", Actual: " + actual, Math.abs(expected - actual) <= tolerance);
    }

}

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.component.matrix;

import org.junit.Test;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;

import static org.junit.Assert.assertTrue;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.ifft;

public class FourierTestWithFiles {
	int progressInterval = 5000;

	// prior to executing the following tests it is necessary to run the Numpy Script in FourierTestData.py
	// and add the generated files to the root of the project.
	@Test
	public void testFftWithNumpyData() throws IOException {

		String filename = "fft_data.csv"; // Path to your CSV file
		String path = "./src/test/java/org/apache/sysds/test/component/matrix/";

		BufferedReader reader = new BufferedReader(new FileReader(path+filename));
		String line;
		int lineNumber = 0;
		long totalTime = 0; // Total time for all FFT computations
		int numCalculations = 0; // Number of FFT computations

		while ((line = reader.readLine()) != null) {
			lineNumber++;

			String[] values = line.split(",");
			int n = values.length / 3;

			double[] re =  new double[n];
			double[] im =  new double[n];

			double[][] expected = new double[2][n]; // First row for real, second row for imaginary parts

			for (int i = 0; i < n; i++) {
				re[i] = Double.parseDouble(values[i]);
				expected[0][i] = Double.parseDouble(values[n + i]); // Real part
				expected[1][i] = Double.parseDouble(values[n * 2 + i]); // Imaginary part
			}

			long startTime = System.nanoTime();
			fft(re, im, 1, n);
			long endTime = System.nanoTime();

			if(lineNumber > 1000){
				totalTime += (endTime - startTime);
				numCalculations++;

				if (numCalculations % progressInterval == 0) {
					double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
					System.out.println("fft(double[][][] in): Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000) + " s");
				}
			}

			double[][] actual = {re, im};

			// Validate the FFT results
			validateFftResults(expected, actual, lineNumber);
		}

		reader.close();
	}

	private void validateFftResults(double[][] expected, double[][] actual, int lineNumber) {

		int length = expected[0].length;

		for (int i = 0; i < length; i++) {
			double realActual = actual[0][i];
			double imagActual = actual[1][i];
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
		String path = "./src/test/java/org/apache/sysds/test/component/matrix/";

		BufferedReader reader = new BufferedReader(new FileReader(path+filename));
		String line;
		int lineNumber = 0;
		long totalTime = 0; // Total time for all FFT computations
		int numCalculations = 0; // Number of FFT computations

		while ((line = reader.readLine()) != null) {

			lineNumber++;
			String[] values = line.split(",");
			int n = values.length / 3;

			double[] re =  new double[n];
			double[] im =  new double[n];

			for (int i = 0; i < n; i++) {
				re[i] = Double.parseDouble(values[i]); // Real part
				im[i] = Double.parseDouble(values[n + i]); // Imaginary part
			}

			long startTime = System.nanoTime();

			fft(re, im, 1, n);

			long endTime = System.nanoTime();

			if(lineNumber > 1000){
				totalTime += (endTime - startTime);
				numCalculations++;

				if (numCalculations % progressInterval == 0) {
					double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
					System.out.println("fft_old(double[][][] in, boolean calcInv) Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime/1000) + " s");
				}
			}
		}

		reader.close();
	}

	@Test
	public void testFftExecutionTimeOfOneDimFFT() throws IOException {

		String filename = "fft_data.csv"; // Path to your CSV file
		String path = "./src/test/java/org/apache/sysds/test/component/matrix/";

		BufferedReader reader = new BufferedReader(new FileReader(path+filename));
		String line;
		int lineNumber = 0;
		long totalTime = 0; // Total time for all FFT computations
		int numCalculations = 0; // Number of FFT computations

		while ((line = reader.readLine()) != null) {

			lineNumber++;
			String[] values = line.split(",");
			int n = values.length / 2;

			double[] re =  new double[n];
			double[] im =  new double[n];

			for (int i = 0; i < n; i++) {
				re[i] = Double.parseDouble(values[i]); // Real part
				im[i] = Double.parseDouble(values[n + i]); // Imaginary part
			}

			long startTime = System.nanoTime();
			// one dimensional
			fft(re, im, 1, n);

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
		String path = "./src/test/java/org/apache/sysds/test/component/matrix/";

		BufferedReader reader = new BufferedReader(new FileReader(path+filename));
		String line;
		int lineNumber = 0;

		while ((line = reader.readLine()) != null) {

			lineNumber++;
			String[] values = line.split(",");
			int n = values.length / 3;

			double[] re =  new double[n];
			double[] im =  new double[n];

			double[][] expected = new double[2][n]; // First row for real, second row for imaginary parts

			for (int i = 0; i < n; i++) {
				re[i] = Double.parseDouble(values[i]); // Real part of input
				// Imaginary part of input is assumed to be 0
				expected[0][i] = Double.parseDouble(values[n + i]); // Real part of expected output
				expected[1][i] = Double.parseDouble(values[n * 2 + i]); // Imaginary part of expected output
			}

			ifft(re, im, 1, n); // Perform IFFT

			double[][] actual = new double[][]{re, im};
			// Validate the IFFT results
			validateFftResults(expected, actual, lineNumber);
		}

		reader.close();

	}

	@Test
	public void testIfftWithComplexNumpyData() throws IOException {

		String filename = "complex_ifft_data.csv"; // Adjusted path to your IFFT data file with complex inputs
		String path = "./src/test/java/org/apache/sysds/test/component/matrix/";

		BufferedReader reader = new BufferedReader(new FileReader(path+filename));
		String line;
		int lineNumber = 0;
		long totalTime = 0; // Total time for all IFFT computations
		int numCalculations = 0; // Number of IFFT computations

		while ((line = reader.readLine()) != null) {
			lineNumber++;
			String[] values = line.split(",");
			int n = values.length / 4; // Adjusted for complex numbers

			// Real and imaginary parts
			double[] re =  new double[n];
			double[] im =  new double[n];

			double[][] expected = new double[2][n]; // Expected real and imaginary parts

			for (int i = 0; i < n; i++) {
				re[i] = Double.parseDouble(values[i]); // Real part of input
				im[i] = Double.parseDouble(values[i + n]); // Imaginary part of input
				expected[0][i] = Double.parseDouble(values[i + 2 * n]); // Expected real part
				expected[1][i] = Double.parseDouble(values[i + 3 * n]); // Expected imaginary part
			}

			long startTime = System.nanoTime();

			ifft(re, im, 1, n); // Perform IFFT

			long endTime = System.nanoTime();

			if (lineNumber > 1000) {
				totalTime += (endTime - startTime);
				numCalculations++;
			}

			double[][] actual = new double[][]{re, im};
			// Validate the IFFT results
			validateComplexIFftResults(expected, actual, lineNumber);

			if (lineNumber % progressInterval == 0) {
				double averageTime = (totalTime / 1e6) / numCalculations; // Average time in milliseconds
				System.out.println("ifft: Average execution time after " + numCalculations + " calculations: " + String.format("%.8f", averageTime / 1000) + " s");
			}
		}

		reader.close();
	}

	private void validateComplexIFftResults(double[][] expected, double[][] actual, int lineNumber) {

		int length = expected[0].length;

		for (int i = 0; i < length; i++) {
			double realActual = actual[0][i];
			double imagActual = actual[1][i];
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
		String path = "./src/test/java/org/apache/sysds/test/component/matrix/";

		BufferedReader reader = new BufferedReader(new FileReader(path+filename));
		String line;
		int lineNumber = 0;
		long totalTime = 0; // Total time for all FFT 2D computations
		int numCalculations = 0; // Number of FFT 2D computations

		while ((line = reader.readLine()) != null) {
			lineNumber++;
			String[] values = line.split(",");
			int halfLength = values.length / 4;
			int sideLength = (int) Math.sqrt(halfLength); // Assuming square matrix

			// Real and imaginary parts
			double[] re =  new double[sideLength*sideLength];
			double[] im =  new double[sideLength*sideLength];

			double[][][] expected = new double[2][sideLength][sideLength];

			for (int i = 0; i < halfLength; i++) {
				int row = i / sideLength;
				int col = i % sideLength;

				// i == row*sideLength+col?!
				re[row*sideLength+col] = Double.parseDouble(values[i]);
				im[row*sideLength+col] = Double.parseDouble(values[i + halfLength]);
				expected[0][row][col] = Double.parseDouble(values[i + 2 * halfLength]);
				expected[1][row][col] = Double.parseDouble(values[i + 3 * halfLength]);
			}

			long startTime = System.nanoTime();
			// Use your fft2d implementation
			fft(re, im, sideLength, sideLength);
			//double[][][] javaFftResult = fft_old(input, false); // Use your fft2d implementation
			long endTime = System.nanoTime();
			totalTime += (endTime - startTime);
			numCalculations++;

			for (int i = 0; i < sideLength; i++) {
				for (int j = 0; j < sideLength; j++) {
					assertComplexEquals("Mismatch at [" + i + "][" + j + "] in line " + lineNumber,
							expected[0][i][j], expected[1][i][j],
							re[i*sideLength+j], im[i*sideLength+j]);
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
		String path = "./src/test/java/org/apache/sysds/test/component/matrix/";

		BufferedReader reader = new BufferedReader(new FileReader(path+filename));
		String line;
		int lineNumber = 0;
		long totalTime = 0; // Total time for all IFFT 2D computations
		int numCalculations = 0; // Number of IFFT 2D computations

		while ((line = reader.readLine()) != null) {

			lineNumber++;
			String[] values = line.split(",");
			int halfLength = values.length / 4;
			int sideLength = (int) Math.sqrt(halfLength); // Assuming square matrix

			// Real and imaginary parts
			double[] re =  new double[sideLength*sideLength];
			double[] im =  new double[sideLength*sideLength];

			double[][][] expected = new double[2][sideLength][sideLength];

			for (int i = 0; i < halfLength; i++) {
				int row = i / sideLength;
				int col = i % sideLength;

				re[row*sideLength+col] = Double.parseDouble(values[i]);
				im[row*sideLength+col] = Double.parseDouble(values[i + halfLength]);

				expected[0][row][col] = Double.parseDouble(values[i + 2 * halfLength]);
				expected[1][row][col] = Double.parseDouble(values[i + 3 * halfLength]);
			}

			long startTime = System.nanoTime();

			// Use your ifft2d implementation
			ifft(re, im, sideLength, sideLength);

			long endTime = System.nanoTime();
			if(lineNumber > 1000){
				totalTime += (endTime - startTime);
				numCalculations++;
			}

			for (int i = 0; i < sideLength; i++) {
				for (int j = 0; j < sideLength; j++) {
					assertComplexEquals("Mismatch at [" + i + "][" + j + "] in line " + lineNumber,
							expected[0][i][j], expected[1][i][j],
							re[i*sideLength+j], im[i*sideLength+j]);
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

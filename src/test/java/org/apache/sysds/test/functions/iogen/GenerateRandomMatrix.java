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

package org.apache.sysds.test.functions.iogen;

import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Ignore;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class GenerateRandomMatrix extends AutomatedTestBase {

	protected final static String TEST_DIR = "functions/iogen/";

	@Override
	public void setUp() {
	
	}

	// Generate Random Symmetric 2D Data
	@SuppressWarnings("unused")
	private double[][] generateRandomSymmetric(int size, double min, double max, double sparsity, boolean isSkew) {
		double[][] sampleMatrix = getRandomMatrix(size, size, min, max, sparsity, 714);
		int conf = isSkew ? -1 : 1;
		for(int i = 0; i < size; i++) {
			for(int j = 0; j <= i; j++) {

				if(i != j)
					sampleMatrix[i][j] = sampleMatrix[j][i] * conf;
				else
					sampleMatrix[i][j] = 0;
			}
		}
		return sampleMatrix;
	}

	// Generate Random 2D Data
	private double[][] generateRandom2DData(int nrows, int ncols, double min, double max, double sparsity) {
		double[][] data = getRandomMatrix(nrows, ncols, min, max, sparsity, 714);
		return data;
	}

	// Generate Random Symmetric 2D Data
	protected double[][] getSymmetric2DData(double[][] data, int size, boolean isSkew) {
		double[][] result = new double[size][size];
		int conf = isSkew ? -1 : 1;
		// Update Data
		for(int i = 0; i < size; i++) {
			for(int j = 0; j <= i; j++) {
				if(i != j) {
					result[i][j] = data[i][j] * conf;
					result[j][i] = result[i][j];
				}
				else
					result[i][j] = 0;
			}
		}
		return result;
	}

	// Load Random 2D data from file
	@SuppressWarnings("unused")
	private static double[][] load2DData(String fileName, int nrows, int ncols) throws Exception {

		Path path = Paths.get(fileName);
		FileChannel inStreamRegularFile = FileChannel.open(path);
		int bufferSize = ncols * 8;

		double[][] result = new double[nrows][ncols];
		try {
			for(int r = 0; r < nrows; r++) {
				inStreamRegularFile.position((long) r * ncols * 8);
				ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize);
				inStreamRegularFile.read(buffer);
				buffer.flip();

				for(int c = 0; c < ncols; c++) {
					result[r][c] = buffer.getDouble();
				}
			}
			inStreamRegularFile.close();
		}
		catch(IOException e) {
			throw new Exception("Can't read matrix from ByteArray", e);
		}
		return result;
	}

	// Write 2D Data in CSV format
	private static void writeInCSVFormat(double[][] data, int nrows, int ncols, String fileName, String separator,
		String[] naString) throws Exception {

		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName + ".raw"));

		for(int r = 0; r < nrows; r++) {
			StringBuilder row = new StringBuilder();
			for(int c = 0; c < ncols; c++) {
				if(data[r][c] != 0) {
					row.append(data[r][c]).append(separator);
				}
				else {
					Random rn = new Random();
					int rni = rn.nextInt(naString.length);
					row.append(naString[rni]).append(separator);
				}
			}
			String srow = row.substring(0, row.length() - separator.length());
			writer.write(srow);
			if(r != nrows - 1)
				writer.write("\n");
		}
		writer.close();
	}

	// Write 2D in LIBSVM format
	private static double[][] writeInLIBSVMFormat(int firstIndex, double[][] data, int nrows, int ncols, String fileName,
		String separator, String indexSeparator) throws IOException {

		int indexRow = 0;
		double[][] sampleMatrix = new double[2 * nrows][ncols + 1];
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName + ".raw"));

		for(int r = 0; r < nrows; r++) {
			StringBuilder row1 = new StringBuilder();
			StringBuilder row2 = new StringBuilder();
			row1.append("+1");

			for(int c = 0; c < ncols; c++) {
				if(data[r][c] > 0) {
					sampleMatrix[indexRow][c] = data[r][c];
					row1.append(separator).append(c + firstIndex).append(indexSeparator).append(data[r][c]);
				}
				else {
					sampleMatrix[indexRow][c] = 0;
				}
			}
			sampleMatrix[indexRow++][ncols] = 1;

			row2.append("-1");
			for(int c = 0; c < ncols; c++) {
				if(data[r][c] < 0) {
					sampleMatrix[indexRow][c] = data[r][c];
					row2.append(separator).append(c + firstIndex).append(indexSeparator).append(data[r][c]);
				}
				else {
					sampleMatrix[indexRow][c] = 0;
				}
			}

			sampleMatrix[indexRow++][ncols] = -1;
			writer.write(row1.toString());
			writer.write("\n");
			writer.write(row2.toString());
			if(r != nrows - 1)
				writer.append("\n");
		}
		writer.close();
		return sampleMatrix;
	}

	// Write in Matrix Market Format
	private static void writeInMatrixMarketFormat(int firstIndex, double[][] data, int nrows, int ncols, String fileName,
		String separator) throws IOException {

		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName + ".raw"));

		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				if(data[r][c] != 0) {
					String rs = (r + firstIndex) + separator + (c + firstIndex) + separator + data[r][c];
					writer.write(rs);
					if(r != nrows - 1 || c != ncols - 1)
						writer.write("\n");
				}
			}
		}
		writer.close();
	}

	private static void writeInSymmetricMatrixMarketFormat(int firstIndex, double[][] data, String fileName, int size,
		String separator, boolean isUpperTriangular) throws IOException {

		int start, end;
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName + ".raw"));
		for(int r = 0; r < size; r++) {
			if(isUpperTriangular) {
				start = r;
				end = size;
			}
			else {
				start = 0;
				end = r + 1;
			}
			for(int c = start; c < end; c++) {
				if(data[r][c] != 0) {
					String rs = (r + firstIndex) + separator + (c + firstIndex) + separator + data[r][c];
					writer.write(rs);
					if(r != size - 1 || c != size - 1)
						writer.write("\n");
				}
			}
		}
		writer.close();
	}

	// Write 2D Data in CSV format
	private static void writeSampleMatrix(double[][] sample, String fileName, int nrows, int ncols) throws Exception {

		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName + ".matrix"));
		for(int r = 0; r < nrows; r++) {
			StringBuilder row = new StringBuilder();
			for(int c = 0; c < ncols; c++) {
				row.append(sample[r][c]);
				if(c != ncols - 1)
					row.append(",");
			}
			writer.write(row.toString());
			if(r != nrows - 1)
				writer.write("\n");
		}
		writer.close();
	}

	@Test
	@Ignore
	public void generateDataset() throws Exception {
		int nrows = 5000;
		int ncols = 5000;
		double sparsity = 1;
		String HOME = SCRIPT_DIR + TEST_DIR;
		String[] naString = {"Nan", "NAN", "", "inf", "null", "NULL"};
		double[][] data = generateRandom2DData(nrows, ncols, -100, 100, sparsity);
		saveData(data, nrows, ncols, " ", ":", naString, HOME + "/data/", sparsity, false);

		for(int r = 10; r <= 100; r += 10) {
			saveData(data, r, r, " ", ":", naString, HOME + "/samples/", sparsity, true);
		}
	}

	private void saveData(double[][] data, int nrows, int ncols, String separator, String indexSeparator,
		String[] naStrings, String HOME, double sparsity, boolean saveSampleMatrix) throws Exception {

		String baseFileName = "_nrows_" + nrows + "_ncols_" + ncols + "_sparsity_" + sparsity;

		String csv = HOME + "CSV" + baseFileName;

		String libsvmFirstZero = HOME + "LIBSVM-FZ" + baseFileName;
		String libsvmFirstOne = HOME + "LIBSVM-FO" + baseFileName;

		String mmFirstZero = HOME + "MM-FZ" + baseFileName;
		String mmFirstOne = HOME + "MM-FO" + baseFileName;

		String mmFirstZeroSymUT = HOME + "MM-FZ-SYM-UT" + baseFileName;
		String mmFirstZeroSymLT = HOME + "MM-FZ-SYM-LT" + baseFileName;
		String mmFirstOneSymUT = HOME + "MM-FO-SYM-UT" + baseFileName;
		String mmFirstOneSymLT = HOME + "MM-FO-SYM-LT" + baseFileName;

		String mmFirstZeroSkewUT = HOME + "MM-FZ-SKEW-UT" + baseFileName;
		String mmFirstZeroSkewLT = HOME + "MM-FZ-SKEW-LT" + baseFileName;
		String mmFirstOneSkewUT = HOME + "MM-FO-SKEW-UT" + baseFileName;
		String mmFirstOneSkewLT = HOME + "MM-FO-SKEW-LT" + baseFileName;

		// Write all data as a source dataset
		writeInCSVFormat(data, nrows, ncols, csv, separator, naStrings);
		double[][] libsvm = writeInLIBSVMFormat(0, data, nrows, ncols, libsvmFirstZero, separator, indexSeparator);
		writeInLIBSVMFormat(1, data, nrows, ncols, libsvmFirstOne, separator, indexSeparator);
		writeInMatrixMarketFormat(0, data, nrows, ncols, mmFirstZero, separator);
		writeInMatrixMarketFormat(1, data, nrows, ncols, mmFirstOne, separator);

		if(saveSampleMatrix) {
			writeSampleMatrix(data, csv, nrows, ncols);
			writeSampleMatrix(libsvm, HOME + "LIBSVM" + baseFileName, 2 * nrows, ncols + 1);
			writeSampleMatrix(data, HOME + "MM" + baseFileName, nrows, ncols);
		}

		// Write MM Symmetric and Skew
		if(nrows == ncols) {
			double[][] mm = getSymmetric2DData(data, nrows, false);
			writeInSymmetricMatrixMarketFormat(0, mm, mmFirstZeroSymUT, ncols, separator, true);
			writeInSymmetricMatrixMarketFormat(1, mm, mmFirstOneSymUT, ncols, separator, true);
			writeInSymmetricMatrixMarketFormat(0, mm, mmFirstZeroSymLT, ncols, separator, false);
			writeInSymmetricMatrixMarketFormat(1, mm, mmFirstOneSymLT, ncols, separator, false);
			if(saveSampleMatrix)
				writeSampleMatrix(mm, HOME + "MM-SYM" + baseFileName, nrows, nrows);

			mm = getSymmetric2DData(data, nrows, true);
			writeInSymmetricMatrixMarketFormat(0, mm, mmFirstZeroSkewUT, ncols, separator, true);
			writeInSymmetricMatrixMarketFormat(1, mm, mmFirstOneSkewUT, ncols, separator, true);
			writeInSymmetricMatrixMarketFormat(0, mm, mmFirstZeroSkewLT, ncols, separator, false);
			writeInSymmetricMatrixMarketFormat(1, mm, mmFirstOneSkewLT, ncols, separator, false);
			if(saveSampleMatrix)
				writeSampleMatrix(mm, HOME + "MM-SKEW" + baseFileName, nrows, nrows);
		}
	}
}

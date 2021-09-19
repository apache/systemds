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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Ignore;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class GenerateRandomFrame extends AutomatedTestBase {

	protected final static String TEST_DIR = "functions/iogen/";

	@Override
	public void setUp() {
	
	}


	protected Types.ValueType[] types = {Types.ValueType.STRING, Types.ValueType.INT32, Types.ValueType.INT64,
		Types.ValueType.FP32, Types.ValueType.FP64};

	protected String[][] generateRandomData(Types.ValueType[] types, int nrows, int ncols, double min, double max,
		double sparsity, String[] naStrings) {
		String[][] data = new String[nrows][ncols];
		for(int i = 0; i < ncols; i++) {
			if(types[i] == Types.ValueType.STRING)
				generateRandomString(nrows, 100, naStrings, sparsity, data, i);
			if(types[i].isNumeric()) {
				generateRandomNumeric(nrows, types[i], min, max, naStrings, sparsity, data, i);
			}
		}
		return data;
	}

	protected String getRandomString(int length) {
		//String alphabet1 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890";
		String alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
		StringBuilder salt = new StringBuilder();
		Random rnd = new Random();
		while(salt.length() < length) { // length of the random string.
			int index = (int) (rnd.nextFloat() * alphabet.length());
			salt.append(alphabet.charAt(index));
		}
		String saltStr = salt.toString();
		return saltStr;

	}

	protected void generateRandomString(int size, int maxStringLength, String[] naStrings, double sparsity,
		String[][] data, int colIndex) {

		double[][] lengths = getRandomMatrix(size, 1, 10, maxStringLength, sparsity, 714);

		for(int i = 0; i < size; i++) {
			int length = (int) lengths[i][0];
			if(length > 0) {
				String generatedString = getRandomString(length);
				data[i][colIndex] = generatedString;
			}
			else {
				data[i][colIndex] = null;
			}
		}
	}

	@SuppressWarnings("incomplete-switch")
	protected void generateRandomNumeric(int size, Types.ValueType type, double min, double max, String[] naStrings,
		double sparsity, String[][] data, int colIndex) {

		double[][] randomData = getRandomMatrix(size, 1, min, max, sparsity, -1);
		for(int i = 0; i < size; i++) {
			if(randomData[i][0] != 0) {
				Object o = null;
				switch(type) {
					case INT32:
						o = UtilFunctions.objectToObject(type, (int) randomData[i][0]);
						break;
					case INT64:
						o = UtilFunctions.objectToObject(type, (long) randomData[i][0]);
						break;
					case FP32:
						o = UtilFunctions.objectToObject(type, (float) randomData[i][0]);
						break;
					case FP64:
						o = UtilFunctions.objectToObject(type, randomData[i][0]);
						break;
				}
				String s = UtilFunctions.objectToString(o);
				data[i][colIndex] = s;
			}
			else {
				data[i][colIndex] = "0";
			}
		}
	}
	// Write 2D Data in CSV format
	private static void writeInCSVFormat(String[][] data, int nrows, int ncols, String fileName, String separator,
		String[] naString) throws Exception {

		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName + ".raw"));

		for(int r = 0; r < nrows; r++) {
			StringBuilder row = new StringBuilder();
			for(int c = 0; c < ncols; c++) {
				row.append(data[r][c]);
				if(c != ncols - 1)
					row.append(separator);
			}
			writer.write(row.toString());
			if(r != nrows - 1)
				writer.write("\n");
		}
		writer.close();
	}

	// Write 2D in LIBSVM format
	private static String[][] writeInLIBSVMFormat(int firstIndex,Types.ValueType[] schema, String[][] data, int nrows, int ncols, String fileName,
		String separator, String indexSeparator) throws IOException {

		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName + ".raw"));
		int mid = ncols/2;
		String[][] dataLibSVM = new String[2 * nrows][ncols+1];
		StringBuilder sb = new StringBuilder();
		int indexRow = 0;
		for(int r = 0; r < nrows; r++) {
			StringBuilder row1 = new StringBuilder();
			StringBuilder row2 = new StringBuilder();
			row1.append("+1");
			for(int c = 0; c < ncols; c++) {
				if(mid > c) {
					dataLibSVM[indexRow][c] = data[r][c];
					row1.append(separator).append(c + firstIndex).append(indexSeparator).append(data[r][c]);
				}
				else {
					if(schema[c].isNumeric() || schema[c] == Types.ValueType.BOOLEAN){
						dataLibSVM[indexRow][c] = "0";
					}
					else if(schema[c] == Types.ValueType.STRING)
						dataLibSVM[indexRow][c] = "";
				}
			}
			dataLibSVM[indexRow++][ncols] = "+1";

			row2.append("-1");
			for(int c = 0; c < ncols ; c++) {
				if(mid <= c) {
					dataLibSVM[indexRow][c] = data[r][c];
					row2.append(separator).append(c + firstIndex).append(indexSeparator).append(data[r][c]);
				}
				else {
					if(schema[c].isNumeric() || schema[c] == Types.ValueType.BOOLEAN){
						dataLibSVM[indexRow][c] = "0";
					}
					else if(schema[c] == Types.ValueType.STRING)
						dataLibSVM[indexRow][c] = "";
				}
			}
			dataLibSVM[indexRow++][ncols] = "-1";
			writer.write(row1.toString());
			writer.write("\n");
			writer.write(row2.toString());
			if(r != nrows - 1)
				writer.append("\n");

			sb.append(row1).append("\n");
			sb.append(row2);
			if(r != nrows - 1)
				sb.append("\n");
		}
		writer.close();
		return dataLibSVM;
	}

	// Write in Matrix Market Format
	private static void writeInMatrixMarketFormat(int firstIndex, String[][] data, int nrows, int ncols, String fileName,
		String separator) throws IOException {

		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName + ".raw"));
		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				if(data[r][c] != null && !data[r][c].equals("0")) {
					String rs = (r + firstIndex) + separator + (c + firstIndex) + separator + data[r][c];
					writer.write(rs);
					if(r != nrows - 1 || c != ncols - 1)
						writer.write("\n");
				}
			}
		}
		writer.close();
	}



	// Write 2D Data in CSV format
	private static void writeSampleFrame(Types.ValueType[] schema, String[][] sample, String fileName, int nrows, int ncols)
		throws Exception {
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName + ".frame"));
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

		writer = new BufferedWriter(new FileWriter(fileName + ".schema"));
		StringBuilder sb = new StringBuilder();
		for(int c = 0; c < ncols; c++) {
			sb.append(schema[c]);
			if(c != ncols - 1)
				sb.append(",");
		}

		writer.write(sb.toString());
		writer.close();
	}

	@Test
	@Ignore
	public void generateDataset() throws Exception {
		int nrows = 5000;
		int ncols = 5000;
		double sparsity = 1;
		String HOME = SCRIPT_DIR + TEST_DIR;
		String[] naStrings = {"Nan", "NAN", "", "inf", "null", "NULL"};
		String[] names = new String[ncols];
		Types.ValueType[] schema = new Types.ValueType[ncols];

		for(int i = 0; i < nrows; i++) {
			names[i] = "C_" + i;
			Random rn = new Random();
			int rnt = rn.nextInt(types.length);
			schema[i] = types[rnt];
		}
		String[][] data = generateRandomData(schema, nrows, ncols, -100, 100, sparsity, naStrings);
		saveData(schema, data, nrows, ncols, " ", ":", naStrings, HOME + "/data/", sparsity, false);

		for(int r = 10; r <= 100; r += 10) {
			saveData(schema, data, r, r, " ", ":", naStrings, HOME + "/samples/", sparsity, true);
		}

		BufferedWriter writer = new BufferedWriter(new FileWriter(HOME+"/data/data"+"_nrows_" + nrows + "_ncols_" + ncols + "_sparsity_" + sparsity + ".schema"));
		StringBuilder sb = new StringBuilder();
		for(int c = 0; c < ncols; c++) {
			sb.append(schema[c]);
			if(c != ncols - 1)
				sb.append(",");
		}
		writer.write(sb.toString());
		writer.close();
	}

	private static void saveData(Types.ValueType[] schema, String[][] data, int nrows, int ncols, String separator,
		String indexSeparator, String[] naStrings, String HOME, double sparsity, boolean saveSampleFrame)
		throws Exception {

		String baseFileName = "_nrows_" + nrows + "_ncols_" + ncols + "_sparsity_" + sparsity;

		String csv = HOME + "CSV" + baseFileName;

		String libsvmFirstZero = HOME + "LIBSVM-FZ" + baseFileName;
		String libsvmFirstOne = HOME + "LIBSVM-FO" + baseFileName;

		String mmFirstZero = HOME + "MM-FZ" + baseFileName;
		String mmFirstOne = HOME + "MM-FO" + baseFileName;

		// Write all data as a source dataset
		writeInCSVFormat(data, nrows, ncols, csv, separator, naStrings);
		String[][] libsvm = writeInLIBSVMFormat(0,schema, data, nrows, ncols, libsvmFirstZero, separator, indexSeparator);
		writeInLIBSVMFormat(1,schema, data, nrows, ncols, libsvmFirstOne, separator, indexSeparator);
		writeInMatrixMarketFormat(0, data, nrows, ncols, mmFirstZero, separator);
		writeInMatrixMarketFormat(1, data, nrows, ncols, mmFirstOne, separator);

		if(saveSampleFrame) {
			writeSampleFrame(schema,data, csv, nrows, ncols);
			Types.ValueType[] libsvmSchema = new Types.ValueType[ncols+1];
			for(int i=0;i<ncols;i++)
				libsvmSchema[i] = schema[i];
			libsvmSchema[ncols] = Types.ValueType.INT32;
			writeSampleFrame(libsvmSchema,libsvm, HOME + "LIBSVM" + baseFileName, 2 * nrows, ncols + 1);
			writeSampleFrame(schema,data, HOME + "MM" + baseFileName, nrows, ncols);
		}
	}
}

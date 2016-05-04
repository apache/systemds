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

package org.apache.sysml.test.utils;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Locale;
import java.util.Random;
import java.util.StringTokenizer;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.BinaryMatrixCharacteristics;

import junit.framework.Assert;


/**
 * <p>
 * Provides methods to easily create tests. Implemented methods can be used for
 * </p>
 * <ul>
 * <li>data comparison</li>
 * <li>test data generation</li>
 * <li>writing files</li>
 * <li>reading files</li>
 * <li>clean up</li>
 * </ul>
 */
public class TestUtils 
{
	
	/** job configuration used for file system access */
	public static Configuration conf = new Configuration();

	/** global random generator for default seed */
	public static Random random = new Random(System.currentTimeMillis());

	/** internal buffer to store assertion information */
	private static ArrayList<String> _AssertInfos = new ArrayList<String>();
	private static boolean _AssertOccured = false;

	/**
	 * <p>
	 * Compares to arrays for equality. The elements in the array can be in
	 * different order.
	 * </p>
	 * 
	 * @param expecteds
	 *            expected values
	 * @param actuals
	 *            actual values
	 */
	public static void assertInterchangedArraysEquals(String[] expecteds, String[] actuals) {
		assertEquals("different number of elements in arrays", expecteds.length, actuals.length);
		ArrayList<Integer> foundIndexes = new ArrayList<Integer>();
		expactation: for (int i = 0; i < expecteds.length; i++) {
			for (int j = 0; j < actuals.length; j++) {
				if (expecteds[i] == actuals[j] && !foundIndexes.contains(Integer.valueOf(j))) {
					foundIndexes.add(Integer.valueOf(j));
					continue expactation;
				}
			}
			fail("Missing element " + expecteds[i]);
		}
	}

	/**
	 * <p>
	 * Compares to arrays for equality. The elements in the array can be in
	 * different order.
	 * </p>
	 * 
	 * @param expecteds
	 *            expected values
	 * @param actuals
	 *            actual values
	 */
	public static void assertInterchangedArraysEquals(int[] expecteds, int[] actuals) {
		assertEquals("different number of elements in arrays", expecteds.length, actuals.length);
		ArrayList<Integer> foundIndexes = new ArrayList<Integer>();
		expactation: for (int i = 0; i < expecteds.length; i++) {
			for (int j = 0; j < actuals.length; j++) {
				if (expecteds[i] == actuals[j] && !foundIndexes.contains(Integer.valueOf(j))) {
					foundIndexes.add(Integer.valueOf(j));
					continue expactation;
				}
			}
			fail("Missing element " + expecteds[i]);
		}
	}

	/**
	 * <p>
	 * Compares to arrays for equality. The elements in the array can be in
	 * different order.
	 * </p>
	 * 
	 * @param expecteds
	 *            expected values
	 * @param actuals
	 *            actual values
	 */
	public static void assertInterchangedArraysEquals(double[] expecteds, double[] actuals) {
		assertEquals("different number of elements in arrays", expecteds.length, actuals.length);
		ArrayList<Integer> foundIndexes = new ArrayList<Integer>();
		expactation: for (int i = 0; i < expecteds.length; i++) {
			for (int j = 0; j < actuals.length; j++) {
				if (expecteds[i] == actuals[j] && !foundIndexes.contains(Integer.valueOf(j))) {
					foundIndexes.add(Integer.valueOf(j));
					continue expactation;
				}
			}
			fail("Missing element " + expecteds[i]);
		}
	}
	
	/* Compare expected scalar generated by Java with actual scalar generated by DML */
	public static void compareDMLScalarWithJavaScalar(String expectedFile, String actualFile, double epsilon) {
		try {
			String lineExpected = null;
			String lineActual = null;
			FileSystem fs = FileSystem.get(conf);
			
			Path compareFile = new Path(expectedFile);
			FSDataInputStream fsin = fs.open(compareFile);
			BufferedReader compareIn = new BufferedReader(new InputStreamReader(fsin));
			lineExpected = compareIn.readLine();
			compareIn.close();
			
			Path outFile = new Path(actualFile);
			FSDataInputStream fsout = fs.open(outFile);
			BufferedReader outIn = new BufferedReader(new InputStreamReader(fsout));
			lineActual = outIn.readLine();
			outIn.close();

			assertTrue(expectedFile + ": " + lineExpected + " vs " + actualFile + ": " + lineActual, 
					   lineActual.equals(lineExpected));
		} catch (IOException e) {
			fail("unable to read file: " + e.getMessage());
		}
	}
	
	/**
	 * Compares contents of an expected file with the actual file, where rows may be permuted
	 * @param expectedFile
	 * @param actualDir
	 * @param epsilon
	 */
	public static void compareDMLMatrixWithJavaMatrixRowsOutOfOrder(String expectedFile, String actualDir, double epsilon)
	{
		try {
			FileSystem fs = FileSystem.get(conf);
			Path outDirectory = new Path(actualDir);
			Path compareFile = new Path(expectedFile);
			FSDataInputStream fsin = fs.open(compareFile);
			BufferedReader compareIn = new BufferedReader(new InputStreamReader(fsin));
			
			HashMap<CellIndex, Double> expectedValues = new HashMap<CellIndex, Double>();
			String line;
			while ((line = compareIn.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(line, " ");
				int i = Integer.parseInt(st.nextToken());
				int j = Integer.parseInt(st.nextToken());
				double v = Double.parseDouble(st.nextToken());
				expectedValues.put(new CellIndex(i, j), v);
			}
			compareIn.close();

			HashMap<CellIndex, Double> actualValues = new HashMap<CellIndex, Double>();

			FileStatus[] outFiles = fs.listStatus(outDirectory);

			for (FileStatus file : outFiles) {
				FSDataInputStream fsout = fs.open(file.getPath());
				BufferedReader outIn = new BufferedReader(new InputStreamReader(fsout));
				
				while ((line = outIn.readLine()) != null) {
					StringTokenizer st = new StringTokenizer(line, " ");
					int i = Integer.parseInt(st.nextToken());
					int j = Integer.parseInt(st.nextToken());
					double v = Double.parseDouble(st.nextToken());
					actualValues.put(new CellIndex(i, j), v);
				}
				outIn.close();
			}

			ArrayList<Double> e_list = new ArrayList <Double>();
			for (CellIndex index : expectedValues.keySet()) {
				Double expectedValue = expectedValues.get(index);
				if(expectedValue != 0.0)
					e_list.add(expectedValue);
			}
			
			ArrayList<Double> a_list = new ArrayList <Double>();
			for (CellIndex index : actualValues.keySet()) {
				Double actualValue = actualValues.get(index);
				if(actualValue != 0.0)
					a_list.add(actualValue);
			}
			
			Collections.sort(e_list);
			Collections.sort(a_list);
			
			assertTrue("Matrix nzs not equal", e_list.size() == a_list.size());
			for(int i=0; i < e_list.size(); i++)
			{
				assertTrue("Matrix values not equals", Math.abs(e_list.get(i) - a_list.get(i)) <= epsilon);
			}
			
		} catch (IOException e) {
			fail("unable to read file: " + e.getMessage());
		}
	}
	
	/**
	 * <p>
	 * Compares the expected values calculated in Java by testcase and which are
	 * in the normal filesystem, with those calculated by SystemML located in
	 * HDFS with Matrix Market format
	 * </p>
	 * 
	 * @param expectedFile
	 *            file with expected values, which is located in OS filesystem
	 * @param actualDir
	 *            file with actual values, which is located in HDFS
	 * @param epsilon
	 *            tolerance for value comparison
	 */
	public static void compareMMMatrixWithJavaMatrix(String expectedFile, String actualDir, double epsilon) {
		try {
			FileSystem fs = FileSystem.get(conf);
			Path outDirectory = new Path(actualDir);
			Path compareFile = new Path(expectedFile);
			FSDataInputStream fsin = fs.open(compareFile);
			BufferedReader compareIn = new BufferedReader(new InputStreamReader(fsin));
			
			HashMap<CellIndex, Double> expectedValues = new HashMap<CellIndex, Double>();
			
			// skip the header of Matrix Market file
			String line = compareIn.readLine();
			
			// rows, cols and nnz
			line = compareIn.readLine();
			String [] expRcn = line.split(" ");
			
			while ((line = compareIn.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(line, " ");
				int i = Integer.parseInt(st.nextToken());
				int j = Integer.parseInt(st.nextToken());
				double v = Double.parseDouble(st.nextToken());
				expectedValues.put(new CellIndex(i, j), v);
			}
			compareIn.close();

			HashMap<CellIndex, Double> actualValues = new HashMap<CellIndex, Double>();

			FSDataInputStream fsout = fs.open(outDirectory);
			BufferedReader outIn = new BufferedReader(new InputStreamReader(fsout));
			
			//skip MM header
			line = outIn.readLine();
			
			//rows, cols and nnz
			line = outIn.readLine();
			String[] rcn = line.split(" ");
			
			if (Integer.parseInt(expRcn[0]) != Integer.parseInt(rcn[0])) {
				System.out.println(" Rows mismatch: expected " + Integer.parseInt(expRcn[0]) + ", actual " + Integer.parseInt(rcn[0]));
			}
			else if (Integer.parseInt(expRcn[1]) != Integer.parseInt(rcn[1])) {
				System.out.println(" Cols mismatch: expected " + Integer.parseInt(expRcn[1]) + ", actual " + Integer.parseInt(rcn[1]));
			}
			else if (Integer.parseInt(expRcn[2]) != Integer.parseInt(rcn[2])) {
				System.out.println(" Nnz mismatch: expected " + Integer.parseInt(expRcn[2]) + ", actual " + Integer.parseInt(rcn[2]));
			}
			
			while ((line = outIn.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(line, " ");
				int i = Integer.parseInt(st.nextToken());
				int j = Integer.parseInt(st.nextToken());
				double v = Double.parseDouble(st.nextToken());
				actualValues.put(new CellIndex(i, j), v);
			}
		
			

			int countErrors = 0;
			for (CellIndex index : expectedValues.keySet()) {
				Double expectedValue = expectedValues.get(index);
				Double actualValue = actualValues.get(index);
				if (expectedValue == null)
					expectedValue = 0.0;
				if (actualValue == null)
					actualValue = 0.0;
				
			//	System.out.println("actual value: "+actualValue+", expected value: "+expectedValue);
				
				if (!compareCellValue(expectedValue, actualValue, epsilon, false)) {
					System.out.println(expectedFile+": "+index+" mismatch: expected " + expectedValue + ", actual " + actualValue);
					countErrors++;
				}
			}
			assertTrue("for file " + actualDir + " " + countErrors + " values are not equal", countErrors == 0);
		} catch (IOException e) {
			fail("unable to read file: " + e.getMessage());
		}
	}
	/**
	 * <p>
	 * Compares the expected values calculated in Java by testcase and which are
	 * in the normal filesystem, with those calculated by SystemML located in
	 * HDFS
	 * </p>
	 * 
	 * @param expectedFile
	 *            file with expected values, which is located in OS filesystem
	 * @param actualDir
	 *            file with actual values, which is located in HDFS
	 * @param epsilon
	 *            tolerance for value comparison
	 */
	public static void compareDMLMatrixWithJavaMatrix(String expectedFile, String actualDir, double epsilon) {
		try {
			FileSystem fs = FileSystem.get(conf);
			Path outDirectory = new Path(actualDir);
			Path compareFile = new Path(expectedFile);
			FSDataInputStream fsin = fs.open(compareFile);
			BufferedReader compareIn = new BufferedReader(new InputStreamReader(fsin));
			
			HashMap<CellIndex, Double> expectedValues = new HashMap<CellIndex, Double>();
			String line;
			while ((line = compareIn.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(line, " ");
				int i = Integer.parseInt(st.nextToken());
				int j = Integer.parseInt(st.nextToken());
				double v = Double.parseDouble(st.nextToken());
				expectedValues.put(new CellIndex(i, j), v);
			}
			compareIn.close();

			HashMap<CellIndex, Double> actualValues = new HashMap<CellIndex, Double>();

			FileStatus[] outFiles = fs.listStatus(outDirectory);

			for (FileStatus file : outFiles) {
				FSDataInputStream fsout = fs.open(file.getPath());
				BufferedReader outIn = new BufferedReader(new InputStreamReader(fsout));
				
				while ((line = outIn.readLine()) != null) {
					StringTokenizer st = new StringTokenizer(line, " ");
					int i = Integer.parseInt(st.nextToken());
					int j = Integer.parseInt(st.nextToken());
					double v = Double.parseDouble(st.nextToken());
					actualValues.put(new CellIndex(i, j), v);
				}
				outIn.close();
			}

			int countErrors = 0;
			for (CellIndex index : expectedValues.keySet()) {
				Double expectedValue = expectedValues.get(index);
				Double actualValue = actualValues.get(index);
				if (expectedValue == null)
					expectedValue = 0.0;
				if (actualValue == null)
					actualValue = 0.0;
				
			//	System.out.println("actual value: "+actualValue+", expected value: "+expectedValue);
				
				if (!compareCellValue(expectedValue, actualValue, epsilon, false)) {
					System.out.println(expectedFile+": "+index+" mismatch: expected " + expectedValue + ", actual " + actualValue);
					countErrors++;
				}
			}
			assertTrue("for file " + actualDir + " " + countErrors + " values are not equal", countErrors == 0);
		} catch (IOException e) {
			fail("unable to read file: " + e.getMessage());
		}
	}

	/**
	 * Reads values from a matrix file in HDFS in DML format
	 * 
	 * @deprecated You should not use this method, it is recommended to use the
	 *             corresponding method in AutomatedTestBase
	 * @param filePath
	 * @return
	 */
	public static HashMap<CellIndex, Double> readDMLMatrixFromHDFS(String filePath) 
	{
		HashMap<CellIndex, Double> expectedValues = new HashMap<CellIndex, Double>();
		
		try 
		{
			FileSystem fs = FileSystem.get(conf);
			Path outDirectory = new Path(filePath);
			String line;

			FileStatus[] outFiles = fs.listStatus(outDirectory);
			for (FileStatus file : outFiles) {
				FSDataInputStream outIn = fs.open(file.getPath());
				BufferedReader reader = new BufferedReader(new InputStreamReader(outIn));
				while ((line = reader.readLine()) != null) {
					StringTokenizer st = new StringTokenizer(line, " ");
					int i = Integer.parseInt(st.nextToken());
					int j = Integer.parseInt(st.nextToken());
					double v = Double.parseDouble(st.nextToken());
					expectedValues.put(new CellIndex(i,j), v);
				}
				outIn.close();
			}
		} 
		catch (IOException e) {
			assertTrue("could not read from file " + filePath, false);
		}

		return expectedValues;
	}

	/**
	 * Reads values from a matrix file in OS's FS in R format
	 * 
	 * @deprecated You should not use this method, it is recommended to use the
	 *             corresponding method in AutomatedTestBase
	 * 
	 * @param filePath
	 * @return
	 */
	
	// TODO: we must use http://www.inf.uni-konstanz.de/algo/lehre/ws05/pp/mtj/mvio/MatrixVectorReader.html
	// to read matrices from R
	
	public static HashMap<CellIndex, Double> readRMatrixFromFS(String filePath) 
	{
		HashMap<CellIndex, Double> expectedValues = new HashMap<CellIndex, Double>();
		BufferedReader reader = null;
		
		try 
		{
			reader = new BufferedReader(new FileReader(filePath));

			// skip both R header lines
			String line = reader.readLine();
			
			int matrixType = -1;
			if ( line.endsWith(" general") )
				matrixType = 1;
			if ( line.endsWith(" symmetric") )
				matrixType = 2;
			
			if ( matrixType == -1 )
				throw new RuntimeException("unknown matrix type while reading R matrix: ." + line);
			
			line = reader.readLine(); // header line with dimension and nnz information
			
			while ((line = reader.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(line, " ");
				int i = Integer.parseInt(st.nextToken());
				int j = Integer.parseInt(st.nextToken());
				if( st.hasMoreTokens() ) {
					double v = Double.parseDouble(st.nextToken());
					if( v==0.0 ) continue;
					expectedValues.put(new CellIndex(i, j), v);
					if ( matrixType == 2 )
						expectedValues.put(new CellIndex(j, i), v);
				}
				else
					expectedValues.put(new CellIndex(i, j), 1.0);
			}
		} 
		catch (IOException e) {
			assertTrue("could not read from file " + filePath, false);
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
		
		return expectedValues;
	}
	
	/**
	 * Reads a scalar value in DML format from HDFS
	 */
	public static HashMap<CellIndex, Double> readDMLScalarFromHDFS(String filePath) {
		HashMap<CellIndex, Double> expectedValues = new HashMap<CellIndex, Double>();
		expectedValues.put(new CellIndex(1,1), readDMLScalar(filePath));
		return expectedValues;
	}

	public static double readDMLScalar(String filePath) {
		FileSystem fs;
		try {
			double d=Double.NaN;
			fs = FileSystem.get(conf);
			Path outDirectory = new Path(filePath);
			String line;
			FileStatus[] outFiles = fs.listStatus(outDirectory);
			for (FileStatus file : outFiles) {
				FSDataInputStream fsout = fs.open(file.getPath());
				BufferedReader outIn = new BufferedReader(new InputStreamReader(fsout));
				
				while ((line = outIn.readLine()) != null) { // only 1 scalar value in file
					d = Double.parseDouble(line);
				}
				outIn.close();
			}
			return d;
		} catch (IOException e) {
			assertTrue("could not read from file " + filePath, false);
		}
		return Double.NaN;
	}

	public static boolean readDMLBoolean(String filePath) {
		FileSystem fs;
		try {
			Boolean b = null;
			fs = FileSystem.get(conf);
			Path outDirectory = new Path(filePath);
			String line;
			FileStatus[] outFiles = fs.listStatus(outDirectory);
			for (FileStatus file : outFiles) {
				FSDataInputStream fsout = fs.open(file.getPath());
				BufferedReader outIn = new BufferedReader(new InputStreamReader(fsout));
				
				while ((line = outIn.readLine()) != null) { // only 1 scalar value in file
					b = Boolean.valueOf(Boolean.parseBoolean(line));
				}
				outIn.close();
			}
			return b.booleanValue();
		} catch (IOException e) {
			assertTrue("could not read from file " + filePath, false);
		}
		return _AssertOccured;
	}
	
	public static String readDMLString(String filePath) {
		FileSystem fs;
		try {
			StringBuffer sb =  new StringBuffer();
			fs = FileSystem.get(conf);
			Path outDirectory = new Path(filePath);
			FileStatus[] outFiles = fs.listStatus(outDirectory);
			for (FileStatus file : outFiles) {
				FSDataInputStream fsout = fs.open(file.getPath());
				sb.append(IOUtils.toString(new InputStreamReader(fsout)));
			}
			return sb.toString();
		} catch (IOException e) {
			assertTrue("could not read from file " + filePath, false);
		}
		return null;
	}
		
	
	/**
	 * Reads a scalar value in R format from OS's FS
	 */
	public static HashMap<CellIndex, Double> readRScalarFromFS(String filePath) {
		HashMap<CellIndex, Double> expectedValues = new HashMap<CellIndex, Double>();
		expectedValues.put(new CellIndex(1,1), readRScalar(filePath));
		return expectedValues;
	}
	
	public static Double readRScalar(String filePath) {
		try {
			double d = Double.NaN;
			BufferedReader compareIn = new BufferedReader(new FileReader(filePath));
			String line;
			while ((line = compareIn.readLine()) != null) { // only 1 scalar value in file
				d = Double.parseDouble(line);
			}
			compareIn.close();
			return d;
		} catch (IOException e) {
			assertTrue("could not read from file " + filePath, false);
		}
		return Double.NaN;
	}
	
	public static String processMultiPartCSVForR(String csvFile) throws IOException {
		File csv = new File(csvFile);
		if (csv.isDirectory()) {
			File[] parts = csv.listFiles();
			
			int count=0;
			int index = -1;
			for(int i=0; i < parts.length; i++ ) {
				File f = parts[i];
				String path = f.getPath();
				if (path.startsWith(".") && path.endsWith(".crc"))
					continue;
				count++;
				index = i;
			}
			
			if ( count == 1) {
				csvFile = parts[index].toString();
			}
			else if ( count > 1 ) {
				File tmp = new File(csvFile+"_temp.csv");
				OutputStreamWriter out = null;

				try {
					out = new OutputStreamWriter(new FileOutputStream(tmp),
							"UTF-8");

					// Directory listing may contain .crc files or may be in the
					// wrong order. Sanitize the list of names.
					ArrayList<String> partNames = new ArrayList<String>();
					for (File part : parts) {
						String partName = part.getName();
						if (false == partName.endsWith(".crc")) {
							partNames.add(partName);
						}
					}
					Collections.sort(partNames);

					for (String name : partNames) {
						File part = new File(csv, name);
						// Assume that each file fits into memory.
						String fileContents = FileUtils.readFileToString(part,
								"UTF-8");
						out.append(fileContents);
					}
				} finally {
					if (null != out)
						out.close();
				}
				
				csvFile = tmp.getCanonicalPath();
			}
			else {
				throw new RuntimeException("Unexpected error while reading a CSV file in R: " + count);
			}
		}
		return csvFile;
	}

	/**
	 * Compares two double values regarding tolerance t. If one or both of them
	 * is null it is converted to 0.0.
	 * 
	 * @param v1
	 * @param v2
	 * @param t
	 *            Tolerance
	 * @return
	 */
	public static boolean compareCellValue(Double v1, Double v2, double t, boolean ignoreNaN) {
		if (v1 == null)
			v1 = 0.0;
		if (v2 == null)
			v2 = 0.0;
		if( ignoreNaN && (v1.isNaN() || v1.isInfinite() || v2.isNaN() || v2.isInfinite()) )
			return true;
		if (v1.equals(v2))
			return true;

		return Math.abs(v1 - v2) <= t;
	}

	/**
	 * <p>
	 * Compares two matrices in array format.
	 * </p>
	 * 
	 * @param expectedMatrix
	 *            expected values
	 * @param actualMatrix
	 *            actual values
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param epsilon
	 *            tolerance for value comparison
	 */
	public static void compareMatrices(double[][] expectedMatrix, double[][] actualMatrix, int rows, int cols,
			double epsilon) {
		int countErrors = 0;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (!compareCellValue(expectedMatrix[i][j], actualMatrix[i][j], epsilon, false))
					countErrors++;
			}
		}
		assertTrue("" + countErrors + " values are not in equal", countErrors == 0);
	}
	
	public static void compareScalars(double d1, double d2, double tol) {
		if(!compareCellValue(d1, d2, tol, false)) {
			assertTrue("Given scalars do not match: " + d1 + " != " + d2 , false);
		}
	}
	
	public static void compareScalars(String expected, String actual) {
			assertEquals(expected, actual);
	}

	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param tolerance
	 * @param name1
	 * @param name2
	 * @param ignoreNaN
	 * @return
	 */
	public static boolean compareMatrices(HashMap<CellIndex, Double> m1, HashMap<CellIndex, Double> m2,
			double tolerance, String name1, String name2) 
	{
		return compareMatrices(m1, m2, tolerance, name1, name2, false);
	}
	
	/**
	 * Compares two matrices given as HashMaps. The matrix containing more nnz
	 * is iterated and each cell value compared against the corresponding cell
	 * in the smaller matrix, to ensure that all values are compared.<br/>
	 * This method does not assert. Instead statistics are added to
	 * AssertionBuffer, at the end of the test you should call
	 * {@link TestUtils#displayAssertionBuffer()}.
	 * 
	 * @param m1
	 * @param m2
	 * @param tolerance
	 * @return True if matrices are identical regarding tolerance.
	 */
	public static boolean compareMatrices(HashMap<CellIndex, Double> m1, HashMap<CellIndex, Double> m2,
			double tolerance, String name1, String name2, boolean ignoreNaN) {
		HashMap<CellIndex, Double> first = m2;
		HashMap<CellIndex, Double> second = m1;
		String namefirst = name2;
		String namesecond = name1;
		boolean flag = true;
		
		/** to ensure that always the matrix with more nnz is iterated */
		if (m1.size() > m2.size()) {
			first = m1;
			second = m2;
			namefirst = name1;
			namesecond = name2;
			flag=false;
		}

		int countErrorWithinTolerance = 0;
		int countIdentical = 0;
		double minerr = -1;
		double maxerr = 0;

		for (CellIndex index : first.keySet()) {
			Double v1 = first.get(index);
			Double v2 = second.get(index);
			if (v1 == null)
				v1 = 0.0;
			if (v2 == null)
				v2 = 0.0;
			if (Math.abs(v1 - v2) < minerr || minerr == -1)
				minerr = Math.abs(v1 - v2);
			if (Math.abs(v1 - v2) > maxerr)
				maxerr = Math.abs(v1 - v2);

			if (!compareCellValue(first.get(index), second.get(index), 0, ignoreNaN)) {
				if (!compareCellValue(first.get(index), second.get(index), tolerance, ignoreNaN)) {
					countErrorWithinTolerance++;
					if(!flag)
						System.out.println(index+": "+first.get(index)+" <--> "+second.get(index));
					else 
						System.out.println(index+": "+second.get(index)+" <--> "+first.get(index));
				}
			} else {
				countIdentical++;
			}
		}

		String assertPrefix = (countErrorWithinTolerance == 0) ? "    " : "!  ";
		_AssertInfos.add(assertPrefix + name1 + "<->" + name2 + " # stored values in " + namefirst + ": "
				+ first.size());
		_AssertInfos.add(assertPrefix + name1 + "<->" + name2 + " # stored values in " + namesecond + ": "
				+ second.size());
		_AssertInfos.add(assertPrefix + name1 + "<->" + name2 + " identical values(z=0): " + countIdentical);
		_AssertInfos.add(assertPrefix + name1 + "<->" + name2 + " wrong values(z=" + tolerance + "): "
				+ countErrorWithinTolerance);
		_AssertInfos.add(assertPrefix + name1 + "<->" + name2 + " min error: " + minerr);
		_AssertInfos.add(assertPrefix + name1 + "<->" + name2 + " max error: " + maxerr);

		if (countErrorWithinTolerance == 0)
			return true;

		_AssertOccured = true;
		return false;
	}

	/**
	 * Converts a 2D array into a sparse hashmap matrix.
	 * 
	 * @param matrix
	 * @return
	 */
	public static HashMap<CellIndex, Double> convert2DDoubleArrayToHashMap(double[][] matrix) {
		HashMap<CellIndex, Double> hmMatrix = new HashMap<CellIndex, Double>();
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				if (matrix[i][j] != 0)
					hmMatrix.put(new CellIndex(i + 1, j + 1), matrix[i][j]);
			}
		}

		return hmMatrix;
	}
	
	/**
	 * Method to convert a hashmap of matrix entries into a double array
	 * @param matrix
	 * @return
	 */
	public static double[][] convertHashMapToDoubleArray(HashMap <CellIndex, Double> matrix)
	{
		int max_rows = -1, max_cols= -1;
		for(CellIndex ci :matrix.keySet())
		{
			if(ci.row > max_rows)
			{
				max_rows = ci.row;
			}
			if(ci.column > max_cols)
			{
				max_cols = ci.column;
			}
		}
		
		double [][] ret_arr = new double[max_rows][max_cols];
		
		for(CellIndex ci:matrix.keySet())
		{
			int i = ci.row-1;
			int j = ci.column-1;
			ret_arr[i][j] = matrix.get(ci);
		}
		
		return ret_arr;
		
	}
	
	/**
	 * 
	 * @param matrix
	 * @param rows
	 * @param cols
	 * @return
	 */
	public static double[][] convertHashMapToDoubleArray(HashMap <CellIndex, Double> matrix, int rows, int cols)
	{		
		double [][] ret_arr = new double[rows][cols];
		
		for(CellIndex ci:matrix.keySet())
		{
			int i = ci.row-1;
			int j = ci.column-1;
			ret_arr[i][j] = matrix.get(ci);
		}
		
		return ret_arr;
		
	}

	/**
	 * Converts a 2D double array into a 1D double array.
	 * 
	 * @param array
	 * @return
	 */
	public static double[] convert2Dto1DDoubleArray(double[][] array) {
		double[] ret = new double[array.length * array[0].length];
		int c = 0;
		for (int i = 0; i < array.length; i++) {
			for (int j = 0; j < array[0].length; j++) {
				ret[c++] = array[i][j];
			}
		}

		return ret;
	}

	/**
	 * Converts a 1D double array into a 2D double array.
	 * 
	 * @param array
	 * @return
	 */
	public static double[][] convert1Dto2DDoubleArray(double[] array, int rows) {
		int cols = array.length / rows;
		double[][] ret = new double[rows][cols];

		for (int c = 0; c < array.length; c++) {
			ret[c % cols][c / cols] = array[c];
		}

		return ret;
	}

	/**
	 * Asserts the content of assertion buffer, which may contain of all methods
	 * which assert not themselves but add information to that buffer.
	 */
	public static void displayAssertionBuffer() {
		String msg = "Detailed matrices characteristics:\n";
		for (String cur : _AssertInfos) {
			msg += cur + "\n";
		}

		assertTrue(msg, !_AssertOccured);
	}

	/**
	 * <p>
	 * Compares a dml matrix file in HDFS with a file in normal file system
	 * generated by R
	 * </p>
	 * 
	 * @param rFile
	 *            file with values calculated by R
	 * @param hdfsDir
	 *            file with actual values calculated by DML
	 * @param epsilon
	 *            tolerance for value comparison
	 */
	public static void compareDMLHDFSFileWithRFile(String rFile, String hdfsDir, double epsilon) {
		try {
			FileSystem fs = FileSystem.get(conf);
			Path outDirectory = new Path(hdfsDir);
			BufferedReader compareIn = new BufferedReader(new FileReader(rFile));
			HashMap<CellIndex, Double> expectedValues = new HashMap<CellIndex, Double>();
			HashMap<CellIndex, Double> actualValues = new HashMap<CellIndex, Double>();
			String line;
			/** skip both R header lines */
			compareIn.readLine();
			compareIn.readLine();
			while ((line = compareIn.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(line, " ");
				int i = Integer.parseInt(st.nextToken());
				int j = Integer.parseInt(st.nextToken());
				double v = Double.parseDouble(st.nextToken());
				expectedValues.put(new CellIndex(i, j), v);
			}
			compareIn.close();

			FileStatus[] outFiles = fs.listStatus(outDirectory);

			for (FileStatus file : outFiles) {
				FSDataInputStream fsout = fs.open(file.getPath());
				BufferedReader outIn = new BufferedReader(new InputStreamReader(fsout));
				
				while ((line = outIn.readLine()) != null) {
					StringTokenizer st = new StringTokenizer(line, " ");
					int i = Integer.parseInt(st.nextToken());
					int j = Integer.parseInt(st.nextToken());
					double v = Double.parseDouble(st.nextToken());
					actualValues.put(new CellIndex(i, j), v);
				}
				outIn.close();
			}

			int countErrors = 0;
			for (CellIndex index : expectedValues.keySet()) {
				Double expectedValue = expectedValues.get(index);
				Double actualValue = actualValues.get(index);
				if (expectedValue == null)
					expectedValue = 0.0;
				if (actualValue == null)
					actualValue = 0.0;

				if (!compareCellValue(expectedValue, actualValue, epsilon, false))
					countErrors++;
			}
			assertTrue("for file " + hdfsDir + " " + countErrors + " values are not in equal", countErrors == 0);
		} catch (IOException e) {
			fail("unable to read file: " + e.getMessage());
		}
	}

	/**
	 * <p>
	 * Checks a matrix against a number of specifications.
	 * </p>
	 * 
	 * @param matrix
	 *            matrix
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param min
	 *            minimum value
	 * @param max
	 *            maximum value
	 */
	public static void checkMatrix(BinaryMatrixCharacteristics matrix, long rows, long cols, double min, double max) {
		assertEquals(rows, matrix.getRows());
		assertEquals(cols, matrix.getCols());
		double[][] matrixValues = matrix.getValues();
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				assertTrue("invalid value",
						((matrixValues[i][j] >= min && matrixValues[i][j] <= max) || matrixValues[i][j] == 0));
			}
		}
	}

	/**
	 * <p>
	 * Checks a matrix read from a file in text format against a number of
	 * specifications.
	 * </p>
	 * 
	 * @param outDir
	 *            directory containing the matrix
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param min
	 *            minimum value
	 * @param max
	 *            maximum value
	 */
	public static void checkMatrix(String outDir, long rows, long cols, double min, double max) {
		try {
			FileSystem fs = FileSystem.get(conf);
			Path outDirectory = new Path(outDir);
			assertTrue(outDir + " does not exist", fs.exists(outDirectory));
			
			if( fs.getFileStatus(outDirectory).isDirectory() )
			{
				FileStatus[] outFiles = fs.listStatus(outDirectory);
				for (FileStatus file : outFiles) {
					FSDataInputStream fsout = fs.open(file.getPath());
					BufferedReader outIn = new BufferedReader(new InputStreamReader(fsout));
					
					String line;
					while ((line = outIn.readLine()) != null) {
						String[] rcv = line.split(" ");
						long row = Long.parseLong(rcv[0]);
						long col = Long.parseLong(rcv[1]);
						double value = Double.parseDouble(rcv[2]);
						assertTrue("invalid row index", (row > 0 && row <= rows));
						assertTrue("invlaid column index", (col > 0 && col <= cols));
						assertTrue("invalid value", ((value >= min && value <= max) || value == 0));
					}
					outIn.close();
				}
			}
			else
			{
				FSDataInputStream fsout = fs.open(outDirectory);
				BufferedReader outIn = new BufferedReader(new InputStreamReader(fsout));
				
				String line;
				while ((line = outIn.readLine()) != null) {
					String[] rcv = line.split(" ");
					long row = Long.parseLong(rcv[0]);
					long col = Long.parseLong(rcv[1]);
					double value = Double.parseDouble(rcv[2]);
					assertTrue("invalid row index", (row > 0 && row <= rows));
					assertTrue("invlaid column index", (col > 0 && col <= cols));
					assertTrue("invalid value", ((value >= min && value <= max) || value == 0));
				}
				outIn.close();
			}
		} catch (IOException e) {
			fail("unable to read file: " + e.getMessage());
		}
	}

	/**
	 * <p>
	 * Checks for matrix in directory existence.
	 * </p>
	 * 
	 * @param outDir
	 *            directory
	 */
	public static void checkForOutputExistence(String outDir) {
		try {
			FileSystem fs = FileSystem.get(conf);
			Path outDirectory = new Path(outDir);
			FileStatus[] outFiles = fs.listStatus(outDirectory);
			assertEquals("number of files in directory not 1", 1, outFiles.length);
			FSDataInputStream fsout = fs.open(outFiles[0].getPath());
			BufferedReader outIn = new BufferedReader(new InputStreamReader(fsout));
			
			String outLine = outIn.readLine();
			outIn.close();
			assertNotNull("file is empty", outLine);
			assertTrue("file is empty", outLine.length() > 0);
		} catch (IOException e) {
			fail("unable to read " + outDir + ": " + e.getMessage());
		}
	}

	/**
	 * <p>
	 * Removes all the directories specified in the array in HDFS
	 * </p>
	 * 
	 * @param directories
	 *            directories array
	 */
	public static void removeHDFSDirectories(String[] directories) {
		try {
			FileSystem fs = FileSystem.get(conf);
			for (String directory : directories) {
				Path dir = new Path(directory);
				if (fs.exists(dir) && fs.getFileStatus(dir).isDirectory()) {
					fs.delete(dir, true);
				}
			}
		} catch (IOException e) {
		}
	}

	/**
	 * <p>
	 * Removes all the directories specified in the array in OS filesystem
	 * </p>
	 * 
	 * @param directories
	 *            directories array
	 */
	public static void removeDirectories(String[] directories) {
		for (String directory : directories) {
			File dir = new File(directory);
			deleteDirectory(dir);
		}
	}

	private static boolean deleteDirectory(File path) {
		if (path.exists()) {
			File[] files = path.listFiles();
			for (int i = 0; i < files.length; i++) {
				if (files[i].isDirectory()) {
					deleteDirectory(files[i]);
				} else {
					files[i].delete();
				}
			}
		}
		return (path.delete());
	}

	/**
	 * <p>
	 * Removes all the files specified in the array in HDFS
	 * </p>
	 * 
	 * @param files
	 *            files array
	 */
	public static void removeHDFSFiles(String[] files) {
		try {
			FileSystem fs = FileSystem.get(conf);
			for (String directory : files) {
				Path dir = new Path(directory);
				if (fs.exists(dir) && !fs.getFileStatus(dir).isDirectory()) {
					fs.delete(dir, false);
				}
			}
		} catch (IOException e) {
		}
	}

	/**
	 * <p>
	 * Removes all the files specified in the array in OS filesystem
	 * </p>
	 * 
	 * @param files
	 *            files array
	 */
	public static void removeFiles(String[] files) {
		for (String directory : files) {
			File f = new File(directory);
			if (!f.exists() || !f.canWrite() || f.isDirectory())
				continue;

			f.delete();
		}
	}

	/**
	 * <p>
	 * Clears a complete directory.
	 * </p>
	 * 
	 * @param directory
	 *            directory
	 */
	public static void clearDirectory(String directory) {
		try {
			FileSystem fs = FileSystem.get(conf);
			FileStatus[] directoryContent = fs.listStatus(new Path(directory));
			for (FileStatus content : directoryContent) {
				fs.delete(content.getPath(), true);
			}
		} catch (IOException e) {
		}
	}

	/**
	 * <p>
	 * Generates a test matrix with the specified parameters as a two
	 * dimensional array.
	 * </p>
	 * <p>
	 * Set seed to -1 to use the current time as seed.
	 * </p>
	 * 
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param min
	 *            minimum value
	 * @param max
	 *            maximum value
	 * @param sparsity
	 *            sparsity
	 * @param seed
	 *            seed
	 * @return random matrix
	 */
	public static double[][] generateTestMatrix(int rows, int cols, double min, double max, double sparsity, long seed) {
		double[][] matrix = new double[rows][cols];
		Random random;
		if (seed == -1)
			random = TestUtils.random;
		else
			random = new Random(seed);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (random.nextDouble() > sparsity)
					continue;
				matrix[i][j] = (random.nextDouble() * (max - min) + min);
				// System.out.print(matrix[i][j] + "(" + i + "," + j + ")");
			}
			// System.out.println();
		}

		return matrix;
	}

	/**
	 * <p>
	 * Generates a test matrix with the specified parameters as a two
	 * dimensional array. The matrix will not contain any zero values.
	 * </p>
	 * <p>
	 * Set seed to -1 to use the current time as seed.
	 * </p>
	 * 
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param min
	 *            minimum value
	 * @param max
	 *            maximum value
	 * @param seed
	 *            seed
	 * @return random matrix
	 */
	public static double[][] generateNonZeroTestMatrix(int rows, int cols, double min, double max, long seed) {
		double[][] matrix = new double[rows][cols];
		Random random;
		if (seed == -1)
			random = TestUtils.random;
		else
			random = new Random(seed);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				double randValue;
				do {
					randValue = random.nextDouble();
				} while (randValue == 0);
				matrix[i][j] = (randValue * (max - min) + min);
			}
		}

		return matrix;
	}

	/**
	 * <p>
	 * Generates a test matrix with the specified parameters and writes it to a
	 * file using the text format.
	 * </p>
	 * <p>
	 * Set seed to -1 to use the current time as seed.
	 * </p>
	 * 
	 * @param file
	 *            output file
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param min
	 *            minimum value
	 * @param max
	 *            maximum value
	 * @param sparsity
	 *            sparsity
	 * @param seed
	 *            seed
	 */
	public static void generateTestMatrixToFile(String file, int rows, int cols, double min, double max,
			double sparsity, long seed) {
		try {
			FileSystem fs = FileSystem.get(conf);
			Path inFile = new Path(file);
			DataOutputStream out = fs.create(inFile);
			PrintWriter pw = new PrintWriter(out);
			Random random;
			if (seed == -1)
				random = TestUtils.random;
			else
				random = new Random(seed);

			for (int i = 1; i <= rows; i++) {
				for (int j = 1; j <= cols; j++) {
					if (random.nextDouble() > sparsity)
						continue;
					double value = (random.nextDouble() * (max - min) + min);
					if (value != 0)
						pw.println(i + " " + j + " " + value);
				}
			}
			pw.close();
			out.close();
		} catch (IOException e) {
			fail("unable to write test matrix: " + e.getMessage());
		}
	}

	/**
	 * Counts the number of NNZ values in a matrix
	 * 
	 * @param matrix
	 * @return
	 */
	public static int countNNZ(double[][] matrix) {
		int n = 0;
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				if (matrix[i][j] != 0)
					n++;
			}
		}
		return n;
	}

	public static void writeCSVTestMatrix(String file, double[][] matrix) 
	{
		try 
		{
			//create outputstream to HDFS / FS and writer
			DataOutputStream out = null;
			FileSystem fs = FileSystem.get(conf);
			out = fs.create(new Path(file), true);
			
			BufferedWriter pw = new BufferedWriter(new OutputStreamWriter(out));
			
			//writer actual matrix
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < matrix.length; i++) {
				sb.setLength(0);
				if ( matrix[i][0] != 0 )
					sb.append(matrix[i][0]);
				for (int j = 1; j < matrix[i].length; j++) {
					sb.append(",");
					if ( matrix[i][j] == 0 ) 
						continue;
					sb.append(matrix[i][j]);
				}
				sb.append('\n');
				pw.append(sb.toString());
			}
			
			//close writer and streams
			pw.close();
			out.close();
		} 
		catch (IOException e) 
		{
			fail("unable to write (csv) test matrix (" + file + "): " + e.getMessage());
		}
	}

	/**
	 * <p>
	 * Writes a matrix to a file using the text format.
	 * </p>
	 * 
	 * @param file
	 *            file name
	 * @param matrix
	 *            matrix
	 * @param isR
	 *            when true, writes a R matrix to disk
	 * 
	 */
	public static void writeTestMatrix(String file, double[][] matrix, boolean isR) 
	{
		try 
		{
			//create outputstream to HDFS / FS and writer
			DataOutputStream out = null;
			if (!isR) {
				FileSystem fs = FileSystem.get(conf);
				out = fs.create(new Path(file), true);
			} 
			else {
				out = new DataOutputStream(new FileOutputStream(file));
			}
			
			BufferedWriter pw = new BufferedWriter(new OutputStreamWriter(out));
			
			//write header
			if( isR ) {
				/** add R header */
				pw.append("%%MatrixMarket matrix coordinate real general\n");
				pw.append("" + matrix.length + " " + matrix[0].length + " " + matrix.length*matrix[0].length+"\n");
			}
			
			//writer actual matrix
			StringBuilder sb = new StringBuilder();
			boolean emptyOutput = true;
			for (int i = 0; i < matrix.length; i++) {
				for (int j = 0; j < matrix[i].length; j++) {
					if ( matrix[i][j] == 0 ) 
						continue;
					sb.append(i + 1);
					sb.append(' ');
					sb.append(j + 1);
					sb.append(' ');
					sb.append(matrix[i][j]);
					sb.append('\n');
					pw.append(sb.toString());
					sb.setLength(0);
					emptyOutput = false;
				}
			}
			
			//writer dummy entry if empty
			if( emptyOutput )
				pw.append("1 1 " + matrix[0][0]);
			
			//close writer and streams
			pw.close();
			out.close();
		} 
		catch (IOException e) 
		{
			fail("unable to write test matrix (" + file + "): " + e.getMessage());
		}
	}

	/**
	 * <p>
	 * Writes a matrix to a file using the text format.
	 * </p>
	 * 
	 * @param file
	 *            file name
	 * @param matrix
	 *            matrix
	 */
	public static void writeTestMatrix(String file, double[][] matrix) {
		writeTestMatrix(file, matrix, false);
	}

	/* Write a scalar value to a file */
	public static void writeTestScalar(String file, double value) {
		try {
			DataOutputStream out = new DataOutputStream(new FileOutputStream(file));
			PrintWriter pw = new PrintWriter(out);
			pw.println(value);
			pw.close();
			out.close();
		} catch (IOException e) {
			fail("unable to write test scalar (" + file + "): " + e.getMessage());
		}
	}
	
	/**
	 * <p>
	 * Writes a matrix to a file using the binary cells format.
	 * </p>
	 * 
	 * @param file
	 *            file name
	 * @param matrix
	 *            matrix
	 */
	@SuppressWarnings("deprecation")
	public static void writeBinaryTestMatrixCells(String file, double[][] matrix) {
		try {
			SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(conf), conf, new Path(file),
					MatrixIndexes.class, MatrixCell.class);

			MatrixIndexes index = new MatrixIndexes();
			MatrixCell value = new MatrixCell();
			for (int i = 0; i < matrix.length; i++) {
				for (int j = 0; j < matrix[i].length; j++) {
					if (matrix[i][j] != 0) {
						index.setIndexes((i + 1), (j + 1));
						value.setValue(matrix[i][j]);
						writer.append(index, value);
					}
				}
			}

			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to write test matrix: " + e.getMessage());
		}
	}

	/**
	 * <p>
	 * Writes a matrix to a file using the binary blocks format.
	 * </p>
	 * 
	 * @param file
	 *            file name
	 * @param matrix
	 *            matrix
	 * @param rowsInBlock
	 *            rows in block
	 * @param colsInBlock
	 *            columns in block
	 * @param sparseFormat
	 *            sparse format
	 */
	@SuppressWarnings("deprecation")
	public static void writeBinaryTestMatrixBlocks(String file, double[][] matrix, int rowsInBlock, int colsInBlock,
			boolean sparseFormat) {
		try {
			SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(conf), conf, new Path(file),
					MatrixIndexes.class, MatrixBlock.class);

			MatrixIndexes index = new MatrixIndexes();
			MatrixBlock value = new MatrixBlock();
			for (int i = 0; i < matrix.length; i += rowsInBlock) {
				int rows = Math.min(rowsInBlock, (matrix.length - i));
				for (int j = 0; j < matrix[i].length; j += colsInBlock) {
					int cols = Math.min(colsInBlock, (matrix[i].length - j));
					index.setIndexes(((i / rowsInBlock) + 1), ((j / colsInBlock) + 1));
					value = new MatrixBlock(rows, cols, sparseFormat);
					for (int k = 0; k < rows; k++) {
						for (int l = 0; l < cols; l++) {
							value.setValue(k, l, matrix[i + k][j + l]);
						}
					}
					writer.append(index, value);
				}
			}

			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to write test matrix: " + e.getMessage());
		}
	}

	/**
	 * <p>
	 * Prints out a DML script.
	 * </p>
	 * 
	 * @param dmlScriptfile
	 *            filename of DML script
	 */
	public static void printDMLScript(String dmlScriptFile) {
		try {
			System.out.println("Running script: " + dmlScriptFile + "\n");
			System.out.println("******************* DML script *******************");
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(dmlScriptFile)));
			String content;
			while ((content = in.readLine()) != null) {
				System.out.println(content);
			}
			in.close();
			System.out.println("**************************************************\n\n");
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to print dml script: " + e.getMessage());
		}
	}

	/**
	 * <p>
	 * Prints out a PYDML script.
	 * </p>
	 * 
	 * @param pydmlScriptfile
	 *            filename of PYDML script
	 */
	public static void printPYDMLScript(String pydmlScriptFile) {
		try {
			System.out.println("Running script: " + pydmlScriptFile + "\n");
			System.out.println("******************* PYDML script *******************");
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(pydmlScriptFile)));
			String content;
			while ((content = in.readLine()) != null) {
				System.out.println(content);
			}
			in.close();
			System.out.println("**************************************************\n\n");
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to print pydml script: " + e.getMessage());
		}
	}
	
	/**
	 * <p>
	 * Prints out an R script.
	 * </p>
	 * 
	 * @param dmlScriptfile
	 *            filename of RL script
	 */
	public static void printRScript(String dmlScriptFile) {
		try {
			System.out.println("Running script: " + dmlScriptFile + "\n");
			System.out.println("******************* R script *******************");
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(dmlScriptFile)));
			String content;
			while ((content = in.readLine()) != null) {
				System.out.println(content);
			}
			in.close();
			System.out.println("**************************************************\n\n");
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to print R script: " + e.getMessage());
		}
	}

	/**
	 * <p>
	 * Renames a temporary DML script file back to it's original name.
	 * </p>
	 * 
	 * @param dmlScriptFile
	 *            temporary script file
	 */
	public static void renameTempDMLScript(String dmlScriptFile) {
		// try {
		// FileSystem fs = FileSystem.get(conf);
		// Path oldPath = new Path(dmlScriptFile + "t");
		// Path newPath = new Path(dmlScriptFile);
		// if (fs.exists(oldPath))
		// fs.rename(oldPath, newPath);
		File oldPath = new File(dmlScriptFile + "t");
		File newPath = new File(dmlScriptFile);
		oldPath.renameTo(newPath);

		/*
		 * } catch (IOException e) { e.printStackTrace();
		 * fail("unable to write dml script back: " + e.getMessage()); }
		 */
	}

	/**
	 * <p>
	 * Removes all temporary files and directories in the current working
	 * directory.
	 * </p>
	 */
	public static void removeTemporaryFiles() {
		try {
			FileSystem fs = FileSystem.get(conf);
			Path workingDir = new Path(".");
			FileStatus[] files = fs.listStatus(workingDir);
			for (FileStatus file : files) {
				String fileName = file.getPath().toString().substring(
						file.getPath().getParent().toString().length() + 1);
				if (fileName.contains("temp"))
					fs.delete(file.getPath(), false);
			}
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to remove temporary files: " + e.getMessage());
		}
	}

	/**
	 * <p>
	 * Checks if any temporary files or directories exist in the current working
	 * directory.
	 * </p>
	 * 
	 * @return true if temporary files or directories are available
	 */
	public static boolean checkForTemporaryFiles() {
		try {
			FileSystem fs = FileSystem.get(conf);
			Path workingDir = new Path(".");
			FileStatus[] files = fs.listStatus(workingDir);
			for (FileStatus file : files) {
				String fileName = file.getPath().toString().substring(
						file.getPath().getParent().toString().length() + 1);
				if (fileName.contains("temp"))
					return true;
			}
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to remove temporary files: " + e.getMessage());
		}

		return false;
	}

	/**
	 * <p>
	 * Reads binary cells from a file. A matrix characteristic is created which
	 * contains the characteristics of the matrix read from the file and the
	 * values.
	 * </p>
	 * 
	 * @param directory
	 *            directory containing the matrix
	 * @return matrix characteristics
	 */
	@SuppressWarnings("deprecation")
	public static BinaryMatrixCharacteristics readCellsFromSequenceFile(String directory) {
		try {
			FileSystem fs = FileSystem.get(conf);
			FileStatus[] files = fs.listStatus(new Path(directory));

			HashMap<MatrixIndexes, Double> valueMap = new HashMap<MatrixIndexes, Double>();
			int rows = 0;
			int cols = 0;
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixCell value = new MatrixCell();
			for (FileStatus file : files) {
				SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(conf), file.getPath(), conf);

				while (reader.next(indexes, value)) {
					if (rows < indexes.getRowIndex())
						rows = (int) indexes.getRowIndex();
					if (cols < indexes.getColumnIndex())
						cols = (int) indexes.getColumnIndex();

					valueMap.put(new MatrixIndexes(indexes), value.getValue());
				}

				reader.close();
			}

			double[][] values = new double[rows][cols];
			long nonZeros = 0;
			for (MatrixIndexes index : valueMap.keySet()) {
				values[(int)index.getRowIndex() - 1][(int)index.getColumnIndex() - 1] = valueMap.get(index);
				if (valueMap.get(index) != 0)
					nonZeros++;
			}

			return new BinaryMatrixCharacteristics(values, rows, cols, 0, 0, 0, 0, nonZeros);
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to read sequence file in " + directory);
		}

		return null;
	}

	/**
	 * <p>
	 * Reads binary blocks from a file. A matrix characteristic is created which
	 * contains the characteristics of the matrix read from the file and the
	 * values.
	 * </p>
	 * 
	 * @param directory
	 *            directory containing the matrix
	 * @param rowsInBlock
	 *            rows in block
	 * @param colsInBlock
	 *            columns in block
	 * @return matrix characteristics
	 */
	@SuppressWarnings("deprecation")
	public static BinaryMatrixCharacteristics readBlocksFromSequenceFile(String directory, int rowsInBlock,
			int colsInBlock) {
		try {
			FileSystem fs = FileSystem.get(conf);
			FileStatus[] files = fs.listStatus(new Path(directory));

			HashMap<MatrixIndexes, Double> valueMap = new HashMap<MatrixIndexes, Double>();
			int rowsInLastBlock = -1;
			int colsInLastBlock = -1;
			int rows = 0;
			int cols = 0;
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixBlock value = new MatrixBlock();
			for (FileStatus file : files) {
				SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(conf), file.getPath(), conf);

				while (reader.next(indexes, value)) {
					if (value.getNumRows() < rowsInBlock) {
						if (rowsInLastBlock == -1)
							rowsInLastBlock = value.getNumRows();
						else if (rowsInLastBlock != value.getNumRows())
							fail("invalid block sizes");
						rows = (int) ((indexes.getRowIndex() - 1) * rowsInBlock + value.getNumRows());
					} else if (value.getNumRows() == rowsInBlock) {
						if (rows <= (indexes.getRowIndex() * rowsInBlock + value.getNumRows())) {
							if (rowsInLastBlock == -1)
								rows = (int) ((indexes.getRowIndex() - 1) * rowsInBlock + value.getNumRows());
							else
								fail("invalid block sizes");
						}
					} else {
						fail("invalid block sizes");
					}

					if (value.getNumColumns() < colsInBlock) {
						if (colsInLastBlock == -1)
							colsInLastBlock = value.getNumColumns();
						else if (colsInLastBlock != value.getNumColumns())
							fail("invalid block sizes");
						cols = (int) ((indexes.getColumnIndex() - 1) * colsInBlock + value.getNumColumns());
					} else if (value.getNumColumns() == colsInBlock) {
						if (cols <= (indexes.getColumnIndex() * colsInBlock + value.getNumColumns())) {
							if (colsInLastBlock == -1)
								cols = (int) ((indexes.getColumnIndex() - 1) * colsInBlock + value.getNumColumns());
							else
								fail("invalid block sizes");
						}
					} else {
						fail("invalid block sizes");
					}

					if (value.isInSparseFormat()) {
						Iterator<IJV> iter = value.getSparseBlockIterator();
						while( iter.hasNext() )
						{
							IJV cell = iter.next();
							valueMap.put(new MatrixIndexes(((indexes.getRowIndex() - 1) * rowsInBlock + cell.getI()),
									(int) ((indexes.getColumnIndex() - 1) * colsInBlock + cell.getJ())), cell.getV());
						}
						
					} else {
						double[] valuesInBlock = value.getDenseBlock();
						for (int i = 0; i < value.getNumRows(); i++) {
							for (int j = 0; j < value.getNumColumns(); j++) {
								valueMap.put(new MatrixIndexes(((indexes.getRowIndex() - 1) * rowsInBlock + i),
										(int) ((indexes.getColumnIndex() - 1) * colsInBlock + j)), valuesInBlock[i
										* value.getNumColumns() + j]);
							}
						}
					}
				}

				reader.close();
			}

			long nonZeros = 0;
			double[][] values = new double[rows][cols];
			for (MatrixIndexes index : valueMap.keySet()) {
				values[(int)index.getRowIndex()][(int)index.getColumnIndex()] = valueMap.get(index);
				if (valueMap.get(index) != 0)
					nonZeros++;
			}

			return new BinaryMatrixCharacteristics(values, rows, cols, rowsInBlock, rowsInLastBlock, colsInBlock,
					colsInLastBlock, nonZeros);
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to read sequence file in " + directory);
		}

		return null;
	}

	/**
	 * <p>
	 * Returns the path to a file in a directory if it is the only file in the
	 * directory.
	 * </p>
	 * 
	 * @param directory
	 *            directory containing the file
	 * @return path of the file
	 */
	public static Path getFileInDirectory(String directory) {
		try {
			FileSystem fs = FileSystem.get(conf);
			FileStatus[] files = fs.listStatus(new Path(directory));
			if (files.length != 1)
				throw new IOException("requires exactly one file in directory " + directory);

			return files[0].getPath();
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to open file in " + directory);
		}

		return null;
	}

	/**
	 * <p>
	 * Creates an empty file.
	 * </p>
	 * 
	 * @param file
	 *            filename
	 */
	public static void createFile(String filename) throws IOException {
			FileSystem fs = FileSystem.get(conf);
			fs.create(new Path(filename));
	}

	/**
	 * <p>
	 * Performs transpose onto a matrix and returns the result.
	 * </p>
	 * 
	 * @param a
	 *            matrix
	 * @return transposed matrix
	 */
	public static double[][] performTranspose(double[][] a) {
		int rows = a[0].length;
		int cols = a.length;
		double[][] result = new double[rows][cols];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = a[j][i];
			}
		}

		return result;
	}

	/**
	 * <p>
	 * Performs matrix multiplication onto two matrices and returns the result.
	 * </p>
	 * 
	 * @param a
	 *            left matrix
	 * @param b
	 *            right matrix
	 * @return computed result
	 */
	public static double[][] performMatrixMultiplication(double[][] a, double[][] b) {
		int rows = a.length;
		int cols = b[0].length;
		double[][] result = new double[rows][cols];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				double value = 0;
				for (int k = 0; k < a[i].length; k++) {
					value += (a[i][k] * b[k][j]);
				}
				result[i][j] = value;
			}
		}

		return result;
	}

	/**
	 * <p>
	 * Returns a random integer value.
	 * </p>
	 * 
	 * @return random integer value
	 */
	public static int getRandomInt() {
		Random random = new Random(System.currentTimeMillis());
		int randomValue = random.nextInt();
		return randomValue;
	}

	/**
	 * <p>
	 * Returns a positive random integer value.
	 * </p>
	 * 
	 * @return positive random integer value
	 */
	public static int getPositiveRandomInt() {
		int randomValue = TestUtils.getRandomInt();
		if (randomValue < 0)
			randomValue = -randomValue;
		return randomValue;
	}

	/**
	 * <p>
	 * Returns a negative random integer value.
	 * </p>
	 * 
	 * @return negative random integer value
	 */
	public static int getNegativeRandomInt() {
		int randomValue = TestUtils.getRandomInt();
		if (randomValue > 0)
			randomValue = -randomValue;
		return randomValue;
	}

	/**
	 * <p>
	 * Returns a random double value.
	 * </p>
	 * 
	 * @return random double value
	 */
	public static double getRandomDouble() {
		Random random = new Random(System.currentTimeMillis());
		double randomValue = random.nextInt() * random.nextDouble();
		return randomValue;
	}

	/**
	 * <p>
	 * Returns a positive random double value.
	 * </p>
	 * 
	 * @return positive random double value
	 */
	public static double getPositiveRandomDouble() {
		double randomValue = TestUtils.getRandomDouble();
		if (randomValue < 0)
			randomValue = -randomValue;
		return randomValue;
	}

	/**
	 * <p>
	 * Returns a negative random double value.
	 * </p>
	 * 
	 * @return negative random double value
	 */
	public static double getNegativeRandomDouble() {
		double randomValue = TestUtils.getRandomDouble();
		if (randomValue > 0)
			randomValue = -randomValue;
		return randomValue;
	}

	/**
	 * <p>
	 * Returns the string representation of a double value which can be used in
	 * a DML script.
	 * </p>
	 * 
	 * @param value
	 *            double value
	 * @return string representation
	 */
	public static String getStringRepresentationForDouble(double value) {
		NumberFormat nf = DecimalFormat.getInstance(new Locale("EN"));
		nf.setGroupingUsed(false);
		nf.setMinimumFractionDigits(1);
		nf.setMaximumFractionDigits(20);
		return nf.format(value);
	}

	/**
	 * Clears internal assertion information storage
	 */
	public static void clearAssertionInformation() {
		_AssertInfos.clear();
		_AssertOccured = false;
	}

	/**
	 * <p>
	 * Generates a matrix containing easy to debug values in its cells.
	 * </p>
	 * 
	 * @param rows
	 * @param cols
	 * @param bContainsZeros
	 *            If true, the matrix contains zeros. If false, the matrix
	 *            contains only positive values.
	 * @return
	 */
	public static double[][] createNonRandomMatrixValues(int rows, int cols, boolean bContainsZeros) {
		double[][] matrix = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (!bContainsZeros)
					matrix[i][j] = (i + 1) * 10 + (j + 1);
				else
					matrix[i][j] = (i) * 10 + (j);
			}
		}
		return matrix;
	}
	
	/**
	 * 
	 * @param data
	 * @return
	 */
	public static double[][] round(double[][] data) 
	{
		for(int i=0; i<data.length; i++)
			for(int j=0; j<data[i].length; j++)
				data[i][j]=Math.round(data[i][j]);
		
		return data;
	}
	
	/**
	 * 
	 * @param data
	 * @param rows
	 * @param cols
	 * @return
	 */
	public static double sum(double[][] data, int rows, int cols)
	{
		double sum = 0;
		for (int i = 0; i< rows; i++){
			for (int j = 0; j < cols; j++){
				sum += data[i][j];
			}
		}
		return sum;
	}
}

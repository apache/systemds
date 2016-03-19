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

package org.apache.sysml.test.integration.functions.data;

import static org.junit.Assert.fail;

import java.io.IOException;

import org.junit.Test;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;


/**
 * <p>
 * <b>Positive tests:</b>
 * </p>
 * <ul>
 * <li>text format</li>
 * <li>binary format</li>
 * </ul>
 * <p>
 * <b>Negative tests:</b>
 * </p>
 * <ul>
 * <li>wrong row dimension (format=text)</li>
 * <li>wrong column dimension (format=text)</li>
 * <li>wrong row and column dimensions (format=text)</li>
 * <li>wrong format (format=text)</li>
 * <li>wrong row dimension (format=binary)</li>
 * <li>wrong column dimension (format=binary)</li>
 * <li>wrong row and column dimensions (format=binary)</li>
 * <li>wrong format (format=binary)</li>
 * </ul>
 * 
 * 
 */
public class ReadMMTest extends AutomatedTestBase 
{
	
	private static final String TEST_DIR = "functions/data/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReadMMTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
	
		// positive tests
		addTestConfiguration("TextSimpleTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));
		addTestConfiguration("BinarySimpleTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));

		// negative tests
		addTestConfiguration("TextWrongRowDimensionTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));
		addTestConfiguration("TextWrongColDimensionTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));
		addTestConfiguration("TextWrongDimensionsTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));
		addTestConfiguration("TextWrongFormatTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));
		addTestConfiguration("BinaryWrongRowDimensionTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));
		addTestConfiguration("BinaryWrongColDimensionTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));
		addTestConfiguration("BinaryWrongDimensionsTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));
		addTestConfiguration("BinaryWrongFormatTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));
		addTestConfiguration("TextWrongIndexBaseTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMIndexTest", new String[] { "b" }));
		addTestConfiguration("EmptyTextTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));
		addTestConfiguration("EmptyBinaryTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadMMTest", new String[] { "a" }));
	}

	@Test
	public void testTextSimple() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("TextSimpleTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
	
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a);
		writeExpectedMatrix("a", a);

		runTest();

		compareResults();
	}

	@Test
	public void testTextWrongRowDimension() {
		int rows = 5;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("TextWrongRowDimensionTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
		
		loadTestConfiguration(config);

		createRandomMatrix("a", (rows + 5), cols, -1, 1, 1, -1);

		runTest(true, DMLException.class);
	}

	@Test
	public void testTextWrongColDimension() {
		int rows = 10;
		int cols = 5;

		TestConfiguration config = availableTestConfigurations.get("TextWrongColDimensionTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
		
		loadTestConfiguration(config);

		createRandomMatrix("a", rows, (cols + 5), -1, 1, 1, -1);

		runTest(true, DMLException.class);
	}

	/**
	 * Reads in given input matrix, writes it to disk and compares result to
	 * expected matrix. <br>
	 * The given input matrix has larger dimensions then specified in readMM as
	 * rows and cols parameter.
	 */
	@Test
	public void testTextWrongDimensions() {
		int rows = 3;
		int cols = 2;

		TestConfiguration config = availableTestConfigurations.get("TextWrongDimensionsTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
		
		double[][] a = new double[cols + 5][rows + 5];
		for (int j = 0; j < cols + 5; j++) {
			for (int i = 0; i < rows + 5; i++) {
				a[j][i] = (i + 1) * (j + 1);
			}
		}

		loadTestConfiguration(config);

		writeInputMatrix("a", a);

		runTest(true, DMLException.class);
	}

	/**
	 * Tries to read in wrong index-based matrix. Input matrix is zero-indexed
	 * instead of 1-indexed
	 */
	@Test
	public void testTextWrongIndexBase() {
		int rows = 1;
		int cols = 2;

		TestConfiguration config = availableTestConfigurations.get("TextWrongIndexBaseTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
		
		loadTestConfiguration(config);

		runTest(true, DMLException.class);
	}

	@Test
	public void testTextWrongFormat() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("TextWrongFormatTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");

		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 1, -1);
		writeInputBinaryMatrix("a", a, rows, cols, false);

		runTest(true, DMLException.class);
	}

	@Test
	public void testBinaryWrongRowDimension() throws IOException {
		int rows = 5;
		int cols = 10;
		int rowsInBlock = OptimizerUtils.DEFAULT_BLOCKSIZE;
		int colsInBlock = OptimizerUtils.DEFAULT_BLOCKSIZE;

		TestConfiguration config = availableTestConfigurations.get("BinaryWrongRowDimensionTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "binary");
		
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix((rows + 5), cols, -1, 1, 1, -1);
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, rowsInBlock, colsInBlock);
		writeInputBinaryMatrixWithMTD("a", a, rowsInBlock, colsInBlock, false, mc);
		runTest(true, DMLException.class);
	}

	@Test
	public void testBinaryWrongColDimension() throws IOException {
		int rows = 10;
		int cols = 5;
		int rowsInBlock = OptimizerUtils.DEFAULT_BLOCKSIZE;
		int colsInBlock = OptimizerUtils.DEFAULT_BLOCKSIZE;

		TestConfiguration config = availableTestConfigurations.get("BinaryWrongColDimensionTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "binary");
		
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, (cols + 5), -1, 1, 1, -1);
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, rowsInBlock, colsInBlock);
		writeInputBinaryMatrixWithMTD("a", a, rowsInBlock, colsInBlock, false, mc);

		runTest(true, DMLException.class);
	}

	/**
	 * Reads in given input matrix, writes it to disk and compares result to
	 * expected matrix. <br>
	 * The given input matrix has larger dimensions then specified in readMM as
	 * rows and cols parameter.
	 * @throws IOException 
	 */
	@Test
	public void testBinaryWrongDimensions() throws IOException {
		int rows = 3;
		int cols = 2;
		int rowsInBlock = OptimizerUtils.DEFAULT_BLOCKSIZE;
		int colsInBlock = OptimizerUtils.DEFAULT_BLOCKSIZE;

		TestConfiguration config = availableTestConfigurations.get("TextWrongDimensionsTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "binary");
		
		double[][] a = new double[cols + 5][rows + 5];
		for (int j = 0; j < cols + 5; j++) {
			for (int i = 0; i < rows + 5; i++) {
				a[j][i] = (i + 1) * (j + 1);
			}
		}

		loadTestConfiguration(config);
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, rowsInBlock, colsInBlock);
		writeInputBinaryMatrixWithMTD("a", a, rowsInBlock, colsInBlock, false, mc);

		runTest(true, DMLException.class);
	}

	@Test
	public void testBinaryWrongFormat() throws IOException {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("BinaryWrongFormatTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "binary");
		
		loadTestConfiguration(config);

		//createRandomMatrix("a", rows, cols, -1, 1, 1, -1);

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 1, -1);

		
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1);
		writeInputMatrixWithMTD("a", a, false, mc);
		//protected double[][] writeInputMatrixWithMTD(String name, double[][] matrix, boolean bIncludeR, MatrixCharacteristics mc) throws IOException {

		runTest(true, DMLException.class);
	}

	@Test
	public void testEmptyText() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("EmptyTextTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
		
		loadTestConfiguration(config);

		try {
			TestUtils.createFile(input("a/in"));
			runTest(true, DMLException.class);
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to create file " + input("a/in"));
		}

	}

	@Test
	public void testEmptyBinary() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("EmptyBinaryTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "binary");
		
		loadTestConfiguration(config);

		try {
			String fname = input("a");
			MapReduceTool.deleteFileIfExistOnHDFS(fname);
			MapReduceTool.deleteFileIfExistOnHDFS(fname + ".mtd");
			TestUtils.createFile(fname + "/in");
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, OptimizerUtils.DEFAULT_BLOCKSIZE, OptimizerUtils.DEFAULT_BLOCKSIZE);
			MapReduceTool.writeMetaDataFile(fname + ".mtd", ValueType.DOUBLE, mc, OutputInfo.stringToOutputInfo("binaryblock"));
			runTest(true, DMLException.class);
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to create file " + input("a/in"));
		}
	}

}

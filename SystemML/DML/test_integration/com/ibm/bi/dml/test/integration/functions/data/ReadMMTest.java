/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.data;

import static org.junit.Assert.fail;

import java.io.IOException;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/data/";

		// positive tests
		availableTestConfigurations.put("TextSimpleTest", new TestConfiguration("ReadMMTest", new String[] { "a" }));
		availableTestConfigurations.put("BinarySimpleTest", new TestConfiguration("ReadMMTest", new String[] { "a" }));

		// negative tests
		availableTestConfigurations.put("TextWrongRowDimensionTest", new TestConfiguration("ReadMMTest",
				new String[] { "a" }));
		availableTestConfigurations.put("TextWrongColDimensionTest", new TestConfiguration("ReadMMTest",
				new String[] { "a" }));
		availableTestConfigurations.put("TextWrongDimensionsTest", new TestConfiguration("ReadMMTest",
				new String[] { "a" }));
		availableTestConfigurations.put("TextWrongFormatTest",
				new TestConfiguration("ReadMMTest", new String[] { "a" }));
		availableTestConfigurations.put("BinaryWrongRowDimensionTest", new TestConfiguration("ReadMMTest",
				new String[] { "a" }));
		availableTestConfigurations.put("BinaryWrongColDimensionTest", new TestConfiguration("ReadMMTest",
				new String[] { "a" }));
		availableTestConfigurations.put("BinaryWrongDimensionsTest", new TestConfiguration("ReadMMTest",
				new String[] { "a" }));
		availableTestConfigurations.put("BinaryWrongFormatTest", new TestConfiguration("ReadMMTest",
				new String[] { "a" }));
		availableTestConfigurations.put("TextWrongIndexBaseTest", new TestConfiguration("ReadMMIndexTest",
				new String[] { "b" }));
		availableTestConfigurations.put("EmptyTextTest", new TestConfiguration("ReadMMTest", new String[] { "a" }));
		availableTestConfigurations.put("EmptyBinaryTest", new TestConfiguration("ReadMMTest", new String[] { "a" }));
	}

	@Test
	public void testTextSimple() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("TextSimpleTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
	
		loadTestConfiguration("TextSimpleTest");

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
		
		loadTestConfiguration("TextWrongRowDimensionTest");

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
		
		loadTestConfiguration("TextWrongColDimensionTest");

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

		loadTestConfiguration("TextWrongDimensionsTest");

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
		
		loadTestConfiguration("TextWrongIndexBaseTest");

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

		loadTestConfiguration("TextWrongFormatTest");

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 1, -1);
		writeInputBinaryMatrix("a", a, rows, cols, false);

		runTest(true, DMLException.class);
	}

	@Test
	public void testBinaryWrongRowDimension() throws IOException {
		int rows = 5;
		int cols = 10;
		int rowsInBlock = DMLTranslator.DMLBlockSize;
		int colsInBlock = DMLTranslator.DMLBlockSize;

		TestConfiguration config = availableTestConfigurations.get("BinaryWrongRowDimensionTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "binary");
		
		loadTestConfiguration("BinaryWrongRowDimensionTest");

		double[][] a = getRandomMatrix((rows + 5), cols, -1, 1, 1, -1);
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, rowsInBlock, colsInBlock);
		writeInputBinaryMatrixWithMTD("a", a, rowsInBlock, colsInBlock, false, mc);
		runTest(true, DMLException.class);
	}

	@Test
	public void testBinaryWrongColDimension() throws IOException {
		int rows = 10;
		int cols = 5;
		int rowsInBlock = DMLTranslator.DMLBlockSize;
		int colsInBlock = DMLTranslator.DMLBlockSize;

		TestConfiguration config = availableTestConfigurations.get("BinaryWrongColDimensionTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "binary");
		
		loadTestConfiguration("BinaryWrongColDimensionTest");

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
		int rowsInBlock = DMLTranslator.DMLBlockSize;
		int colsInBlock = DMLTranslator.DMLBlockSize;

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

		loadTestConfiguration("TextWrongDimensionsTest");
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
		
		loadTestConfiguration("BinaryWrongFormatTest");

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
		
		loadTestConfiguration("EmptyTextTest");

		try {
			TestUtils.createFile(baseDirectory + INPUT_DIR + "a/in");
			runTest(true, DMLException.class);
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to create file " + baseDirectory + INPUT_DIR + "a/in");
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
		
		loadTestConfiguration("EmptyBinaryTest");

		try {
			String fname = baseDirectory + INPUT_DIR + "a";
			MapReduceTool.deleteFileIfExistOnHDFS(fname);
			MapReduceTool.deleteFileIfExistOnHDFS(fname + ".mtd");
			TestUtils.createFile(fname + "/in");
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
			MapReduceTool.writeMetaDataFile(fname + ".mtd", ValueType.DOUBLE, mc, OutputInfo.stringToOutputInfo("binaryblock"));
			runTest(true, DMLException.class);
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to create file " + baseDirectory + INPUT_DIR + "a/in");
		}
	}

}

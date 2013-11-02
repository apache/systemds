/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.io.csv;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

@RunWith(value = Parameterized.class)
public class FormatChangeTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "csv_test";
	private final static String TEST_DIR = "functions/io/csv/";
	
	//private final static int rows = 1200;
	//private final static int cols = 100;
	//private final static double sparsity = 1;
	private static String format1, format2;
	private final static double eps = 1e-9;

	private int _rows, _cols;
	private double _sparsity;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   );  
	}
	
	public FormatChangeTest(int r, int c, double sp) {
		_rows = r; 
		_cols = c; 
		_sparsity = sp;
	}

	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { { 2000, 500, 0.01 }, { 1500, 150, 1 } };
	   return Arrays.asList(data);
	 }
	 
	private void setup() {
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", _rows);
		config.addVariable("cols", _cols);
		config.addVariable("format1", "text");
		config.addVariable("format2", "binary");
		
		loadTestConfiguration(config);
	}
	
	@Test
	public void testFormatChangeCP() {
		setup();
		RUNTIME_PLATFORM old_platform = rtplatform;
		rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
		formatChangeTest();
		rtplatform =  old_platform;
	}
	
	@Test
	public void testFormatChangeMR() {
		setup();
		RUNTIME_PLATFORM old_platform = rtplatform;
		rtplatform = RUNTIME_PLATFORM.HADOOP;
		formatChangeTest();
		rtplatform =  old_platform;
	}
	
	@Test
	public void testFormatChangeHybrid() {
		setup();
		RUNTIME_PLATFORM old_platform = rtplatform;
		rtplatform = RUNTIME_PLATFORM.HYBRID;
		formatChangeTest();
		rtplatform =  old_platform;
	}
	
	private void formatChangeTest() {

		int rows = _rows;
		int cols = _cols;
		double sparsity = _sparsity;

		//generate actual dataset
		double[][] D = getRandomMatrix(rows, cols, 0, 1, sparsity, 7777); 
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1);
		writeInputMatrixWithMTD("D", D, true, mc);

		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		String[] oldProgramArgs = programArgs = new String[]{"-args", 
											HOME + INPUT_DIR + "D",
                							format1,
                							HOME + INPUT_DIR + "D.binary",
                							format2
                							};
		
		String txtFile = HOME + INPUT_DIR + "D";
		String binFile = HOME + INPUT_DIR + "D.binary";
		String csvFile  = HOME + OUTPUT_DIR + "D.csv";
		
		// text to binary format
		programArgs[2] = "text";
		programArgs[3] = binFile;
		programArgs[4] = "binary";
		runTest(true, false, null, -1);

		// Test TextCell -> CSV conversion
		System.out.println("TextCell -> CSV");
		programArgs[2] = "text";
		programArgs[3] = csvFile;
		programArgs[4] = "csv";
		runTest(true, false, null, -1);
		
		compareFiles(rows, cols, sparsity, txtFile, "text", csvFile);

		// Test BinaryBlock -> CSV conversion
		System.out.println("BinaryBlock -> CSV");
		programArgs = oldProgramArgs;
		programArgs[1] = binFile;
		programArgs[2] = "binary";
		programArgs[3] = csvFile;
		programArgs[4] = "csv";
		runTest(true, false, null, -1);
		
		compareFiles(rows, cols, sparsity, binFile, "binary", csvFile);

		// Test CSV -> TextCell conversion
		System.out.println("CSV -> TextCell");
		programArgs = oldProgramArgs;
		programArgs[1] = csvFile;
		programArgs[2] = "csv";
		programArgs[3] = txtFile;
		programArgs[4] = "text";
		runTest(true, false, null, -1);
		
		compareFiles(rows, cols, sparsity, txtFile, "text", csvFile);

		// Test CSV -> BinaryBlock conversion
		System.out.println("CSV -> BinaryBlock");
		programArgs = oldProgramArgs;
		programArgs[1] = csvFile;
		programArgs[2] = "csv";
		programArgs[3] = binFile;
		programArgs[4] = "binary";
		runTest(true, false, null, -1);
		
		compareFiles(rows, cols, sparsity, binFile, "binary", csvFile);

		
		//fullRScriptName = HOME + TEST_NAME + ".R";
		//rCmd = "Rscript" + " " + fullRScriptName + " " + 
		//      HOME + INPUT_DIR + " " + Integer.toString((int)maxVal) + " " + HOME + EXPECTED_DIR;

	}
	
	private void compareFiles(int rows, int cols, double sparsity, String dmlFile, String dmlFormat, String mmFile) {
		String HOME = SCRIPT_DIR + TEST_DIR;
		
		// backup old DML and R script files
		String oldDMLScript = fullDMLScriptName;
		String oldRScript = fullRScriptName;
		
		String dmlOutput = HOME + OUTPUT_DIR + "dml.scalar";
		String rOutput = HOME + OUTPUT_DIR + "R.scalar";
		
		fullDMLScriptName = HOME + "csv_verify.dml";
		programArgs = new String[]{"-args", dmlFile,
                							Integer.toString(rows),
                							Integer.toString(cols),
                							dmlFormat,
                							dmlOutput
                							};
		
		fullRScriptName = HOME + "csv_verify.R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       mmFile + " " + rOutput;
		
		// Run the verify test
		runTest(true, false, null, -1);
		runRScript(true);
		
		double dmlScalar = TestUtils.readDMLScalar(dmlOutput); 
		double rScalar = TestUtils.readRScalar(rOutput); 
		
		TestUtils.compareScalars(dmlScalar, rScalar, eps);
		
		// restore old DML and R script files
		fullDMLScriptName = oldDMLScript;
		fullRScriptName = oldRScript;
		
	}
	
}
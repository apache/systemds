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
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

@RunWith(value = Parameterized.class)
public class CSVParametersTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "csvprop_test";
	private final static String TEST_DIR = "functions/io/csv/";
	
	private final static int rows = 1200;
	private final static int cols = 100;
	private final static double eps = 1e-9;
	
	private static double sparsity = 0.1;

	private boolean _header = false;
	private String _delim = ",";
	private boolean _sparse = true;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   );  
	}
	
	public CSVParametersTest(boolean header, String delim, boolean sparse) {
		_header = header;
		_delim = delim;
		_sparse = sparse;
	}

	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { 
			   //header  sep   sparse
			   { false,  ",",  true }, 
			   { false,  ",",  false }, 
			   { true,   ",",  true }, 
			   { true,   ",",  false },
			   { false,  "|.",  true }, 
			   { false,  "|.",  false }, 
			   { true,   "|.",  true }, 
			   { true,   "|.",  false } 
			  };
	   
	   return Arrays.asList(data);
	 }
	 
	private void setup() {
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("w_header", _header);
		config.addVariable("w_delim", _delim);
		config.addVariable("w_sparse", _sparse);
		
		loadTestConfiguration(config);
	}
	
	@Test
	public void testCSVParametersSparseCP() {
		setup();
		sparsity = 0.1;
		
		RUNTIME_PLATFORM old_platform = rtplatform;
		
		rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
		csvParameterTest(rtplatform, sparsity);
		
		rtplatform = old_platform;
	}
	
	@Test
	public void testCSVParametersDenseCP() {
		setup();
		sparsity = 1.0;
		
		RUNTIME_PLATFORM old_platform = rtplatform;

		rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
		csvParameterTest(rtplatform, sparsity);
		
		rtplatform = old_platform;
	}
	
	@Test
	public void testCSVParametersSparseMR() {
		setup();
		sparsity = 0.1;

		RUNTIME_PLATFORM old_platform = rtplatform;

		rtplatform = RUNTIME_PLATFORM.HADOOP;
		csvParameterTest(rtplatform, sparsity);
		
		rtplatform = old_platform;
	}
	
	@Test
	public void testCSVParametersDenseMR() {
		setup();
		sparsity = 1.0;

		RUNTIME_PLATFORM old_platform = rtplatform;

		rtplatform = RUNTIME_PLATFORM.HADOOP;
		csvParameterTest(rtplatform, sparsity);
		
		rtplatform = old_platform;
	}
	
	@Test
	public void testCSVParametersSparseHybrid() {
		setup();
		sparsity = 0.1;
		
		RUNTIME_PLATFORM old_platform = rtplatform;

		rtplatform = RUNTIME_PLATFORM.HYBRID;
		csvParameterTest(rtplatform, sparsity);
		
		rtplatform = old_platform;
	}
	
	@Test
	public void testCSVParametersDenseHybrid() {
		setup();
		sparsity = 1.0;
		
		RUNTIME_PLATFORM old_platform = rtplatform;

		rtplatform = RUNTIME_PLATFORM.HYBRID;
		csvParameterTest(rtplatform, sparsity);
		
		rtplatform = old_platform;
	}
	
	private void csvParameterTest(RUNTIME_PLATFORM platform, double sp) {
		
		//generate actual dataset
		double[][] D = getRandomMatrix(rows, cols, 0, 1, sp, 7777); 
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1);
		writeInputMatrixWithMTD("D", D, true, mc);
		D = null;

		String HOME = SCRIPT_DIR + TEST_DIR;
		String txtFile = HOME + INPUT_DIR + "D";
		//String binFile = HOME + INPUT_DIR + "D.binary";
		String csvFile  = HOME + OUTPUT_DIR + "D.csv";
		String scalarFile = HOME + OUTPUT_DIR + "diff.scalar";
		
		String writeDML = HOME + "csvprop_write.dml";
		String[] writeArgs = new String[]{"-args", 
				txtFile,
				csvFile,
				Boolean.toString(_header).toUpperCase(),
				_delim,
				Boolean.toString(_sparse).toUpperCase()
				};
		
		String readDML = HOME + "csvprop_read.dml";
		String[] readArgs = new String[]{"-args", 
				txtFile,
				csvFile,
				Boolean.toString(_header).toUpperCase(),
				_delim,
				Boolean.toString(_sparse).toUpperCase(),
				Double.toString(0.0),
				scalarFile
				};
		
		//System.out.println("Text -> CSV");
		// Text -> CSV 
		fullDMLScriptName = writeDML;
		programArgs = writeArgs;
		runTest(true, false, null, -1);

		// Evaluate the written CSV file 
		//System.out.println("CSV -> SCALAR");
		fullDMLScriptName = readDML;
		programArgs = readArgs;
		//boolean exceptionExpected = (!_sparse && sparsity < 1.0);
		runTest(true, false, null, -1);

		double dmlScalar = TestUtils.readDMLScalar(scalarFile); 
		TestUtils.compareScalars(dmlScalar, 0.0, eps);
	}
	
}
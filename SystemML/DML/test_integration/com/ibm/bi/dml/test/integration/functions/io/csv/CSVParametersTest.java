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
	private final static double sparsity = 1;
	private final static double eps = 1e-9;

	//private int _rows, _cols;
	//private double _sparsity;
	
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
	 
	@Test
	public void testFormatChange() {
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("w_header", _header);
		config.addVariable("w_delim", _delim);
		config.addVariable("w_sparse", _sparse);
		
		loadTestConfiguration(config);
		
		//generate actual dataset
		double[][] D = getRandomMatrix(rows, cols, 0, 1, sparsity, 7777); 
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1);
		writeInputMatrixWithMTD("D", D, true, mc);

		String HOME = SCRIPT_DIR + TEST_DIR;
		String txtFile = HOME + INPUT_DIR + "D";
		//String binFile = HOME + INPUT_DIR + "D.binary";
		String csvFile  = HOME + OUTPUT_DIR + "D.csv";
		String scalarFile = HOME + OUTPUT_DIR + "diff.scalar";
		
		String writeDML = HOME + "csvprop_write.dml";
		String[] writeArgs = new String[]{"-args", 
				txtFile,
				csvFile,
				Boolean.toString(_header),
				_delim,
				Boolean.toString(_sparse)
				};
		
		String readDML = HOME + "csvprop_read.dml";
		String[] readArgs = new String[]{"-args", 
				txtFile,
				csvFile,
				Boolean.toString(_header),
				_delim,
				Boolean.toString(_sparse),
				Double.toString(0.0),
				scalarFile
				};
		
		// Text -> CSV 
		fullDMLScriptName = writeDML;
		programArgs = writeArgs;
		runTest(true, false, null, -1);

		// Evaluate the written CSV file 
		fullDMLScriptName = readDML;
		programArgs = readArgs;
		runTest(true, false, null, -1);

		double dmlScalar = TestUtils.readDMLScalar(scalarFile); 
		
		// Add a test case that fails when fill=false and there are missing fields (sparsity must be changed to < 1)
		
		TestUtils.compareScalars(dmlScalar, 0.01, eps);

	}
	
}
/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.data;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * Tests if Rand produces the same output, for a given set of parameters, across different (CP vs. MR) runtime platforms.   
 * 
 */
@RunWith(value = Parameterized.class)
public class SequenceTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_NAME = "Sequence";
	
	private enum TEST_TYPE { THREE_INPUTS, TWO_INPUTS, ERROR };
	
	private final static double eps = 1e-10;
	
	private TEST_TYPE test_type;
	private double from, to, incr;
	
	public SequenceTest(TEST_TYPE tp, double f, double t, double i) {
		//numInputs = ni;
		test_type = tp;
		from = f;
		to = t;
		incr = i;
	}
	
	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] { 
				// 3 inputs
				{TEST_TYPE.THREE_INPUTS, 1, 2000, 1},
				{TEST_TYPE.THREE_INPUTS, 2000, 1, -1},
				{TEST_TYPE.THREE_INPUTS, 0, 150, 0.1},
				{TEST_TYPE.THREE_INPUTS, 150, 0, -0.1},
				{TEST_TYPE.THREE_INPUTS, 4000, 0, -3},
				
				// 2 inputs
				{TEST_TYPE.TWO_INPUTS, 1, 2000, Double.NaN},
				{TEST_TYPE.TWO_INPUTS, 2000, 1, Double.NaN},
				{TEST_TYPE.TWO_INPUTS, 0, 150, Double.NaN},
				{TEST_TYPE.TWO_INPUTS, 150, 0, Double.NaN}, 
				{TEST_TYPE.TWO_INPUTS, 4, 4, Double.NaN}, 
				
				// Error
				{TEST_TYPE.ERROR, 1, 2000, -1},
				{TEST_TYPE.ERROR, 150, 0, 1}
				};
		return Arrays.asList(data);
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_DIR, TEST_NAME,new String[]{"A"})); 
	}
	
	@Test
	public void testSequence() {
		RUNTIME_PLATFORM platformOld = rtplatform;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			boolean exceptionExpected = false;
			if ( test_type == TEST_TYPE.THREE_INPUTS || test_type == TEST_TYPE.ERROR ) {
				fullDMLScriptName = HOME + TEST_NAME + ".dml";
				fullRScriptName = HOME + TEST_NAME + ".R";
				programArgs = new String[]{"-args", Double.toString(from),
                        Double.toString(to),
                        Double.toString(incr),
                        HOME + OUTPUT_DIR + "A" };
				rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       from + " "+ to + " " + incr + " " + HOME + EXPECTED_DIR;
				
				if ( test_type == TEST_TYPE.ERROR ) 
					exceptionExpected = true;
			}
			else {
				fullDMLScriptName = HOME + TEST_NAME + "2inputs.dml";
				fullRScriptName = HOME + TEST_NAME + "2inputs.R";
				programArgs = new String[]{"-args", Double.toString(from),
                        Double.toString(to),
                        HOME + OUTPUT_DIR + "A" };
				rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       from + " "+ to + " " + HOME + EXPECTED_DIR;
			
			}
			loadTestConfiguration(config);
			int outputIndex = programArgs.length-1;
	
			rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
			programArgs[outputIndex] = HOME + OUTPUT_DIR + "A_CP";
			runTest(true, exceptionExpected, null, -1); 
			
			rtplatform = RUNTIME_PLATFORM.HADOOP;
			programArgs[outputIndex] = HOME + OUTPUT_DIR + "A_HADOOP";
			runTest(true, exceptionExpected, null, -1); 
			
			rtplatform = RUNTIME_PLATFORM.HYBRID;
			programArgs[outputIndex] = HOME + OUTPUT_DIR + "A_HYBRID";
			runTest(true, exceptionExpected, null, -1);
			
			rtplatform = RUNTIME_PLATFORM.SPARK;
			boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
			try {
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
				programArgs[outputIndex] = HOME + OUTPUT_DIR + "A_SPARK";
				runTest(true, exceptionExpected, null, -1);
			}
			finally {
				DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			}
			 
			
			if ( exceptionExpected == false ) {
				runRScript(true);
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("A_CP");
				HashMap<CellIndex, Double> rfile = readRMatrixFromFS("A");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "A-CP", "A-R");
				
				dmlfile = readDMLMatrixFromHDFS("A_HYBRID");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "A-HYBRID", "A-R");
			
				dmlfile = readDMLMatrixFromHDFS("A_HADOOP");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "A-HADOOP", "A-R");
				
				dmlfile = readDMLMatrixFromHDFS("A_SPARK");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "A-SPARK", "A-R");
			}
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
}

/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 *  This test checks for valid meta information after csv read with unknown size
 *  in singlenode exec mode (w/o reblock).
 *  
 */
public class NrowNcolUnknownCSVReadTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/misc/";

	private final static String TEST_NAME1 = "NrowUnknownCSVTest";
	private final static String TEST_NAME2 = "NcolUnknownCSVTest";
	private final static String TEST_NAME3 = "LengthUnknownCSVTest";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] {}));
	}
	
	@Test
	public void testNrowUnknownCSV() 
	{ 
		runNxxUnkownCSVTest( TEST_NAME1 ); 
	}
	
	@Test
	public void testNcolUnknownCSV() 
	{ 
		runNxxUnkownCSVTest( TEST_NAME2 ); 
	}
	
	@Test
	public void testLengthUnknownCSV() 
	{ 
		runNxxUnkownCSVTest( TEST_NAME3 ); 
	}
	
	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runNxxUnkownCSVTest( String testName ) 
	{
		String TEST_NAME = testName;
		RUNTIME_PLATFORM oldplatform = rtplatform;
		
		try
		{	
			rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
			
			//test configuration
			TestConfiguration config = getTestConfiguration(TEST_NAME);
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A"};
			
			loadTestConfiguration(config);
			
			//generate input data w/o mtd file
			int rows = 1034;
			int cols = 73;
			double[][] A = getRandomMatrix(rows, cols, 0, 1, 1.0, 7);
			MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
			DataConverter.writeMatrixToHDFS(mb, HOME + INPUT_DIR + "A", OutputInfo.CSVOutputInfo, 
					                        new MatrixCharacteristics(rows,cols,-1,-1));
	        
			//run tests
	        runTest(true, false, null, -1);
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
		finally
		{
			rtplatform = oldplatform;
		}
	}
}

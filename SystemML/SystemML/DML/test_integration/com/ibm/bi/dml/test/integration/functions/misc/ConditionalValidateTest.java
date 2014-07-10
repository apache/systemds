/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 *   
 */
public class ConditionalValidateTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/misc/";

	private final static String TEST_NAME1 = "conditionalValidate1"; //plain
	private final static String TEST_NAME2 = "conditionalValidate2"; //if
	private final static String TEST_NAME3 = "conditionalValidate3"; //for
	private final static String TEST_NAME4 = "conditionalValidate4"; //while
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {"R"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] {"R"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_DIR, TEST_NAME4, new String[] {"R"}));
	}
	
	@Test
	public void testUnconditionalReadNoError() 
	{ 
		runTest( TEST_NAME1, false, true ); 
	}
	
	@Test
	public void testUnconditionalReadError() 
	{ 
		runTest( TEST_NAME1, true, false ); 
	}
	
	@Test
	public void testIfConditionalReadNoErrorExists() 
	{ 
		runTest( TEST_NAME2, false, true ); 
	}
	
	@Test
	public void testIfConditionalReadNoErrorNotExists() 
	{ 
		runTest( TEST_NAME2, false, false ); 
	}
	
	@Test
	public void testForConditionalReadNoErrorExists() 
	{ 
		runTest( TEST_NAME3, false, true ); 
	}
	
	@Test
	public void testForConditionalReadNoErrorNotExists() 
	{ 
		runTest( TEST_NAME3, false, false ); 
	}
	
	@Test
	public void testWhileConditionalReadNoErrorExists() 
	{ 
		runTest( TEST_NAME4, false, true ); 
	}
	
	@Test
	public void testWhileConditionalReadNoErrorNotExists() 
	{ 
		runTest( TEST_NAME4, false, false ); 
	}
	
	
	
	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runTest( String testName, boolean exceptionExpected, boolean fileExists ) 
	{
		String TEST_NAME = testName;

		try
		{		
			TestConfiguration config = getTestConfiguration(TEST_NAME);

		    String HOME = SCRIPT_DIR + TEST_DIR;
		    String input = HOME + INPUT_DIR + "Y";
			
		    fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input };
			
			loadTestConfiguration(config);
			
			//write input
			double[][] Y = getRandomMatrix(10, 15, 0, 1, 1.0, 7);
			MatrixBlock mb = DataConverter.convertToMatrixBlock(Y);
			MatrixCharacteristics mc = new MatrixCharacteristics(10,15,1000,1000);
			
			DataConverter.writeMatrixToHDFS(mb, input+(fileExists?"":"b"), OutputInfo.TextCellOutputInfo, mc);
			MapReduceTool.writeMetaDataFile(input+(fileExists?"":"b")+".mtd", ValueType.DOUBLE, mc, OutputInfo.TextCellOutputInfo);
			
			//run tests
	        runTest(true, exceptionExpected, DMLException.class, -1);
	        
	        //cleanup
	        MapReduceTool.deleteFileIfExistOnHDFS(input);
	        MapReduceTool.deleteFileIfExistOnHDFS(input+"b");
	        MapReduceTool.deleteFileIfExistOnHDFS(input+".mtd");
	        MapReduceTool.deleteFileIfExistOnHDFS(input+"b.mtd");	        
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
	}
}

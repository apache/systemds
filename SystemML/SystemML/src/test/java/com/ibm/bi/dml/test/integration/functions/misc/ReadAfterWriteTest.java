/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import java.util.Random;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 *   
 */
public class ReadAfterWriteTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/misc/";

	private final static String TEST_NAME1 = "ReadAfterWriteMatrix1";
	private final static String TEST_NAME2 = "ReadAfterWriteMatrix2";
	private final static String TEST_NAME3 = "ReadAfterWriteScalar1";
	private final static String TEST_NAME4 = "ReadAfterWriteScalar2";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] {}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_DIR, TEST_NAME4, new String[] {}));
	}
	
	@Test
	public void testReadAfterWriteMatrixWithinDagPos() 
	{ 
		runReadAfterWriteTest( TEST_NAME1, true ); 
	}
	
	@Test
	public void testReadAfterWriteMatrixWithinDagNeg() 
	{ 
		runReadAfterWriteTest( TEST_NAME1, false ); 
	}
	
	@Test
	public void testReadAfterWriteMatrixAcrossDagPos() 
	{ 
		runReadAfterWriteTest( TEST_NAME2, true ); 
	}
	
	@Test
	public void testReadAfterWriteMatrixAcrossDagNeg() 
	{ 
		runReadAfterWriteTest( TEST_NAME2, false ); 
	}
	
	@Test
	public void testReadAfterWriteScalarWithinDagPos() 
	{ 
		runReadAfterWriteTest( TEST_NAME3, true ); 
	}
	
	@Test
	public void testReadAfterWriteScalarWithinDagNeg() 
	{ 
		runReadAfterWriteTest( TEST_NAME3, false ); 
	}
	
	@Test
	public void testReadAfterWriteScalarAcrossDagPos() 
	{ 
		runReadAfterWriteTest( TEST_NAME4, true ); 
	}
	
	@Test
	public void testReadAfterWriteScalarAcrossDagNeg() 
	{ 
		runReadAfterWriteTest( TEST_NAME4, false ); 
	}
	
	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runReadAfterWriteTest( String testName, boolean positive ) 
	{
		String TEST_NAME = testName;
		
		try
		{	
			//generate random file suffix
			int suffix = Math.abs(new Random().nextInt());
			String filename = SCRIPT_DIR + TEST_DIR + OUTPUT_DIR + suffix;
			String filename2 = positive ? filename : filename+"XXX";
			
			//test configuration
			TestConfiguration config = getTestConfiguration(TEST_NAME);
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", filename, filename2};
			loadTestConfiguration(config);
			
			//run tests
	        runTest(true, !positive, DMLException.class, -1);
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
		finally
		{
	        //cleanup
	        TestUtils.clearDirectory(SCRIPT_DIR + TEST_DIR + OUTPUT_DIR);
		}
	}
}

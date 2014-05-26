/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.io.binary;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class SerializeTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "SerializeTest";
	private final static String TEST_DIR = "functions/io/binary/";
	
	public static int rows1 = 746;
	public static int cols1 = 586;
	public static int cols2 = 4;
	
	private final static double eps = 1e-14;

	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] { "X" })   );  
	}
	
	@Test
	public void testEmptyBlock() 
	{ 
		runSerializeTest( rows1, cols1, 0.0 ); 
	}
	
	@Test
	public void testDenseBlock() 
	{ 
		runSerializeTest( rows1, cols1, 1.0 ); 
	}
	
	@Test
	public void testDenseSparseBlock() 
	{ 
		runSerializeTest( rows1, cols2, 0.3 ); 
	}
	
	@Test
	public void testDenseUltraSparseBlock() 
	{ 
		runSerializeTest( rows1, cols2, 0.1 ); 
	}
	
	@Test
	public void testSparseBlock() 
	{ 
		runSerializeTest( rows1, cols1, 0.1 ); 
	}
	
	@Test
	public void testSparseUltraSparseBlock() 
	{ 
		runSerializeTest( rows1, cols1, 0.0001 ); 
	}

	private void runSerializeTest( int rows, int cols, double sparsity ) 
	{
		try
		{	
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "X",
											    HOME + OUTPUT_DIR + "X"    };
			
			loadTestConfiguration(config);
	
			//generate actual dataset 
			double[][] X = getRandomMatrix(rows, cols, -1.0, 1.0, sparsity, 7); 
			MatrixBlock mb = DataConverter.convertToMatrixBlock(X);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, 1000, 1000);
			DataConverter.writeMatrixToHDFS(mb, HOME + INPUT_DIR + "X", OutputInfo.BinaryBlockOutputInfo, mc);
			MapReduceTool.writeMetaDataFile(HOME + INPUT_DIR + "X.mtd", ValueType.DOUBLE, mc, OutputInfo.BinaryBlockOutputInfo);
			
			runTest(true, false, null, -1); //mult 7
			
			//compare matrices 
			MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(HOME + OUTPUT_DIR + "X", InputInfo.BinaryBlockInputInfo, rows, cols, 1000, 1000);
			for( int i=0; i<mb.getNumRows(); i++ )
				for( int j=0; j<mb.getNumColumns(); j++ )
				{
					double val1 = mb.quickGetValue(i, j) * 7;
					double val2 = mb2.quickGetValue(i, j);
					Assert.assertEquals(val1, val2, eps);
				}
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
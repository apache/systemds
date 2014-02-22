/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

/**
 *   
 */
public class DataTypeCastingTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/misc/";

	private final static String TEST_NAME1 = "castMatrixScalar";
	private final static String TEST_NAME2 = "castScalarMatrix";
	
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {"R"}));
	}
	
	@Test
	public void testMatrixToScalar() 
	{ 
		runTest( TEST_NAME1, true, false ); 
	}
	
	@Test
	public void testMatrixToScalarWrongSize() 
	{ 
		runTest( TEST_NAME1, true, true ); 
	}
	
	@Test
	public void testScalarToScalar() 
	{ 
		runTest( TEST_NAME1, false, true ); 
	}
	
	@Test
	public void testScalarToMatrix() 
	{ 
		runTest( TEST_NAME2, false, false ); 
	}
	
	@Test
	public void testMatrixToMatrix() 
	{ 
		runTest( TEST_NAME2, true, true ); 
	}
	
	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runTest( String testName, boolean matrixInput, boolean exceptionExpected ) 
	{
		String TEST_NAME = testName;
		int numVals = (exceptionExpected ? 7 : 1);
		
		try
		{		
			TestConfiguration config = getTestConfiguration(TEST_NAME);
		    
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "V" , 
								                Integer.toString(numVals),
								                Integer.toString(numVals),
								                HOME + OUTPUT_DIR + "R", };
			
			loadTestConfiguration(config);
			
			//write input
			double[][] V = getRandomMatrix(numVals, numVals, 0, 1, 1.0, 7);
			if( matrixInput ){
				writeInputMatrix("V", V, false);	
			}
			else{
				MapReduceTool.writeDoubleToHDFS(V[0][0], HOME + INPUT_DIR + "V");
				MapReduceTool.writeScalarMetaDataFile(HOME + INPUT_DIR + "V.mtd", ValueType.DOUBLE);
			}
			
			
			//run tests
	        runTest(true, exceptionExpected, DMLException.class, -1);
	        
	        if( !exceptionExpected ){
		        //read output
		        double ret = -1;
		        if( testName.equals(TEST_NAME2) ){
		        	HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
		        	ret = dmlfile.get(new CellIndex(1,1));		
		        }
				else if( testName.equals(TEST_NAME1) ){
					HashMap<CellIndex, Double> dmlfile = readDMLScalarFromHDFS("R");
					ret = dmlfile.get(new CellIndex(1,1));
				}
		        
		        //compare results
		        Assert.assertEquals(V[0][0], ret, 1e-16);
	        }
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
	}
}

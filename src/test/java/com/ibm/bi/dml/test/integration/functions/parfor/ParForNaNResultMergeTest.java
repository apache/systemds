/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.parfor;

import java.util.HashMap;
import java.util.Map.Entry;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForNaNResultMergeTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_NAME1 = "parfor_NaN1";
	private final static String TEST_NAME2 = "parfor_NaN2";
	
	private final static double eps = 0;
	
	private final static int rows = 384;
	private final static int cols = rows;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "R" }) );
	}

	@Test
	public void testParForNaNOverwriteDense() 
	{
		runParForNaNResultMergeTest(TEST_NAME1, false);
	}
	
	@Test
	public void testParForNaNOverwriteSparse() 
	{
		runParForNaNResultMergeTest(TEST_NAME1, true);
	}
	
	@Test
	public void testParForNaNInsertDense() 
	{
		runParForNaNResultMergeTest(TEST_NAME2, false);
	}
	
	@Test
	public void testParForNaNInsertSparse() 
	{
		runParForNaNResultMergeTest(TEST_NAME2, true);
	}
	

	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForNaNResultMergeTest( String test, boolean sparse )
	{	
		//script
		String TEST_NAME = test;
		int xrow = sparse ? 1 : rows;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", 
									 String.valueOf(rows),
				                     String.valueOf(xrow),  
				                     HOME + OUTPUT_DIR + "R" };
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
				String.valueOf(rows) + " " + String.valueOf(xrow) + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		//run tests
		runTest(true, false, null, -1);
		runRScript(true);
	
		//compare matrices
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
		HashMap<CellIndex, Double> rfile  = replaceNaNValues(readRMatrixFromFS("R"));
		TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");	
	}
	
	/**
	 * Helper to replace all 1.0E308 with NaN because R is incapable of writing NaN
	 * through writeMM.
	 * 
	 * @param ret
	 */
	private HashMap<CellIndex, Double> replaceNaNValues(HashMap<CellIndex, Double> in) 
	{
		HashMap<CellIndex, Double> out = new HashMap<CellIndex, Double>();
		double NaN = 0d/0d;
		
		for( Entry<CellIndex,Double> e : in.entrySet() ) {
			if( e.getValue() == Math.pow(10, 308) )
				out.put(e.getKey(), NaN);
			else
				out.put(e.getKey(), e.getValue());
		}
		
		return out;
	}
}
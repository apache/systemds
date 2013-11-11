/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForAdversarialLiteralsTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1a = "parfor_literals1a"; //local parfor, out filename dynwrite contains _t0
	private final static String TEST_NAME1b = "parfor_literals1b"; //remote parfor, out filename dynwrite contains _t0
	private final static String TEST_NAME1c = "parfor_literals1c"; //local parfor nested, out filename dynwrite contains _t0
	private final static String TEST_NAME2 = "parfor_literals2"; //TODO clarify functions first
	private final static String TEST_NAME3 = "parfor_literals3"; //remote parfor, print delimiters for prog conversion.
	private final static String TEST_NAME4a = "parfor_literals4a"; //local parfor, varname _t0 
	private final static String TEST_NAME4b = "parfor_literals4b"; //remote parfor, varname _t0
	
	private final static String TEST_DIR = "functions/parfor/";
	private final static double eps = 1e-10;
	
	private final static int rows = 20;
	private final static int cols = 10;    
	private final static double sparsity = 1.0;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1a, 
				new TestConfiguration(TEST_DIR, TEST_NAME1a, 
				new String[] { "_t0B" })   );
		addTestConfiguration(
				TEST_NAME1b, 
				new TestConfiguration(TEST_DIR, TEST_NAME1b, 
				new String[] { "_t0B" })   );
		addTestConfiguration(
				TEST_NAME1c, 
				new TestConfiguration(TEST_DIR, TEST_NAME1c, 
				new String[] { "_t0B" })   );
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "B" })   );
		addTestConfiguration(
				TEST_NAME3, 
				new TestConfiguration(TEST_DIR, TEST_NAME3, 
				new String[] { "B" })   );
		addTestConfiguration(
				TEST_NAME4a, 
				new TestConfiguration(TEST_DIR, TEST_NAME4a, 
				new String[] { "B" })   );
		addTestConfiguration(
				TEST_NAME4b, 
				new TestConfiguration(TEST_DIR, TEST_NAME4b, 
				new String[] { "B" })   );
	}

	@Test
	public void testParForLocalThreadIDLiterals() 
	{
		runLiteralTest(TEST_NAME1a);
	}
	
	@Test
	public void testParForRemoteThreadIDLiterals() 
	{
		runLiteralTest(TEST_NAME1b);
	}
	
	@Test
	public void testParForLocalNestedThreadIDLiterals() 
	{
		runLiteralTest(TEST_NAME1c);
	}
	
	
	@Test
	public void testParForExtFuncLiterals()  //TODO
	{
		runLiteralTest(TEST_NAME2);
	}
	
	@Test
	public void testParForDelimiterLiterals() 
	{
		runLiteralTest(TEST_NAME3);
	}
	
	@Test
	public void testParForLocalThreadIDVarname() 
	{
		runLiteralTest(TEST_NAME4a);
	}
	
	@Test
	public void testParForRemoteThreadIDVarname() 
	{
		runLiteralTest(TEST_NAME4b);
	}

	
	private void runLiteralTest( String testName )
	{
		String TEST_NAME = testName;
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		// This is for running the junit test the new way, i.e., construct the arguments directly 
		String HOME = SCRIPT_DIR + TEST_DIR;
		String IN = "A";
		String OUT = (testName.equals(TEST_NAME1a)||testName.equals(TEST_NAME1b))?ProgramConverter.CP_ROOT_THREAD_ID:"B";

		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "A" , 
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        HOME + OUTPUT_DIR + OUT };
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);
        double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
		writeInputMatrix("A", A, false);

		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		
		//compare matrices
		HashMap<CellIndex, Double> dmlin = TestUtils.readDMLMatrixFromHDFS(HOME + INPUT_DIR + IN);
		HashMap<CellIndex, Double> dmlout = TestUtils.readDMLMatrixFromHDFS(HOME + OUTPUT_DIR + OUT);
		
		TestUtils.compareMatrices(dmlin, dmlout, eps, "DMLin", "DMLout");			
	}
	
}
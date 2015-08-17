/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.reorg;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * 
 * 
 */
public class FullTransposeTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "Transpose";

	
	private final static String TEST_DIR = "functions/reorg/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1751;
	private final static int cols1 = 1079;	
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	private enum OpType{
		MATRIX,
		ROW_VECTOR,
		COL_VECTOR,
	}
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_DIR, TEST_NAME1,new String[]{"Y"})); 
	}

	
	@Test
	public void testTransposeMatrixDenseCP() 
	{
		runTransposeTest(OpType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testTransposeRowVectorDenseCP() 
	{
		runTransposeTest(OpType.ROW_VECTOR, false, ExecType.CP);
	}
	
	@Test
	public void testTransposeColVectorDenseCP() 
	{
		runTransposeTest(OpType.COL_VECTOR, false, ExecType.CP);
	}
	
	@Test
	public void testTransposeMatrixSparseCP() 
	{
		runTransposeTest(OpType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testTransposeRowVectorSparseCP() 
	{
		runTransposeTest(OpType.ROW_VECTOR, true, ExecType.CP);
	}
	
	@Test
	public void testTransposeColVectorSparseCP() 
	{
		runTransposeTest(OpType.COL_VECTOR, true, ExecType.CP);
	}

	/// -----------------------
	@Test
	public void testTransposeMatrixDenseSP() 
	{
		runTransposeTest(OpType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testTransposeRowVectorDenseSP() 
	{
		runTransposeTest(OpType.ROW_VECTOR, false, ExecType.SPARK);
	}
	
	@Test
	public void testTransposeColVectorDenseSP() 
	{
		runTransposeTest(OpType.COL_VECTOR, false, ExecType.SPARK);
	}
	
	@Test
	public void testTransposeMatrixSparseSP() 
	{
		runTransposeTest(OpType.MATRIX, true, ExecType.SPARK);
	}
	
	@Test
	public void testTransposeRowVectorSparseSP() 
	{
		runTransposeTest(OpType.ROW_VECTOR, true, ExecType.SPARK);
	}
	
	@Test
	public void testTransposeColVectorSparseSP() 
	{
		runTransposeTest(OpType.COL_VECTOR, true, ExecType.SPARK);
	}
	
	/// -----------------------
	@Test
	public void testTransposeMatrixDenseMR() 
	{
		runTransposeTest(OpType.MATRIX, false, ExecType.MR);
	}
	
	@Test
	public void testTransposeRowVectorDenseMR() 
	{
		runTransposeTest(OpType.ROW_VECTOR, false, ExecType.MR);
	}
	
	@Test
	public void testTransposeColVectorDenseMR() 
	{
		runTransposeTest(OpType.COL_VECTOR, false, ExecType.MR);
	}
	
	@Test
	public void testTransposeMatrixSparseMR() 
	{
		runTransposeTest(OpType.MATRIX, true, ExecType.MR);
	}
	
	@Test
	public void testTransposeRowVectorSparseMR() 
	{
		runTransposeTest(OpType.ROW_VECTOR, true, ExecType.MR);
	}
	
	@Test
	public void testTransposeColVectorSparseMR() 
	{
		runTransposeTest(OpType.COL_VECTOR, true, ExecType.MR);
	}
	
	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runTransposeTest( OpType type, boolean sparse, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			String TEST_NAME = TEST_NAME1;
			
			int rows = -1;
			int cols = -1;
			switch( type )
			{
				case MATRIX: rows=rows1; cols=cols1; break;
				case ROW_VECTOR: rows=1; cols=cols1; break;
				case COL_VECTOR: rows=rows1; cols=1; break;
			}
			
			double sparsity = (sparse) ? sparsity1 : sparsity2;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "X",
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        HOME + OUTPUT_DIR + "Y"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7); 
			writeInputMatrix("X", A, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("Y");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Y");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
		
}
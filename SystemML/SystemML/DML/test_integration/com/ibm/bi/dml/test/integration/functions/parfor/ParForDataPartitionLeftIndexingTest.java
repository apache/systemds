/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForDataPartitionLeftIndexingTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "parfor_cdatapartition_leftindexing";
	private final static String TEST_NAME2 = "parfor_rdatapartition_leftindexing";
	private final static String TEST_DIR = "functions/parfor/";
	private final static double eps = 1e-10;
	
	private final static long dim1 = 10000;
	private final static long dim2 = 16;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1d;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "R" }) );
	}

	//colwise partitioning
	
	@Test
	public void testParForDataPartitionLeftindexingColDenseCP() 
	{
		runParForDataPartitionLeftindexingTest(true, false, ExecType.CP);
	}
	
	@Test
	public void testParForDataPartitionLeftindexingColSparseCP() 
	{
		runParForDataPartitionLeftindexingTest(true, true, ExecType.CP);
	}
	
	@Test
	public void testParForDataPartitionLeftindexingColDenseMR() 
	{
		runParForDataPartitionLeftindexingTest(true, false, ExecType.MR);
	}
	
	@Test
	public void testParForDataPartitionLeftindexingColSparseMR() 
	{
		runParForDataPartitionLeftindexingTest(true, true, ExecType.MR);
	}
	
	@Test
	public void testParForDataPartitionLeftindexingRowDenseCP() 
	{
		runParForDataPartitionLeftindexingTest(false, false, ExecType.CP);
	}
	
	@Test
	public void testParForDataPartitionLeftindexingRowSparseCP() 
	{
		runParForDataPartitionLeftindexingTest(false, true, ExecType.CP);
	}
	
	@Test
	public void testParForDataPartitionLeftindexingRowDenseMR() 
	{
		runParForDataPartitionLeftindexingTest(false, false, ExecType.MR);
	}
	
	@Test
	public void testParForDataPartitionLeftindexingRowSparseMR() 
	{
		runParForDataPartitionLeftindexingTest(false, true, ExecType.MR);
	}

	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForDataPartitionLeftindexingTest( boolean colwise, boolean sparse, ExecType et )
	{
		//rtplatform for MR
		//RUNTIME_PLATFORM platformOld = rtplatform;
		//rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
		
		//force MR leftindexing via very small memory budget
		long oldmem = InfrastructureAnalyzer.getLocalMaxMemory();
		if(et==ExecType.MR) {
			long mem = 1024*1024*1;
			InfrastructureAnalyzer.setLocalMaxMemory(mem);
		}
		
		try
		{
			//inst exec type, influenced via rows
			long rows = -1, cols = -1;
			if( colwise ){
				rows = dim1;
				cols = dim2;
			}
			else{
				rows = dim2;
				cols = dim1;
			}
				
			//script
			String TEST_NAME = colwise ? TEST_NAME1 : TEST_NAME2;
			double sparsity = sparse ? sparsity2 : sparsity1;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "V",
					                            HOME + OUTPUT_DIR + "R" };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate input data
			double[][] V = getRandomMatrix((int)rows, (int)cols, 0, 1, sparsity, 7);
			writeInputMatrixWithMTD("V", V, false);
	
			//run tests
			runTest(true, false, null, -1);
			
			//compare matrices
			HashMap<CellIndex, Double> input = TestUtils.convert2DDoubleArrayToHashMap(V);
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			TestUtils.compareMatrices(dmlfile, input, eps, "DML", "Input");	
		}
		finally
		{
			//rtplatform = platformOld;
			InfrastructureAnalyzer.setLocalMaxMemory(oldmem);
		}
	}
}
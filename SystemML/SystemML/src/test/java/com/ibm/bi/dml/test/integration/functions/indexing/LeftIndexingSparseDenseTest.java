/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.indexing;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class LeftIndexingSparseDenseTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/indexing/";
	private final static String TEST_NAME = "LeftIndexingSparseDenseTest";
	
	private final static int rows1 = 1073;
	private final static int cols1 = 1050;
	private final static int rows2 = 1073;
	private final static int cols2 = 550;
	
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	private enum LixType {
		LEFT_ALIGNED,
		LEFT2_ALIGNED,
		RIGHT_ALIGNED,
		RIGHT2_ALIGNED,
		CENTERED,
	}
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] {"R"}));
	}
	
	@Test
	public void testSparseLeftIndexingLeftAligned() {
		runLeftIndexingSparseSparseTest(LixType.LEFT_ALIGNED);
	}
	
	@Test
	public void testSparseLeftIndexingLeft2Aligned() {
		runLeftIndexingSparseSparseTest(LixType.LEFT2_ALIGNED);
	}
	
	@Test
	public void testSparseLeftIndexingRightAligned() {
		runLeftIndexingSparseSparseTest(LixType.RIGHT_ALIGNED);
	}
	
	@Test
	public void testSparseLeftIndexingRight2Aligned() {
		runLeftIndexingSparseSparseTest(LixType.RIGHT2_ALIGNED);
	}
	
	@Test
	public void testSparseLeftIndexingCentered() {
		runLeftIndexingSparseSparseTest(LixType.CENTERED);
	}
	
	/**
	 * 
	 * @param type
	 */
	public void runLeftIndexingSparseSparseTest(LixType type) 
	{
		//setup range (column lower/upper)
		int cl = -1;
		switch( type ){
			case LEFT_ALIGNED: cl = 1; break;
			case LEFT2_ALIGNED: cl = 2; break;
			case RIGHT_ALIGNED: cl = cols1-cols2+1; break;
			case RIGHT2_ALIGNED: cl = cols1-cols2; break;
			case CENTERED: cl = (cols1-cols2)/2; break;
		}
		int cu = cl+cols2-1;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);    
	    
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args",  
							HOME + INPUT_DIR + "A" , 
							HOME + INPUT_DIR + "B" , 
	               			String.valueOf(cl),
	               			String.valueOf(cu),
	                        HOME + OUTPUT_DIR + "R"
	                        };
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + cl + " " + cu + " " + HOME + EXPECTED_DIR;

		loadTestConfiguration(config);
		
		//generate input data sets
		double[][] A = getRandomMatrix(rows1, cols1, -1, 1, sparsity1, 1234);
        writeInputMatrixWithMTD("A", A, true);
		double[][] B = getRandomMatrix(rows2, cols2, -1, 1, sparsity2, 5678);
        writeInputMatrixWithMTD("B", B, true);
        
        runTest(true, false, null, 1); //REBLOCK
		runRScript(true);
		
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
		HashMap<CellIndex, Double> rfile = readRMatrixFromFS("R");
		TestUtils.compareMatrices(dmlfile, rfile, 0, "DML", "R");
	}
}


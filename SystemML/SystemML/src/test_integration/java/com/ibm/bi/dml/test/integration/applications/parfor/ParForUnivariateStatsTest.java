/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForUnivariateStatsTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "parfor_univariate";
	private final static String TEST_DIR = "applications/parfor/";
	private final static double eps = 1e-10; 
	
	//for test of sort_mr set optimizerutils mem to 0.00001 and decomment the following
	//private final static int rows2 = 10000;//(int) (Hops.CPThreshold+1);  // # of rows in each vector (for MR instructions)
	
	private final static int rows2 = (int) (Hop.CPThreshold+1);  // # of rows in each vector (for MR instructions)
	private final static int cols = 30;      // # of columns in each vector  
	
	private final static double minVal=1;    // minimum value in each vector 
	private final static double maxVal=5; // maximum value in each vector 

	/**
	 * Main method for running one test at a time.
	 */
	public static void main(String[] args) {
		long startMsec = System.currentTimeMillis();

		ParForUnivariateStatsTest t = new ParForUnivariateStatsTest();
		t.setUpBase();
		t.setUp();
		t.testParForUnivariateStatsDefaultMR();
		t.tearDown();

		long elapsedMsec = System.currentTimeMillis() - startMsec;
		System.err.printf("Finished in %1.3f sec\n", elapsedMsec / 1000.0);

	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   );  
	}
	
	@Test
	public void testForUnivariateStatsSerialSerialMR() 
	{
		runParForUnivariateStatsTest(false, PExecMode.LOCAL, PExecMode.LOCAL, ExecType.MR);
	}

	@Test
	public void testParForUnivariateStatsLocalLocalMR() 
	{
		runParForUnivariateStatsTest(true, PExecMode.LOCAL, PExecMode.LOCAL, ExecType.MR);
	}
	

	@Test
	public void testParForUnivariateStatsDefaultMR() 
	{
		runParForUnivariateStatsTest(true, null, null, ExecType.MR);
	}
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForUnivariateStatsTest( boolean parallel, PExecMode outer, PExecMode inner, ExecType instType )
	{
		//inst exec type, influenced via rows
		int rows = -1;
		if( instType == ExecType.CP )
			rows = rows2;
		else //if type MR
			rows = rows2;
		
		//script
		int scriptNum = -1;
		if( parallel )
		{
			if( inner == PExecMode.REMOTE_MR )      scriptNum=2;
			else if( outer == PExecMode.REMOTE_MR ) scriptNum=3;
			else if( outer == PExecMode.LOCAL ) 	scriptNum=1;
			else                                    scriptNum=4; //optimized
		}
		else
		{
			scriptNum = 0;
		}
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME +scriptNum + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "D" ,
				                        HOME + INPUT_DIR + "K",
				                        Integer.toString((int)maxVal),
				                        HOME + OUTPUT_DIR + "univarstats" };
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + Integer.toString((int)maxVal) + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		//generate actual dataset
		double[][] D = getRandomMatrix(rows, cols, minVal, maxVal, 1, 7777); 
		double[] Dkind = new double[cols]; 
		for( int i=0; i<cols; i++ )
		{
			Dkind[i]=(i%3)+1;//kind 1,2,3
			if( Dkind[i]!=1 )
				round(D,i); //for ordinal and categorical vars
		}
		MatrixCharacteristics mc1 = new MatrixCharacteristics(rows,cols,-1,-1);
		writeInputMatrixWithMTD("D", D, true, mc1);

		//generate kind for attributes (1,2,3)
        double[][] K = new double[1][cols];
        for( int i=0; i<cols; i++ )
        {
        	K[0][i] = Dkind[i];
        }
        MatrixCharacteristics mc2 = new MatrixCharacteristics(1,cols,-1,-1);
		writeInputMatrixWithMTD("K", K, true,mc2);		

		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);

		runRScript(true); 
		
		//compare matrices 
		for( String out : new String[]{"base.stats", "categorical.counts" } )
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("univarstats/"+out);
			
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS(out);
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
	}
	
	private void round(double[][] data, int col) {
		for(int i=0; i<data.length; i++)
			data[i][col]=Math.floor(data[i][col]);
	}
}
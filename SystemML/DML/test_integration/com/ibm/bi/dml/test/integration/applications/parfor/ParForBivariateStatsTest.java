package com.ibm.bi.dml.test.integration.applications.parfor;

import java.util.HashMap;

//import org.junit.Test;

import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForBivariateStatsTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "parfor_bivariate";
	private final static String TEST_DIR = "applications/parfor/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1000;  // # of rows in each vector (for CP instructions) 
	private final static int rows2 = 1000;  // # of rows in each vector (for MR instructions)
	private final static int cols = 30;      // # of columns in each vector  
	private final static int cols2 = 7;      // # of columns in each vector   
	
	private final static double minVal=1;    // minimum value in each vector 
	private final static double maxVal=5; // maximum value in each vector 

	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   );  //TODO MB: is this ever used???
	}

	/* TODO even serial test produces incorrect results, for specific datasets
	 * (result mismatch occurs on COV, on second scale-scale of same dataset and
	 *  AFTER a scale-categorical was computed (grouped aggregate error?))
	@Test
	public void testForBivariateStatsSerialSerialMR() 
	{
		runParForBivariateStatsTest(false, PExecMode.LOCAL, PExecMode.LOCAL, ExecType.MR);
	}*/

	
	/*
	@Test 
	public void testParForBivariateStatsLocalLocalCP() 
	{
		runParForBivariateStatsTest(PExecMode.LOCAL, PExecMode.LOCAL, ExecType.CP);
	}
	*/

	/*@Test
	public void testParForBivariateStatsLocalLocalMR() 
	{
		runParForBivariateStatsTest(true, PExecMode.LOCAL, PExecMode.LOCAL, ExecType.MR);
	}*/

	/*
	@Test
	public void testParForBivariateStatsLocalRemoteCP() 
	{
		runParForBivariateStatsTest(PExecMode.LOCAL, PExecMode.REMOTE_MR, ExecType.CP);
	}
	
	@Test
	public void testParForBivariateStatsRemoteLocalCP() 
	{
		runParForBivariateStatsTest(PExecMode.REMOTE_MR, PExecMode.LOCAL, ExecType.CP);
	}*/
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForBivariateStatsTest( boolean parallel, PExecMode outer, PExecMode inner, ExecType instType )
	{
		//inst exec type, influenced via rows
		int rows = -1;
		if( instType == ExecType.CP )
			rows = rows1;
		else //if type MR
			rows = rows2;
		
		//script
		int scriptNum = -1;
		if( parallel )
		{
			if( inner == PExecMode.REMOTE_MR )      scriptNum=2;
			else if( outer == PExecMode.REMOTE_MR ) scriptNum=3;
			else 									scriptNum=1;
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
		dmlArgs = new String[]{"-f", HOME + TEST_NAME +scriptNum + ".dml", "-exec", "singlenode",
				               "-args", HOME + INPUT_DIR + "D" ,
				                        HOME + INPUT_DIR + "S1" ,
				                        HOME + INPUT_DIR + "S2" ,
				                        HOME + INPUT_DIR + "K1" ,
				                        HOME + INPUT_DIR + "K2" ,
				                        HOME + OUTPUT_DIR + "bivarstats",
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        Integer.toString(cols2),
				                        Integer.toString(cols2*cols2),
				                        Integer.toString((int)maxVal)
				                         };
		dmlArgsDebug = new String[]{"-f", HOME + TEST_NAME + scriptNum + ".dml", "-d",
					               "-args", HOME + INPUT_DIR + "D" ,
					                        HOME + INPUT_DIR + "S1" ,
					                        HOME + INPUT_DIR + "S2" ,
					                        HOME + INPUT_DIR + "K1" ,
					                        HOME + INPUT_DIR + "K2" ,
					                        HOME + OUTPUT_DIR + "bivarstats",
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        Integer.toString(cols2),
					                        Integer.toString(cols2*cols2),
					                        Integer.toString((int)maxVal) };
				
		rCmd = "Rscript" + " " + HOME + TEST_NAME + ".R" + " " + 
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
		writeInputMatrix("D", D, true);
		
		//generate attribute sets		
        double[][] S1 = getRandomMatrix(1, cols2, 1, cols+1-eps, 1, 1112);
        double[][] S2 = getRandomMatrix(1, cols2, 1, cols+1-eps, 1, 1113);
        round(S1);
        round(S2);
		writeInputMatrix("S1", S1, true);
		writeInputMatrix("S2", S2, true);	

		//generate kind for attributes (1,2,3)
        double[][] K1 = new double[1][cols2];
        double[][] K2 = new double[1][cols2];
        for( int i=0; i<cols2; i++ )
        {
        	K1[0][i] = Dkind[(int)S1[0][i]-1];
        	K2[0][i] = Dkind[(int)S2[0][i]-1];
        }
        writeInputMatrix("K1", K1, true);
		writeInputMatrix("K2", K2, true);			

		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);

		runRScript(true); 
		
		//compare matrices 
		for( String out : new String[]{"bivar.stats", "category.counts", "category.means",  "category.variances" } )
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("bivarstats/"+out);
			
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS(out);
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
	}
	
	private void round(double[][] data) {
		for(int i=0; i<data.length; i++)
			for(int j=0; j<data[i].length; j++)
				data[i][j]=Math.floor(data[i][j]);
	}
	
	private void round(double[][] data, int col) {
		for(int i=0; i<data.length; i++)
			data[i][col]=Math.floor(data[i][col]);
	}
}
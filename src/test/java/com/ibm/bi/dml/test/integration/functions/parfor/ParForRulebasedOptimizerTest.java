/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForRulebasedOptimizerTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "parfor_optimizer1";
	private final static String TEST_NAME2 = "parfor_optimizer2";
	private final static String TEST_NAME3 = "parfor_optimizer3";
	private final static String TEST_DIR = "functions/parfor/";
	private final static double eps = 1e-10;
	
	
	private final static int rows1 = 1000; //small CP
	private final static int rows2 = 10000; //large MR
	
	private final static int cols11 = 50;  //small single parfor
	private final static int cols12 = 500; //large single parfor	
	
	private final static int cols21 = 5;  //small nested parfor
	private final static int cols22 = 50; //large nested parfor
	private final static int cols31 = 2;  //small nested parfor
	private final static int cols32 = 8; //large nested parfor
	
	
	private final static double sparsity = 0.7;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "Rout" })   ); 
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "Rout" })   ); 
		addTestConfiguration(
				TEST_NAME3, 
				new TestConfiguration(TEST_DIR, TEST_NAME3, 
				new String[] { "Rout" })   ); 		
	}

	
	@Test
	public void testParForOptimizerCorrelationSmallSmall() 
	{
		runParForOptimizerTest(1, false, false);
	}
	
	
	@Test
	public void testParForOptimizerCorrelationSmallLarge() 
	{
		runParForOptimizerTest(1, false, true);
	}
	
	
	@Test
	public void testParForOptimizerCorrelationLargeSmall() 
	{
		runParForOptimizerTest(1, true, false);
	}
	
	@Test
	public void testParForOptimizerCorrelationLargeLarge() 
	{
		runParForOptimizerTest(1, true, true);
	}
	
	
	@Test
	public void testParForOptimizerBivariateStatsSmallSmall() 
	{
		runParForOptimizerTest(2, false, false);
	}
	
	@Test
	public void testParForOptimizerBivariateStatsSmallLarge() 
	{
		runParForOptimizerTest(2, false, true);
	}
	
	@Test
	public void testParForOptimizerBivariateStatsLargeSmall() 
	{
		runParForOptimizerTest(2, true, false);
	}
	
	@Test
	public void testParForOptimizerBivariateStatsLargeLarge() 
	{
		runParForOptimizerTest(2, true, true);
	}
	
	@Test
	public void testParForOptimizerFunctionInvocationSmallSmall() 
	{
		runParForOptimizerTest(3, false, false);
	}
	
	@Test
	public void testParForOptimizerFunctionInvocationSmallLarge() 
	{
		runParForOptimizerTest(3, false, true);
	}
	
	@Test
	public void testParForOptimizerFunctionInvocationLargeSmall() 
	{
		runParForOptimizerTest(3, true, false);
	}
	
	@Test
	public void testParForOptimizerFunctionInvocationLargeLarge() 
	{
		runParForOptimizerTest(3, true, true);
	}
	
	
	private void runParForOptimizerTest( int scriptNum, boolean largeRows, boolean largeCols )
	{
		//find right rows and cols configuration
		int rows=-1, cols=-1;  
		if( largeRows )
			rows = rows2;
		else
			rows = rows1; 
		if( largeCols ){
			switch(scriptNum)
			{
				case 1: cols=cols22; break;
				case 2: cols=cols32; break;
				case 3: cols=cols12; break;				
			}
		}
		else{
			switch(scriptNum)
			{
				case 1: cols=cols21; break;
				case 2: cols=cols31; break;
				case 3: cols=cols11; break;				
			}
		}

		//run actual test
		switch( scriptNum )
		{
			case 1: 
				runUnaryTest(scriptNum, rows, cols);
				break;
			case 2:
				runNaryTest(scriptNum, rows, cols);
				break;
			case 3: 
				runUnaryTest(scriptNum, rows, cols);
				break;	
		}
	}
	
	private void runUnaryTest(int scriptNum, int rows, int cols )
	{
		TestConfiguration config = null;
		String HOME = SCRIPT_DIR + TEST_DIR;
		if( scriptNum==1 )
		{
			config=getTestConfiguration(TEST_NAME1);
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
		}
		else if( scriptNum==3 )
		{
			config=getTestConfiguration(TEST_NAME3);
			fullDMLScriptName = HOME + TEST_NAME3 + ".dml";
		}
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		if( scriptNum==1 )
		{
			programArgs = new String[]{ "-args", HOME + INPUT_DIR + "V" , 
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        HOME + OUTPUT_DIR + "R" };
			rCmd = "Rscript" + " " + HOME + TEST_NAME1 + ".R" + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		}	
		else if( scriptNum==3 )
		{
			programArgs = new String[]{ "-args", HOME + INPUT_DIR + "V" , 
							                Integer.toString(rows),
							                Integer.toString(cols),
							                Integer.toString(cols/2),
							                HOME + OUTPUT_DIR + "R" };	
			rCmd = "Rscript" + " " + HOME + TEST_NAME3 + ".R" + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		}	
		
		
		loadTestConfiguration(config);

		long seed = System.nanoTime();
        double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
		writeInputMatrix("V", V, true);

		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
	
		runRScript(true);
		
		//compare matrices
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");		
	}
	
	private void runNaryTest(int scriptNum, int rows, int cols)
	{
		TestConfiguration config = getTestConfiguration(TEST_NAME2);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME2 + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "D" ,
				                        HOME + INPUT_DIR + "S1" ,
				                        HOME + INPUT_DIR + "S2" ,
				                        HOME + INPUT_DIR + "K1" ,
				                        HOME + INPUT_DIR + "K2" ,
				                        HOME + OUTPUT_DIR + "bivarstats",
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        Integer.toString(cols),
				                        Integer.toString(cols*cols),
				                        Integer.toString(7)
				                         };
		
				
		rCmd = "Rscript" + " " + HOME + TEST_NAME2 + ".R" + " " + 
		       HOME + INPUT_DIR + " " + Integer.toString(7) + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		//generate actual dataset
		double[][] D = getRandomMatrix(rows, cols, 1, 7, 1.0, 7777); 
		double[] Dkind = new double[cols]; 
		for( int i=0; i<cols; i++ )
		{
			Dkind[i]=(i%3)+1;//kind 1,2,3
			if( Dkind[i]!=1 )
				round(D,i); //for ordinal and categorical vars
		}
		writeInputMatrix("D", D, true);
		
		//generate attribute sets		
        double[][] S1 = getRandomMatrix(1, cols, 1, cols+1-eps, 1, 1112);
        double[][] S2 = getRandomMatrix(1, cols, 1, cols+1-eps, 1, 1113);
        round(S1);
        round(S2);
		writeInputMatrix("S1", S1, true);
		writeInputMatrix("S2", S2, true);	

		//generate kind for attributes (1,2,3)
        double[][] K1 = new double[1][cols];
        double[][] K2 = new double[1][cols];
        for( int i=0; i<cols; i++ )
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
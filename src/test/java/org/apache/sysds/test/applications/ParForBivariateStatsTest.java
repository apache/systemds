/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.applications;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class ParForBivariateStatsTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "parfor_bivariate";
	private final static String TEST_DIR = "applications/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForBivariateStatsTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1000;  // # of rows in each vector (for CP instructions) 
	private final static int rows2 = (int) (Hop.CPThreshold+1);  // # of rows in each vector (for MR instructions)
	private final static int cols = 30;      // # of columns in each vector  
	private final static int cols2 = 10;      // # of columns in each vector - initial test: 7 
	
	private final static double minVal=1;    // minimum value in each vector 
	private final static double maxVal=5; // maximum value in each vector 

	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Rout" }) );
	}

	@Test
	public void testParForBivariateStatsLocalRemoteCP() {
		runParForBivariateStatsTest(true, PExecMode.LOCAL, PExecMode.REMOTE_SPARK, ExecType.CP);
	}
	
	@Test
	public void testParForBivariateStatsRemoteLocalCP() {
		runParForBivariateStatsTest(true, PExecMode.REMOTE_SPARK, PExecMode.LOCAL, ExecType.CP);
	}
	
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
			if( inner == PExecMode.REMOTE_SPARK )      scriptNum=2;
			else if( outer == PExecMode.REMOTE_SPARK ) scriptNum=3;
			else if( outer == PExecMode.LOCAL )        scriptNum=1;
			else                                       scriptNum=4; //optimized
		}
		else
		{
			scriptNum = 0;
		}
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		//config.addVariable("rows", rows);
		//config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME +scriptNum + ".dml";
		programArgs = new String[]{"-args", input("D"),
			input("S1"), input("S2"), input("K1"), input("K2"), output("bivarstats"),
			Integer.toString(rows), Integer.toString(cols), Integer.toString(cols2),
			Integer.toString(cols2*cols2), Integer.toString((int)maxVal) };
		
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
			inputDir() + " " + Integer.toString((int)maxVal) + " " + expectedDir();

		//generate actual dataset
		double[][] D = getRandomMatrix(rows, cols, minVal, maxVal, 1, 7777); 
		double[] Dkind = new double[cols]; 
		for( int i=0; i<cols; i++ )
		{
			Dkind[i]=(i%3)+1;//kind 1,2,3
			if( Dkind[i]!=1 )
				TestUtils.floor(D,i); //for ordinal and categorical vars
		}
		writeInputMatrix("D", D, true);
		
		//generate attribute sets		
		double[][] S1 = getRandomMatrix(1, cols2, 1, cols+1-eps, 1, 1112);
		double[][] S2 = getRandomMatrix(1, cols2, 1, cols+1-eps, 1, 1113);
		TestUtils.floor(S1);
		TestUtils.floor(S2);
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
		runTest(true, exceptionExpected, null, 92);

		runRScript(true);
		
		//compare matrices 
		for( String out : new String[]{"bivar.stats", "category.counts", "category.means",  "category.variances" } )
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("bivarstats/"+out);
			
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS(out);
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
	}
}
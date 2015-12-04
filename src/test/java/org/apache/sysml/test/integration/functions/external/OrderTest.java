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

package org.apache.sysml.test.integration.functions.external;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

public class OrderTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "Order";
	private final static String TEST_DIR = "functions/external/";
	
	private final static int rows = 1200;
	private final static int cols = 1100; 
	private final static int sc = 1; 

	private final static double sparsity = 0.7;
	private final static double eps = 1e-10;
	
	@Override
	public void setUp() {
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "B.mtx" })   ); 
	}

	@Test
	public void testOrderAsc()
	{
		runtestOrder( true ); 
	}
	
	@Test
	public void testOrderDesc()
	{
		runtestOrder( false ); 
	}
	
	private void runtestOrder( boolean asc ) 
	{	
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		int sortcol = sc * (asc ? 1 : -1);
		int namesuffix = (asc ? 1 : 2);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + namesuffix + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "A" , 
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        Integer.toString(sc),
				                        HOME + OUTPUT_DIR + "B" };
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + sortcol + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		try 
		{
			long seed = System.nanoTime();
	        double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
			writeInputMatrix("A", A, true);
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			//check number of compiled and executed scripts (assumes IPA and recompile)
			Assert.assertEquals("Unexpected number of compiled MR jobs.", 1, Statistics.getNoOfCompiledMRJobs()); //reblock
			Assert.assertEquals("Unexpected number of executed MR jobs.", 0, Statistics.getNoOfExecutedMRJobs());
			
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B.mtx");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}
}

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

package org.apache.sysml.test.integration.functions.recompile;

import java.io.IOException;
import java.util.HashMap;

import org.junit.Assert;

import org.junit.Test;

import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

public class ReblockRecompileTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "rblk_recompile1";
	private final static String TEST_NAME2 = "rblk_recompile2";
	private final static String TEST_NAME3 = "rblk_recompile3";
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReblockRecompileTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 20;   
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME3, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "Rout" }) );
	}
	
	@Test
	public void testReblockPWrite() 
	{
		runReblockTest(1);
	}

	@Test
	public void testReblockCTable() 
	{
		runReblockTest(2);
	}
	
	@Test
	public void testReblockGroupedAggregate() 
	{
		runReblockTest(3);
	}
	
	private void runReblockTest(int scriptNum)
	{
		String TEST_NAME = null;
		switch(scriptNum) 
		{
			case 1: TEST_NAME = TEST_NAME1; break;
			case 2: TEST_NAME = TEST_NAME2; break;
			case 3: TEST_NAME = TEST_NAME3; break;
		}
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		loadTestConfiguration(config);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", input("V"), Integer.toString(rows), output("R") };
		
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

		long seed = System.nanoTime();
        double[][] V = getRandomMatrix(rows, 1, 1, 5, 1.0d, seed);
		writeInputMatrix("V", V, true);
		
		//cleanup previous executions
		try {
			MapReduceTool.deleteFileIfExistOnHDFS(output("R"));
		} catch (IOException e1){}
		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1); //0 due to recompile 
		runRScript(true);
		
		Assert.assertEquals("Unexpected number of executed MR jobs.", 
				  			0, Statistics.getNoOfExecutedMRJobs());
		
		//compare matrices		
		try 
		{
			MatrixBlock mo = DataConverter.readMatrixFromHDFS(output("R"), InputInfo.BinaryBlockInputInfo, rows, 1, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
			HashMap<CellIndex, Double> dmlfile = new HashMap<CellIndex,Double>();
			for( int i=0; i<mo.getNumRows(); i++ )
				for( int j=0; j<mo.getNumColumns(); j++ )
					dmlfile.put(new CellIndex(i+1,j+1), mo.getValue(i, j));
				
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
			boolean flag = TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
			if( !flag )
				System.out.println("Matrix compare found differences for input data generated with seed="+seed);
		} 
		catch (IOException e) {
			Assert.fail(e.getMessage());
		}
	}
}
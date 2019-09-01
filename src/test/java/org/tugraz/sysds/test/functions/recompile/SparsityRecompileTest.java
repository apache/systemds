/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.test.functions.recompile;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.conf.CompilerConfig;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.HDFSTool;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;
import org.tugraz.sysds.utils.Statistics;

public class SparsityRecompileTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_NAME1 = "while_recompile_sparse";
	private final static String TEST_NAME2 = "if_recompile_sparse";
	private final static String TEST_NAME3 = "for_recompile_sparse";
	private final static String TEST_NAME4 = "parfor_recompile_sparse";
	private final static String TEST_CLASS_DIR = TEST_DIR + SparsityRecompileTest.class.getSimpleName() + "/";
	
	private final static long rows = 1000;
	private final static long cols = 1000000;
	private final static double sparsity = 0.00001d;
	private final static double val = 7.0;
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME3, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME4, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "Rout" }) );
	}

	@Test
	public void testWhileRecompile() {
		runRecompileTest(TEST_NAME1, true);
	}
	
	@Test
	public void testWhileNoRecompile() {
		runRecompileTest(TEST_NAME1, false);
	}
	
	@Test
	public void testIfRecompile() {
		runRecompileTest(TEST_NAME2, true);
	}
	
	@Test
	public void testIfNoRecompile() {
		runRecompileTest(TEST_NAME2, false);
	}
	
	@Test
	public void testForRecompile() {
		runRecompileTest(TEST_NAME3, true);
	}
	
	@Test
	public void testForNoRecompile() {
		runRecompileTest(TEST_NAME3, false);
	}
	
	@Test
	public void testParForRecompile() {
		runRecompileTest(TEST_NAME4, true);
	}
	
	@Test
	public void testParForNoRecompile() {
		runRecompileTest(TEST_NAME4, false);
	}

	private void runRecompileTest( String testname, boolean recompile )
	{
		boolean oldFlagRecompile = CompilerConfig.FLAG_DYN_RECOMPILE;
		
		try
		{
			getAndLoadTestConfiguration(testname);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "-args",
				input("V"), Double.toString(val), output("R") };

			CompilerConfig.FLAG_DYN_RECOMPILE = recompile;
			
			MatrixBlock mb = MatrixBlock.randOperations((int)rows, (int)cols, sparsity, 0, 1, "uniform", System.currentTimeMillis());
			MatrixCharacteristics mc = new MatrixCharacteristics(rows,cols,OptimizerUtils.DEFAULT_BLOCKSIZE,(long)(rows*cols*sparsity));
			
			DataConverter.writeMatrixToHDFS(mb, input("V"), OutputInfo.TextCellOutputInfo, mc);
			HDFSTool.writeMetaDataFile(input("V.mtd"), ValueType.FP64, mc, OutputInfo.TextCellOutputInfo);
			
			runTest(true, false, null, -1);
			
			//CHECK compiled Spark jobs
			int expectNumCompiled = (testname.equals(TEST_NAME2)?3:4) //-1 for if
				+ (testname.equals(TEST_NAME4)?3:0);//(+2 resultmerge, 1 sum)
			Assert.assertEquals("Unexpected number of compiled Spark jobs.", 
				expectNumCompiled, Statistics.getNoOfCompiledSPInst());
		
			//CHECK executed Spark jobs
			int expectNumExecuted = recompile ?
				((testname.equals(TEST_NAME4))?2:0) : //(+2 resultmerge)
				(testname.equals(TEST_NAME2)?3:4) //reblock + 3 (-1 for if)
					+ ((testname.equals(TEST_NAME4))?3:0); //(+2 resultmerge, 1 sum) 
			Assert.assertEquals("Unexpected number of executed Spark jobs.", 
				expectNumExecuted, Statistics.getNoOfExecutedSPInst());
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Assert.assertEquals((Double)val, dmlfile.get(new CellIndex(1,1)));
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			CompilerConfig.FLAG_DYN_RECOMPILE = oldFlagRecompile;
		}
	}
}

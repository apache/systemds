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

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.HDFSTool;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;
import org.tugraz.sysds.utils.Statistics;

/**
 * This test ensures that recursive functions are not marked for recompile-once 
 * during IPA because this could potentially lead to incorrect plans that cause 
 * OOMs or even incorrect results. 
 *
 */
public class RecursiveFunctionRecompileTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_NAME1 = "recursive_func_direct";
	private final static String TEST_NAME2 = "recursive_func_indirect";
	private final static String TEST_NAME3 = "recursive_func_indirect2";
	private final static String TEST_NAME4 = "recursive_func_none";
	private final static String TEST_CLASS_DIR = TEST_DIR + 
		RecursiveFunctionRecompileTest.class.getSimpleName() + "/";
	
	private final static long rows = 5000;
	private final static long cols = 10000;    
	private final static double sparsity = 0.00001d;    
	private final static double val = 7.0;
	
	
	@Override
	public void setUp() {
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
	public void testDirectRecursionRecompileIPA() {
		runRecompileTest(TEST_NAME1, true);
	}
	
	@Test
	public void testIndirectRecursionRecompileIPA() {
		runRecompileTest(TEST_NAME2, true);
	}
	
	@Test
	public void testIndirect2RecursionRecompileIPA() {
		runRecompileTest(TEST_NAME3, true);
	}
	
	@Test
	public void testNoRecursionRecompileIPA() {
		runRecompileTest(TEST_NAME4, true);
	}
	
	@Test
	public void testDirectRecursionRecompileNoIPA() {
		runRecompileTest(TEST_NAME1, false);
	}
	
	@Test
	public void testIndirectRecursionRecompileNoIPA() {
		runRecompileTest(TEST_NAME2, false);
	}
	
	@Test
	public void testIndirect2RecursionRecompileNoIPA() {
		runRecompileTest(TEST_NAME3, false);
	}
	
	@Test
	public void testNoRecursionRecompileNoIPA() {
		runRecompileTest(TEST_NAME4, false);
	}
	
	private void runRecompileTest( String testname, boolean IPA )
	{	
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain","-stats","-args",
				input("V"), Double.toString(val), output("R") };

			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			
			//generate sparse input data
			MatrixBlock mb = MatrixBlock.randOperations((int)rows, (int)cols, sparsity, 0, 1, "uniform", 732);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows,cols,OptimizerUtils.DEFAULT_BLOCKSIZE,(long)(rows*cols*sparsity));
			DataConverter.writeMatrixToHDFS(mb, input("V"), OutputInfo.TextCellOutputInfo, mc);
			HDFSTool.writeMetaDataFile(input("V.mtd"), ValueType.FP64, mc, OutputInfo.TextCellOutputInfo);
			
			//run test
			runTest(true, false, null, -1); 
			
			//check number of recompiled functions (recompile_once is not applicable for recursive functions
			//because the single recompilation on entry would implicitly change the remaining plan of the caller;
			//if not not handled correctly, TEST_NAME1 and TEST_NAME2 would have show with IPA 1111 function recompilations. 
			Assert.assertEquals(testname.equals(TEST_NAME4) && IPA ? 1 : 0, Statistics.getFunRecompiles());
		}
		catch(Exception ex) {
			ex.printStackTrace();
			Assert.fail("Failed to run test: "+ex.getMessage());
		}
		finally {
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
		}
	}	
}
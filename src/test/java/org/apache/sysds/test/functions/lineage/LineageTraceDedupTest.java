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

package org.apache.sysds.test.functions.lineage;

import org.junit.Test;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageRecomputeUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class LineageTraceDedupTest extends AutomatedTestBase
{
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME = "LineageTraceDedup";
	protected static final String TEST_NAME1 = "LineageTraceDedup1";
	protected static final String TEST_NAME10 = "LineageTraceDedup10";
	protected static final String TEST_NAME2 = "LineageTraceDedup2";
	protected static final String TEST_NAME3 = "LineageTraceDedup3";
	protected static final String TEST_NAME4 = "LineageTraceDedup4";
	protected static final String TEST_NAME5 = "LineageTraceDedup5";
	protected static final String TEST_NAME6 = "LineageTraceDedup6";
	protected static final String TEST_NAME7 = "LineageTraceDedup7"; //nested if-else branches
	protected static final String TEST_NAME8 = "LineageTraceDedup8"; //while loop
	protected static final String TEST_NAME9 = "LineageTraceDedup9"; //while loop w/ if
	protected static final String TEST_NAME11 = "LineageTraceDedup11"; //mini-batch
	
	protected String TEST_CLASS_DIR = TEST_DIR + LineageTraceDedupTest.class.getSimpleName() + "/";
	
	protected static final int numRecords = 10;
	protected static final int numFeatures = 5;
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=11; i++)
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}
	
	@Test
	public void testLineageTrace1() {
		testLineageTrace(TEST_NAME1);
	}

	@Test
	public void testLineageTrace10() {
		testLineageTrace(TEST_NAME10);
	}
	
	@Test
	public void testLineageTrace2() {
		testLineageTrace(TEST_NAME2);
	}
	
	@Test
	public void testLineageTrace3() {
		testLineageTrace(TEST_NAME3);
	}
	
	@Test
	public void testLineageTrace4() {
		testLineageTrace(TEST_NAME4);
	}
	
	@Test
	public void testLineageTrace5() {
		testLineageTrace(TEST_NAME5);
	}
	
	@Test
	public void testLineageTrace6() {
		testLineageTrace(TEST_NAME6);
	}

	@Test
	public void testLineageTrace7() {
		testLineageTrace(TEST_NAME7);
	}
	
	@Test
	public void testLineageTrace8() {
		testLineageTrace(TEST_NAME8);
	}
	
	@Test
	public void testLineageTrace9() {
		testLineageTrace(TEST_NAME9);
	}

	@Test
	public void testLineageTrace11() {
		testLineageTrace(TEST_NAME11);
	}
	
	public void testLineageTrace(String testname) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		Types.ExecMode old_rtplatform = AutomatedTestBase.rtplatform;
		
		try {
			System.out.println("------------ BEGIN " + testname + "------------");
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			AutomatedTestBase.rtplatform = Types.ExecMode.SINGLE_NODE;
			
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();
			List<String> proArgs;

			// w/o lineage deduplication
			proArgs = new ArrayList<>();
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("dedup");
			proArgs.add("-args");
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);

			Lineage.resetInternalState();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			//deserialize, generate program and execute
			String Rtrace = readDMLLineageFromHDFS("R");
			String RDedupPatches = readDMLLineageDedupFromHDFS("R");
			Data ret = LineageRecomputeUtils.parseNComputeLineageTrace(Rtrace, RDedupPatches);
			
			//match the original and recomputed results
			HashMap<CellIndex, Double> orig = readDMLMatrixFromHDFS("R");
			MatrixBlock recomputed = ((MatrixObject)ret).acquireReadAndRelease();
			TestUtils.compareMatrices(orig, recomputed, 1e-6);
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
			AutomatedTestBase.rtplatform = old_rtplatform;
			Recompiler.reinitRecompiler(); 
		}
	}
}

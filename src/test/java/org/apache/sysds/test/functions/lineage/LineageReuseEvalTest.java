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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

public class LineageReuseEvalTest extends LineageBase {
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME = "LineageReuseEval";
	protected static final int TEST_VARIANTS = 2;
	protected String TEST_CLASS_DIR = TEST_DIR + LineageReuseEvalTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for( int i=1; i<=TEST_VARIANTS; i++ )
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}

	//FIXME: These tests fail/get stuck in a deadlock if MultiLevel/Hybrid is used.
	//This problem is not yet reproducible locally. 
	@Test
	public void testGridsearchLM() {
		testLineageTrace(TEST_NAME+"1", ReuseCacheType.REUSE_FULL);
	}

	@Test
	public void testGridSearchMLR() {
		testLineageTrace(TEST_NAME+"2", ReuseCacheType.REUSE_FULL);
		//FIXME: 2x slower with reuse. Heavy hitter function is lineageitem equals.
		//This problem only exists with parfor.
	}
	
	public void testLineageTrace(String testname, ReuseCacheType reuseType) {
		ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);
		
		try {
			LOG.debug("------------ BEGIN " + testname + "------------");
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();
			
			// Without lineage-based reuse enabled
			List<String> proArgs = new ArrayList<>();
			proArgs.add("-stats");
			proArgs.add("-args");
			proArgs.add(output("X"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			Lineage.resetInternalState();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> X_orig = readDMLMatrixFromOutputDir("X");
			//long numlmDS = Statistics.getCPHeavyHitterCount("m_lmDS");
			long numMM = Statistics.getCPHeavyHitterCount(Opcodes.MMULT.toString());
			
			// With lineage-based reuse enabled
			proArgs.clear();
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add(reuseType.name().toLowerCase());
			proArgs.add("-args");
			proArgs.add(output("X"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			Lineage.resetInternalState();
			
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> X_reused = readDMLMatrixFromOutputDir("X");
			//long numlmDS_reuse = Statistics.getCPHeavyHitterCount("m_lmDS");
			long numMM_reuse = Statistics.getCPHeavyHitterCount(Opcodes.MMULT.toString());
			
			Lineage.setLinReuseNone();
			TestUtils.compareMatrices(X_orig, X_reused, 1e-6, "Origin", "Reused");

			if (testname.equalsIgnoreCase("LineageReuseEval1")) {  //gridSearchLM
				//lmDS call should be reused for all the 7 values of tolerance 
				//Assert.assertTrue("Violated lmDS reuse count: 7 * "+numlmDS_reuse+" == "+numlmDS, 
				//		7*numlmDS_reuse == numlmDS);
				Assert.assertTrue("Violated ba+* reuse count: "+numMM_reuse+" < "+numMM, numMM_reuse < numMM);
			}
		}
		finally {
			rtplatform = platformOld;
			Recompiler.reinitRecompiler();
		}
	}
}
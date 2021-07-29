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

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class PrefetchRDDTest extends AutomatedTestBase {
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME = "PrefetchRDD";
	protected static final int TEST_VARIANTS = 1;
	protected static String TEST_CLASS_DIR = TEST_DIR + PrefetchRDDTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for( int i=1; i<=TEST_VARIANTS; i++ )
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}
	
	@Test
	public void testAsyncSparkOPs1() {
		runTest(TEST_NAME+"1");
	}
	
	public void runTest(String testname) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		ExecMode oldPlatform = setExecMode(ExecMode.HYBRID);
		
		long oldmem = InfrastructureAnalyzer.getLocalMaxMemory();
		long mem = 1024*1024*8;
		InfrastructureAnalyzer.setLocalMaxMemory(mem);
		
		try {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();
			
			List<String> proArgs = new ArrayList<>();
			
			proArgs.add("-stats");
			proArgs.add("-explain");
			//proArgs.add("RECOMPILE_RUNTIME");
			proArgs.add("-args");
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> R = readDMLScalarFromOutputDir("R");
			System.out.println(R);
		} finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
			resetExecMode(oldPlatform);
			InfrastructureAnalyzer.setLocalMaxMemory(oldmem);
			Recompiler.reinitRecompiler();
		}
	}
}
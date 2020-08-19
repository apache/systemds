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

import org.junit.Test;
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

@net.jcip.annotations.NotThreadSafe
public class LineageTraceParforTest extends AutomatedTestBase {
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "LineageTraceParfor1"; //rand - matrix result - local parfor
	protected static final String TEST_NAME2 = "LineageTraceParfor2"; //rand - matrix result - remote spark parfor
	protected static final String TEST_NAME3 = "LineageTraceParfor3"; //rand - matrix result - remote spark parfor
	protected static final String TEST_NAME4 = "LineageTraceParforSteplm"; //rand - steplm
	protected static final String TEST_NAME5 = "LineageTraceParforKmeans"; //rand - kmeans
	protected static final String TEST_NAME6 = "LineageTraceParforMSVM"; //rand - msvm remote parfor
	
	protected String TEST_CLASS_DIR = TEST_DIR + LineageTraceParforTest.class.getSimpleName() + "/";
	
	protected static final int numRecords = 50;
	
	public LineageTraceParforTest() {
		
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"R"}) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {"R"}) );
		addTestConfiguration( TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {"R"}) );
		addTestConfiguration( TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] {"R"}) );
	}
	
	@Test
	public void testLineageTraceParFor1_1() {
		testLineageTraceParFor(1, TEST_NAME1);
	}
	
	@Test
	public void testLineageTraceParFor1_2() {
		testLineageTraceParFor(2, TEST_NAME1);
	}
	
	@Test
	public void testLineageTraceParFor1_8() {
		testLineageTraceParFor(8, TEST_NAME1);
	}
	
	@Test
	public void testLineageTraceParFor1_32() {
		testLineageTraceParFor(32, TEST_NAME1);
	}
	
	@Test
	public void testLineageTraceParFor2_1() {
		testLineageTraceParFor(1, TEST_NAME2);
	}
	
	@Test
	public void testLineageTraceParFor2_2() {
		testLineageTraceParFor(2, TEST_NAME2);
	}
	
	@Test
	public void testLineageTraceParFor2_8() {
		testLineageTraceParFor(8, TEST_NAME2);
	}
	
	@Test
	public void testLineageTraceParFor2_32() {
		testLineageTraceParFor(32, TEST_NAME2);
	}
	
	@Test
	public void testLineageTraceParFor3_8() {
		testLineageTraceParFor(8, TEST_NAME3);
	}
	
	@Test
	public void testLineageTraceParFor3_32() {
		testLineageTraceParFor(32, TEST_NAME3);
	}
	
	@Test
	public void testLineageTraceSteplm_8() {
		testLineageTraceParFor(8, TEST_NAME4);
	}
	
	@Test
	public void testLineageTraceSteplm_32() {
		testLineageTraceParFor(32, TEST_NAME4);
	}
	
	@Test
	public void testLineageTraceKMeans_8() {
		testLineageTraceParFor(8, TEST_NAME5);
	}
	
	@Test
	public void testLineageTraceKmeans_32() {
		testLineageTraceParFor(32, TEST_NAME5);
	}
	
	@Test
	public void testLineageTraceMSVM_Remote64() {
		testLineageTraceParFor(64, TEST_NAME6);
	}
	
	private void testLineageTraceParFor(int ncol, String testname) {
		try {
			System.out.println("------------ BEGIN " + testname + "------------");
			
			getAndLoadTestConfiguration(testname);
			List<String> proArgs = new ArrayList<>();
			
			proArgs.add("-lineage");
			proArgs.add("-args");
			proArgs.add(output("R"));
			proArgs.add(String.valueOf(numRecords));
			proArgs.add(String.valueOf(ncol));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			fullDMLScriptName = getScript();
			 
			//run the test
			Lineage.resetInternalState();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			
			//get lineage and generate program
			String Rtrace = readDMLLineageFromHDFS("R");
			Data ret = LineageRecomputeUtils.parseNComputeLineageTrace(Rtrace, null);

			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			MatrixBlock tmp = ((MatrixObject) ret).acquireReadAndRelease();
			TestUtils.compareMatrices(dmlfile, tmp, 1e-6);
		}
		finally {
			Recompiler.reinitRecompiler();
		}
	}
}

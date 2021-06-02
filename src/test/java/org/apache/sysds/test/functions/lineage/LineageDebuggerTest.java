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

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageDebugger;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageParser;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Explain;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class LineageDebuggerTest extends LineageBase {
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "LineageDebugger1";
	protected String TEST_CLASS_DIR = TEST_DIR + LineageDebuggerTest.class.getSimpleName() + "/";
	
	protected static final int numRecords = 10;
	protected static final int numFeatures = 5;
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
	}
	
	@Test
	public void testLineageDebuggerNaN() {
		testLineageDebuggerNaN(TEST_NAME1);
	}
	
	@Test
	public void testLineageDebuggerInf1() {
		testLineageDebuggerInf(TEST_NAME1, true, false);
	}
	
	@Test
	public void testLineageDebuggerInf2() { testLineageDebuggerInf(TEST_NAME1, false, true); }
	
	@Test
	public void testLineageDebuggerInf3() { testLineageDebuggerInf(TEST_NAME1, true, true); }
	
	@Test
	public void testLineageDebuggerFirstOccurrence1() { testLineageDebuggerFirstOccurrence(TEST_NAME1); }
	
	public void testLineageDebuggerNaN(String testname) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		
		try {
			LOG.debug("------------ BEGIN " + testname + "------------");
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			
			int rows = numRecords;
			int cols = numFeatures;
			
			getAndLoadTestConfiguration(testname);
			
			List<String> proArgs = new ArrayList<>();
			
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("debugger");			
			proArgs.add("-args");
			proArgs.add(input("X"));
			proArgs.add(input("Y"));
			proArgs.add(output("Z"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			
			fullDMLScriptName = getScript();
			
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
			writeInputMatrixWithMTD("X", X, true);
			
			double[][] Y = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
			Y[3][3] = Double.NaN;
			writeInputMatrixWithMTD("Y", Y, true);
			
			Lineage.resetInternalState();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			
			String Z_lineage = readDMLLineageFromHDFS("Z");
			LineageItem Z_li = LineageParser.parseLineageTrace(Z_lineage);
			TestUtils.compareScalars(Z_lineage, Explain.explain(Z_li));
			
			TestUtils.compareScalars(true, Z_li.getSpecialValueBit(LineageDebugger.POS_NAN));
			TestUtils.compareScalars(false, Z_li.getSpecialValueBit(LineageDebugger.POS_NEGATIVE_INFINITY));
			TestUtils.compareScalars(false, Z_li.getSpecialValueBit(LineageDebugger.POS_POSITIVE_INFINITY));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
			Recompiler.reinitRecompiler(); 
		}
	}
	
	public void testLineageDebuggerInf(String testname, boolean positive, boolean negative) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		
		try {
			LOG.debug("------------ BEGIN " + testname + "------------");
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			
			int rows = numRecords;
			int cols = numFeatures;
			
			getAndLoadTestConfiguration(testname);
			
			List<String> proArgs = new ArrayList<>();
			
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("debugger");
			proArgs.add("-args");
			proArgs.add(input("X"));
			proArgs.add(input("Y"));
			proArgs.add(output("Z"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			
			fullDMLScriptName = getScript();
			
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
			writeInputMatrixWithMTD("X", X, true);
			
			double[][] Y = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
			if (positive)
				Y[2][2] = Double.POSITIVE_INFINITY;
			if (negative)
				Y[3][3] = Double.NEGATIVE_INFINITY;
			writeInputMatrixWithMTD("Y", Y, true);
			
			Lineage.resetInternalState();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			
			String Z_lineage = readDMLLineageFromHDFS("Z");
			LineageItem Z_li = LineageParser.parseLineageTrace(Z_lineage);
			TestUtils.compareScalars(Z_lineage, Explain.explain(Z_li));
			
			if (positive)
				TestUtils.compareScalars(true, Z_li.getSpecialValueBit(LineageDebugger.POS_POSITIVE_INFINITY));
			if (negative)
				TestUtils.compareScalars(true, Z_li.getSpecialValueBit(LineageDebugger.POS_NEGATIVE_INFINITY));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
			Recompiler.reinitRecompiler();
		}
	}
	
	public void testLineageDebuggerFirstOccurrence(String testname) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		
		try {
			LOG.debug("------------ BEGIN " + testname + "------------");
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			
			int rows = numRecords;
			int cols = numFeatures;
			
			getAndLoadTestConfiguration(testname);
			
			List<String> proArgs = new ArrayList<>();
			
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("debugger");
			proArgs.add("-args");
			proArgs.add(input("X"));
			proArgs.add(input("Y"));
			proArgs.add(output("Z"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			
			fullDMLScriptName = getScript();
			
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
			
			writeInputMatrixWithMTD("X", X, true);
			
			double[][] Y = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
			Y[3][3] = Double.NaN;
			writeInputMatrixWithMTD("Y", Y, true);
			
			Lineage.resetInternalState();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			
			String Z_lineage = readDMLLineageFromHDFS("Z");
			LineageItem Z_li = LineageParser.parseLineageTrace(Z_lineage);
			TestUtils.compareScalars(Z_lineage, Explain.explain(Z_li));
			
			LineageItem firstOccurrence = LineageDebugger.firstOccurrenceOfNR(Z_li, LineageDebugger.POS_NAN);
			TestUtils.compareScalars(true, firstOccurrence.getSpecialValueBit(LineageDebugger.POS_NAN));
			for (LineageItem li : firstOccurrence.getInputs())
				TestUtils.compareScalars(false, li.getSpecialValueBit(LineageDebugger.POS_NAN));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
			Recompiler.reinitRecompiler();
		}
	}
	
}

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

package org.apache.sysds.test.functions.countDistinct;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public abstract class CountDistinctBase extends AutomatedTestBase {

	protected abstract String getTestClassDir();

	protected abstract String getTestName();

	protected abstract String getTestDir();

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(),
			new TestConfiguration(getTestClassDir(), getTestName(), new String[] {"A.scalar"}));
	}

	protected double percentTolerance = 0.0;
	protected double baseTolerance = 0.0001;

	@Test
	public void testSmall() {
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		double tolerance = baseTolerance + 50 * percentTolerance;
		countDistinctTest(50, 50, 50, 1.0, ex, tolerance);
	}

	@Test
	public void testLarge() {
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		double tolerance = baseTolerance + 800 * percentTolerance;
		countDistinctTest(800, 1000, 1000, 1.0, ex, tolerance);
	}

	@Test
	public void testXLarge() {
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		double tolerance = baseTolerance + 1723 * percentTolerance;
		countDistinctTest(1723, 5000, 2000, 1.0, ex, tolerance);
	}

	@Test
	public void test1Unique() {
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		double tolerance = 0.00001;
		countDistinctTest(1, 100, 1000, 1.0, ex, tolerance);
	}

	@Test
	public void test2Unique() {
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		double tolerance = 0.00001;
		countDistinctTest(2, 100, 1000, 1.0, ex, tolerance);
	}

	@Test
	public void test120Unique() {
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		double tolerance = 0.00001 + 120 * percentTolerance;
		countDistinctTest(120, 100, 1000, 1.0, ex, tolerance);
	}

	@Test
	public void testSparse500Unique() {
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		double tolerance = 0.00001 + 500 * percentTolerance;
		countDistinctTest(500, 100, 640000, 0.1, ex, tolerance);
	}

	@Test
	public void testSparse120Unique(){
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		double tolerance = 0.00001 + 120 * percentTolerance;
		countDistinctTest(120, 100, 64000, 0.1, ex, tolerance);
	}

	public void countDistinctTest(int numberDistinct, int cols, int rows, double sparsity,
		LopProperties.ExecType instType, double tolerance) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(getTestName()));
			String HOME = SCRIPT_DIR + getTestDir();
			fullDMLScriptName = HOME + getTestName() + ".dml";
			String out = output("A");
			System.out.println(out);
			programArgs = new String[] {"-args", String.valueOf(numberDistinct), String.valueOf(rows),
				String.valueOf(cols), String.valueOf(sparsity), out};

			runTest(true, false, null, -1);
			writeExpectedScalar("A", numberDistinct);
			compareResults(tolerance);
		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Exception in execution: " + e.getMessage(), false);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
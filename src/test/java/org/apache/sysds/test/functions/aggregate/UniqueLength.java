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

package org.apache.sysds.test.functions.aggregate;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class UniqueLength extends AutomatedTestBase {

	private final static String TEST_NAME = "unique_length";
	private final static String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + UniqueLength.class.getSimpleName() + "/";

	private static String[] esT = new String[] {
		// The different types of Estimators
		"count", // EstimatorType.NUM_DISTINCT_COUNT,
		// EstimatorType.NUM_DISTINCT_KMV,
		// EstimatorType.NUM_DISTINCT_HYPER_LOG_LOG
	};

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"A.scalar"}));
	}

	@Parameters
	public static Collection<Object[]> data() {
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		ArrayList<Object[]> tests = new ArrayList<>();
		for(String type : esT) {
			tests.add(new Object[] {1, 1, 1, ex, type});
			// tests.add(new Object[] {100, 100, 100, ex, type});
			// tests.add(new Object[] {1000, 1000, 1000, ex, type});
		}
		return tests;
	}

	@Parameterized.Parameter
	public int numberDistinct;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public int rows;
	@Parameterized.Parameter(3)
	public LopProperties.ExecType instType;
	@Parameterized.Parameter(4)
	public String type;

	@Test
	public void run_unique_length_test() {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			String out = output("A");
			System.out.println(out);
			programArgs = new String[] {"-args", String.valueOf(numberDistinct), String.valueOf(rows),
				String.valueOf(cols), out, type};

			runTest(true, false, null, -1);
			writeExpectedScalar("A", numberDistinct);
			compareResults(0.001);
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
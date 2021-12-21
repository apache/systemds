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

package org.apache.sysds.test.functions.builtin.part2;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class BuiltinRandomForestTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "RandomForest";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinRandomForestTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"C"}));
	}
	
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public int bins;
	@Parameterized.Parameter(3)
	public int depth;
	@Parameterized.Parameter(4)
	public int num_leaf;
	@Parameterized.Parameter(5)
	public int num_trees;
	@Parameterized.Parameter(6)
	public String impurity;
	
	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			//TODO fix randomForest script (currently indexing issues)
			{2000, 7, 4, 7, 10, 1, "Gini"},
			{2000, 7, 4, 7, 10, 1, "entropy"},
			{2000, 7, 4, 3, 5, 3, "Gini"},
			{2000, 7, 4, 3, 5, 3, "entropy"},
		});
	}

	@Ignore
	@Test
	public void testRandomForestSinglenode() {
		runRandomForestTest(ExecMode.SINGLE_NODE);
	}
	
	@Ignore
	@Test
	public void testRandomForestHybrid() {
		runRandomForestTest(ExecMode.HYBRID);
	}
	
	private void runRandomForestTest(ExecMode mode)
	{
		ExecMode platformOld = setExecMode(mode);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",
				input("X"), input("Y"), String.valueOf(bins),
				String.valueOf(depth), String.valueOf(num_leaf),
				String.valueOf(num_trees), impurity, output("B") };

			//generate actual datasets
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.7, 7);
			double[][] Y = TestUtils.round(getRandomMatrix(rows, 1, 1, 5, 1, 3));
			writeInputMatrixWithMTD("X", X, false);
			writeInputMatrixWithMTD("Y", Y, false);

			runTest(true, false, null, -1);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}

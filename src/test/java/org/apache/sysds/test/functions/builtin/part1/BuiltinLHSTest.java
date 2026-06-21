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

package org.apache.sysds.test.functions.builtin.part1;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.HashMap;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;

public class BuiltinLHSTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "lhs";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinLHSTest.class.getSimpleName() + "/";

	public void checkLatinHypercubeValidity(HashMap<CellIndex, Double> m,int N, int d ) {
		for (int col = 1; col <= d; col++) {
            boolean[] seen = new boolean[N + 1];
            for (int row = 1; row <= N; row++) {

                double val = m.get(new CellIndex(row, col));
                int intVal = (int) val;
                assertTrue(intVal >= 1 && intVal <= N);
                
				assertFalse(seen[intVal]);
                seen[intVal] = true;
            }
            for (int i = 1; i <= N; i++) {
                assertTrue(seen[i]);
            }
		}
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	
	@Test
	public void testTwoDim() {
		runLhsTest(5,2);
	}

	@Test
	public void testMultiDim() {
		runLhsTest(10,4);
	}

	private void runLhsTest(int N,int d)
	{
		ExecMode platformOld = setExecMode(ExecMode.HYBRID);

		try 	
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", Integer.toString(N), Integer.toString(d), output("C")};

			//execute test
			runTest(true, false, null, -1);

			
			HashMap<CellIndex, Double> m = readDMLMatrixFromOutputDir("C");
			checkLatinHypercubeValidity(m,N,d);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
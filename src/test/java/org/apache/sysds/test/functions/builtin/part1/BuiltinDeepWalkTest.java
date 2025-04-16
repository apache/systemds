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

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Ignore;
import org.junit.Test;

import java.io.IOException;


public class BuiltinDeepWalkTest extends AutomatedTestBase {

	private final static String TEST_NAME = "deepWalk";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDeepWalkTest.class.getSimpleName() + "/";
	private final static String RESOURCE_DIRECTORY = "./src/test/resources/datasets/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
	}

	@Test
	@Ignore //FIXME
	public void testRunDeepWalkCP() throws IOException {
		runDeepWalk(5, 2, 5, 10, -1, -1, ExecType.CP);
	}

	private void runDeepWalk(int window_size, int embedding_size, int walks_per_vertex, int walk_length,
			double alpha, double beta, ExecType execType) throws IOException
	{
		ExecMode platformOld = setExecMode(execType);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			programArgs = new String[]{
				"-nvargs", "GRAPH=" + RESOURCE_DIRECTORY + "caveman_4_20.ijv",
				"WINDOW_SIZE=" + window_size,
				"EMBEDDING_SIZE=" + embedding_size,
				"WALKS_PER_VERTEX=" + walks_per_vertex,
				"WALK_LENGTH=" + walk_length,
				"OUT_FILE=" + output("B")
			};

			runTest(true, false, null, -1);
			
			// TODO for verification plot the output "B" e.g.: 
			// in python -> clearly separable clusters for this test
		}
		finally {
			rtplatform = platformOld;
		}
	}
}

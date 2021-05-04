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

package org.apache.sysds.test.functions.pipelines;

import org.apache.sysds.common.Types;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public class CleaningTestLogical extends AutomatedTestBase {
	private final static String TEST_NAME = "testLogical";
	private final static String TEST_CLASS_DIR = SCRIPT_DIR + CleaningTestLogical.class.getSimpleName() + "/";

	private static final String RESOURCE = SCRIPT_DIR+"functions/pipelines/";
	private static final String DATA_DIR = DATASET_DIR+ "pipelines/";

	private final static String DIRTY = DATA_DIR+ "dirty.csv";
	private final static String CLEAN = DATA_DIR+ "clean.csv";
	private final static String META = RESOURCE+ "meta/meta_census.csv";

	private static final String PARAM_DIR = "./scripts/pipelines/properties/";
	private final static String PARAM = PARAM_DIR + "param.csv";
	private final static String PRIMITIVES = PARAM_DIR + "primitives.csv";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"R"}));
	}

	@Test
	public void testLogical1() {
		runTestLogical(2, 10, 2, 2, 2, 2,
			"classification", Types.ExecMode.SINGLE_NODE);
	}

	@Ignore
	public void testLogicalSP() {
		runTestLogical(3, 10, 3, 2, 2, 4,
			"classification", Types.ExecMode.SPARK);
	}

	private void runTestLogical(int max_iter, int pipelineLength, int crossfold,
		int num_inst, int num_exec, int n_pop, String target, Types.ExecMode et) {

		//		setOutputBuffering(true);
		String HOME = SCRIPT_DIR+"functions/pipelines/" ;
		Types.ExecMode modeOld = setExecMode(et);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-exec", "singlenode", "-nvargs", "dirtyData="+DIRTY,
				"metaData="+META, "primitives="+PRIMITIVES, "parameters="+PARAM, "max_iter="+ max_iter,
				 "pipLength="+ pipelineLength, "cv="+ crossfold, "num_inst="+ num_inst, "num_exec="+ num_exec,
				"n_pop="+ n_pop,"target="+target, "cleanData="+CLEAN, "O="+output("O")};

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			//expected loss smaller than default invocation
			Assert.assertTrue(TestUtils.readDMLBoolean(output("O")));
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}

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
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinTopkLogicalTest extends AutomatedTestBase {
	private final static String TEST_NAME = "topkLogicalTest";
	private final static String TEST_CLASS_DIR = SCRIPT_DIR + BuiltinTopkLogicalTest.class.getSimpleName() + "/";

	private static final String RESOURCE = SCRIPT_DIR+"functions/pipelines/";
	private static final String DATA_DIR = DATASET_DIR+ "pipelines/";

	private final static String DIRTY = DATA_DIR+ "dirty.csv";
	private final static String META = RESOURCE+ "meta/meta_census.csv";

	private static final String PARAM_DIR = "./scripts/pipelines/properties/";
	private final static String PARAM = PARAM_DIR + "param.csv";
	private final static String PRIMITIVES = PARAM_DIR + "testPrimitives.csv";
	private final static String OUTPUT = RESOURCE+"intermediates/logical.csv";

	private final static double dirtyScore = 0.7;
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"R"}));
	}

	@Test
	public void testLogical1() {
		runTestLogical(4, 2, 2, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testLogical2() {
		runTestLogical(2, 2, 2, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testLogicalHybrid() {
		runTestLogical(3, 3, 2, ExecMode.HYBRID);
	}

	private void runTestLogical(int max_iter,  int num_inst, int num_exec,  Types.ExecMode et) {

//		setOutputBuffering(true);

		String HOME = SCRIPT_DIR+"functions/pipelines/" ;
		Types.ExecMode modeOld = setExecMode(et);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-exec", "singlenode", "-nvargs", "dirtyData="+DIRTY,
				"metaData="+META, "primitives="+PRIMITIVES, "parameters="+PARAM, "max_iter="+ max_iter,
				 "num_inst="+ num_inst, "num_exec="+ num_exec,
				"dirtyScore="+dirtyScore, "output="+OUTPUT, "O="+output("O")};

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			//expected loss smaller than default invocation
			Assert.assertTrue(TestUtils.readDMLBoolean(output("O")));
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}

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

package org.apache.sysds.test.functions.paramserv;

import org.apache.sysds.parser.Statement;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class ParamservLocalNNTestwithNbatches extends AutomatedTestBase {

	private static final String TEST_NAME = "paramserv-nbatches-test";

	private static final String TEST_DIR = "functions/paramserv/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ParamservLocalNNTestwithNbatches.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {}));
	}

	@Test
	public void testParamservBSPNBatchesDisjointContiguous() {
		runDMLTest(3, 2, Statement.PSUpdateType.BSP, Statement.PSFrequency.NBATCHES, 32, Statement.PSScheme.DISJOINT_CONTIGUOUS, 8, false );
	}

	@Test
	public void testParamservBSPNBatchesDisjointContiguousModelAvg() {
		runDMLTest(3, 2, Statement.PSUpdateType.BSP, Statement.PSFrequency.NBATCHES, 32, Statement.PSScheme.DISJOINT_CONTIGUOUS, 8, true );
	}

	@Test
	public void testParamservASPNBatches() {
		runDMLTest(3, 2, Statement.PSUpdateType.ASP, Statement.PSFrequency.NBATCHES, 32, Statement.PSScheme.DISJOINT_CONTIGUOUS, 8, false);
	}

	@Test
	public void testParamservSBPNBatchesDisjointContiguous() {
		runDMLTest(3, 3, Statement.PSUpdateType.SBP, Statement.PSFrequency.NBATCHES, 32, Statement.PSScheme.DISJOINT_CONTIGUOUS, 8, false );
	}

	@Test
	public void testParamservSBPNBatchesDisjointContiguousModelAvg() {
		runDMLTest(3, 3, Statement.PSUpdateType.SBP, Statement.PSFrequency.NBATCHES, 32, Statement.PSScheme.DISJOINT_CONTIGUOUS, 8, true );
	}

	private void runDMLTest(int epochs, int workers, Statement.PSUpdateType utype, Statement.PSFrequency freq, int batchsize, Statement.PSScheme scheme, int nbatches, boolean modelAvg) {
		TestConfiguration config = getTestConfiguration(ParamservLocalNNTestwithNbatches.TEST_NAME);
		loadTestConfiguration(config);
		programArgs = new String[] { "-stats", "-nvargs", "mode=LOCAL", "epochs=" + epochs,
			"workers=" + workers, "utype=" + utype, "freq=" + freq, "batchsize=" + batchsize,
			"scheme=" + scheme, "nbatches=" + nbatches,  "modelAvg=" +modelAvg };
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + ParamservLocalNNTestwithNbatches.TEST_NAME + ".dml";
		runTest(true, false, null, null, -1);
	}
}

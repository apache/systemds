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
import org.junit.Test;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class ParamservSBPTest extends AutomatedTestBase {

	private static final String TEST_NAME = "paramserv-sbp";

	private static final String TEST_DIR = "functions/paramserv/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ParamservSBPTest.class.getSimpleName() + "/";

	private final String HOME = SCRIPT_DIR + TEST_DIR;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {}));
	}

	@Test
	public void testParamservNegativeNumBackupWorkers() {
		runDMLTest(TEST_NAME, "Invalid number of backup workers (with #workers=3): #backup-workers=-1", -1,
			Statement.PSScheme.OVERLAP_RESHUFFLE);
	}

	@Test
	public void testParamservAllBackupWorkers() {
		runDMLTest(TEST_NAME, "Invalid number of backup workers (with #workers=3): #backup-workers=3", 3,
			Statement.PSScheme.OVERLAP_RESHUFFLE);
	}

	@Test
	public void testParamservTooFewEffectiveWorkers() {
		runDMLTest(TEST_NAME,
			"Effective number of workers is smaller or equal to the number of backup workers. Change partitioning scheme to OVERLAP_RESHUFFLE, decrease number of backup workers or increase number of rows in dataset.",
			1, Statement.PSScheme.DISJOINT_CONTIGUOUS);
	}

	@Test
	public void testParamservNormalRun() {
		runDMLTest(TEST_NAME, null, 1, Statement.PSScheme.OVERLAP_RESHUFFLE);
	}

	private void runDMLTest(String testname, String errmsg, int numBackupWorkers, Statement.PSScheme scheme) {
		TestConfiguration config = getTestConfiguration(testname);
		loadTestConfiguration(config);
		programArgs = new String[] {"-explain", "-nvargs", "scheme=" + scheme, "workers=3", "backup_workers=" + numBackupWorkers};
		fullDMLScriptName = HOME + testname + ".dml";
		boolean exceptionExpected = errmsg != null;
		runTest(true, exceptionExpected, DMLRuntimeException.class, errmsg, -1);
	}
}

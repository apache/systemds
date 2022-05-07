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

import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class MiscLineageProbingTest extends LineageBase {
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "MiscProbe1";
	protected static final String TEST_NAME2 = "MiscProbe2";
	
	protected String TEST_CLASS_DIR = TEST_DIR + MiscLineageProbingTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2));
	}
	
	@Test
	public void testLineageTraceLineage1() {
		testLineageTrace(TEST_NAME1);
	}
	
	@Test
	public void testLineageTraceLineage2() {
		testLineageTrace(TEST_NAME2);
	}

	public void testLineageTrace(String testName) {
		try {
			getAndLoadTestConfiguration(testName);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[]{"-explain","-stats", "--lineage", "reuse"};
			
			Timing time = new Timing(true);
			runTest(true, false, null, -1);
			Assert.assertTrue(time.stop() < 20000);
		}
		finally {
			Lineage.setLinReuseNone();
		}
	}
}

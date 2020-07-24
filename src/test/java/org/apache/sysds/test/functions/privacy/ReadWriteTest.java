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

package org.apache.sysds.test.functions.privacy;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.FineGrained.DataRange;
import org.apache.sysds.runtime.privacy.FineGrained.FineGrainedPrivacy;
import org.apache.sysds.runtime.privacy.FineGrained.FineGrainedPrivacyList;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.wink.json4j.JSONObject;
import org.junit.Test;

public class ReadWriteTest extends AutomatedTestBase {

	private static final String TEST_DIR = "functions/privacy/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReadWriteTest.class.getSimpleName() + "/";

	private final int n = 10;
	private final int m = 20;

	@Override
	public void setUp() {
		addTestConfiguration("ReadWriteTest",
			new TestConfiguration(TEST_CLASS_DIR, "ReadWriteTest", new String[]{}));
		addTestConfiguration("ReadWriteTest2",
			new TestConfiguration(TEST_CLASS_DIR, "ReadWriteTest2", new String[]{"b"}));
	}


	@Test
	public void writeFineGrainedPrivacyMetadataTest(){
		TestConfiguration config = availableTestConfigurations.get("ReadWriteTest");
		loadTestConfiguration(config);

		writeA();
		
		JSONObject metadata = getMetaDataJSON("a", "in/");
		assertTrue(metadata.containsKey("fine_grained_privacy"));
	}

	@Test
	public void readAndWriteFineGrainedConstraintsTest(){
		TestConfiguration config = availableTestConfigurations.get("ReadWriteTest2");
		loadTestConfiguration(config);
		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + config.getTestScript() + ".dml";

		double[][] a = writeA();

		writeExpectedMatrix("b", a);
		programArgs = new String[]{"-nvargs",
		"a=" + input("a"), "b=" + output("b"),
		"m=" + m, "n=" + n };
		runTest(true,false,null,-1);
		compareResults(1e-9);

		JSONObject metadata = getMetaDataJSON("b");
		assertTrue(metadata.containsKey("fine_grained_privacy"));
	}

	private double[][] writeA(){
		int k = 15;
		double[][] a = getRandomMatrix(m, n, -1, 1, 1, -1);

		PrivacyConstraint privacyConstraint = new PrivacyConstraint();
		FineGrainedPrivacy fgp = new FineGrainedPrivacyList();
		fgp.put(new DataRange(new long[]{1,2}, new long[]{5,4}), PrivacyLevel.Private);
		fgp.put(new DataRange(new long[]{7,1}, new long[]{9,1}), PrivacyLevel.Private);
		fgp.put(new DataRange(new long[]{10,5}, new long[]{10,9}), PrivacyLevel.PrivateAggregation);
		privacyConstraint.setFineGrainedPrivacyConstraints(fgp);
		MatrixCharacteristics dataCharacteristics = new MatrixCharacteristics(m,n,k,k);
		writeInputMatrixWithMTD("a", a, false, dataCharacteristics, privacyConstraint);
		return a;
	}
	
}
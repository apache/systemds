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

import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

public class PrivacyLineageTest extends AutomatedTestBase {

	private static final String TEST_DIR = "functions/privacy/";
	private final static String TEST_CLASS_DIR = TEST_DIR + PrivacyLineageTest.class.getSimpleName() + "/";

	@Override public void setUp() {
		addTestConfiguration("LineageReuse",
			new TestConfiguration(TEST_CLASS_DIR, "PrivacyLineageTest", new String[]{"c"}));
	}

	@Test
	public void propagatePrivacyWithLineageFullReuseTest() {
		propagationWithLineage(PrivacyConstraint.PrivacyLevel.PrivateAggregation);
	}

	private void propagationWithLineage(PrivacyConstraint.PrivacyLevel privacyLevel) {

		TestConfiguration config = availableTestConfigurations.get("LineageReuse");
		loadTestConfiguration(config);
		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + config.getTestScript() + ".dml";

		int n = 20;
		int m = 20;
		double[][] a = getRandomMatrix(m, n, -1, 1, 1, -1);
		int k = 20;
		double[][] b = getRandomMatrix(n, k, -1, 1, 1, -1);

		PrivacyConstraint privacyConstraint = new PrivacyConstraint(privacyLevel);
		privacyConstraint.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0}, new long[]{4,4}), PrivacyConstraint.PrivacyLevel.Private);
		MatrixCharacteristics aCharacteristics = new MatrixCharacteristics(m, n, k, k);
		MatrixCharacteristics bCharacteristics = new MatrixCharacteristics(n, k, k, k);

		writeInputMatrixWithMTD("A", a, false, aCharacteristics, privacyConstraint);
		writeInputMatrixWithMTD("B", b, false, bCharacteristics);

		programArgs = new String[]{"-lineage", "reuse_full", "-nvargs",
			"A=" + input("A"), "B=" + input("B"), "C=" + output("C")};

		runTest(true,false,null,-1);

		finegrainedAssertions();

		programArgs = new String[]{"-nvargs",
			"A=" + input("A"), "B=" + input("B"), "C=" + output("C")};
		runTest(true,false,null,-1);

		finegrainedAssertions();
	}

	private void finegrainedAssertions(){
		String outputFineGrained = readDMLMetaDataValueCatchException("C", OUTPUT_DIR, DataExpression.FINE_GRAINED_PRIVACY);
		Assert.assertEquals(
			"{\"Private\":[[[0,0],[0,19]],[[1,0],[1,19]],[[2,0],[2,19]],[[3,0],[3,19]],[[4,0],[4,19]]],\"PrivateAggregation\":[]}",
			outputFineGrained);
	}
}

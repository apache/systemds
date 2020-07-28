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
package org.apache.sysds.test.applications;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

public class EntityResolutionBinaryTest extends AutomatedTestBase {
	private final static String TEST_NAME = "EntityResolutionBinary";
	private final static String TEST_DIR = "applications/entity_resolution/binary/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_DIR, TEST_NAME);
	}

	@Test
	public void testParams1() {
		testScriptEndToEnd(1, 1);
	}
	@Test
	public void testParams2() {
		testScriptEndToEnd(1, 3);
	}
	@Test
	public void testParams3() {
		testScriptEndToEnd(3, 1);
	}
	@Test
	public void testParams4() {
		testScriptEndToEnd(5, 5);
	}

	public void testScriptEndToEnd(int numLshHashtables, int numLshHyperplanes) {
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
		fullDMLScriptName = "./scripts/staging/entity-resolution/binary-entity-resolution.dml";;

		programArgs = new String[]{
				"-nvargs", //
				"FX=" + sourceDirectory + "input.csv", //
				"FY=" + sourceDirectory + "input.csv", //
				"OUT=" + output("B"), //
				"num_hashtables=" + numLshHashtables,
				"num_hyperplanes=" + numLshHyperplanes,
		};

		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

		// LSH is not deterministic, so in this test we just assert that it runs and produces a file
		Assert.assertTrue(Files.exists(Paths.get(output("B"))));
	}
}

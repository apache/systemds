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

import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;
import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.test.TestConfiguration;

public class EnumeratorTest extends AutomatedTestBase {

	private final static String TEST_NAME = "enumerator";
	private final static String TEST_DIR = "pipelines/";
	private static final String TEST_CLASS_DIR = TEST_DIR + EnumeratorTest.class.getSimpleName() + "/";
	protected static final String SCRIPT_DIR = "./scripts/staging/";
	private final static String logicalFile = SCRIPT_DIR+TEST_DIR + "logical.csv";
	private final static String outlierPrimitives = SCRIPT_DIR+TEST_DIR + "outlierPrimitives.csv";
	private final static String mviPrimitives = SCRIPT_DIR+TEST_DIR + "mviPrimitives.csv";
	private final static String parameters = SCRIPT_DIR+TEST_DIR + "properties.csv";
	private final static String DATASET = SCRIPT_DIR+TEST_DIR + "airbnb.csv";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testEnumerator(){runEnumerator(Types.ExecMode.SINGLE_NODE);};


	private void runEnumerator(Types.ExecMode instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args", DATASET, logicalFile, outlierPrimitives, mviPrimitives, parameters, output("A")};

			runTest(true, false, null, -1);

		}
		finally {
			rtplatform = platformOld;
		}
	}
}

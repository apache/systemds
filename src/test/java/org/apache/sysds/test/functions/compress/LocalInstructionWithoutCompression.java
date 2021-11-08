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

package org.apache.sysds.test.functions.compress;

import java.io.File;

import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class LocalInstructionWithoutCompression extends LocalInstruction {
	private final static String TEST_NAME1 = "local";
	private final static String TEST_DIR = "functions/compress/local/";
	private final static String TEST_CLASS_DIR = TEST_DIR + LocalInstructionWithoutCompression.class.getSimpleName()
		+ "/";

	public LocalInstructionWithoutCompression() {
		super();
	}

	@Override
	public void setUp() {
		final String dir = TEST_CLASS_DIR + "/local/";
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(dir, TEST_NAME1, new String[] {"B"}));
	}

	@Override
	protected String getTestDir() {
		return TEST_DIR;
	}

	@Test
	public void tests_01() {
		run(TEST_NAME1, 0, 1);
	}

	@Test
	public void tests_02() {
		run(TEST_NAME1, 0);
	}

	@Override
	protected File getConfigTemplateFile() {
		return new File(SCRIPT_DIR + TEST_DIR, "SystemDS-no-compress-config.xml");
	}
}

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

package org.apache.sysds.test.functions.compress.configuration;

import java.io.File;

public class CompressLossyCost extends CompressCost {

	public String TEST_NAME = "compress";
	public String TEST_DIR = "functions/compress/cost";
	public String TEST_CLASS_DIR = TEST_DIR + CompressLossyCost.class.getSimpleName() + "/";
	private String TEST_CONF = "SystemDS-config-compress-cost-lossy.xml";
	private File TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);

	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}

	protected String getTestName() {
		return TEST_NAME;
	}

	protected String getTestDir() {
		return TEST_DIR;
	}

	/**
	 * Override default configuration with custom test configuration to ensure scratch space and local temporary
	 * directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		return TEST_CONF_FILE;
	}
}

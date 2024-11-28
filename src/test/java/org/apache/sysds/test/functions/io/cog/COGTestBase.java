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

package org.apache.sysds.test.functions.io.cog;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public abstract class COGTestBase extends AutomatedTestBase {
	protected final static String TEST_DIR = "functions/io/cog/";
	protected static final Log LOG = LogFactory.getLog(COGTestBase.class.getName());
	protected final static double eps = 1e-6;

	protected abstract String getTestClassDir();

	protected abstract String getTestName();

	protected abstract int getScriptId();

	@Override
	public void setUp() {
		addTestConfiguration(getTestName(),
				new TestConfiguration(getTestClassDir(), getTestName(), new String[] {"Rout"}));
	}
}

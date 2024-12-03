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

package org.apache.sysds.test.applications.nn;

import static org.apache.sysds.api.mlcontext.ScriptFactory.dmlFromFile;
import static org.junit.Assert.assertTrue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.mlcontext.Script;
import org.apache.sysds.test.functions.mlcontext.MLContextTestBase;

public abstract class BaseTest extends MLContextTestBase {
	protected static final Log LOG = LogFactory.getLog(BaseTest.class.getName());

	private static final String ERROR_STRING = "ERROR:";

	public BaseTest() {
		//disable debug and trace logging in mlcontext super class since
		//the nn tests execute a lot of mini-batch operations
		_enableTracing = false;
	}

	protected void run(String name) {
		run(name, false);
	}

	protected void run(String name, boolean printStdOut) {
		Script script = dmlFromFile(getBaseFilePath() + name);
		String stdOut = executeAndCaptureStdOut(script).getRight();
		if(printStdOut){
			LOG.error(stdOut);
		}
		assertTrue(stdOut, !stdOut.contains(ERROR_STRING));
	}

	protected void run(String name, String[] var, Object[] val) {
		Script script = dmlFromFile(getBaseFilePath() + name);
		for(int i = 0; i < var.length; i++)
			script.in(var[i], val[i]);
		String stdOut = executeAndCaptureStdOut(script).getRight();
		assertTrue(stdOut, !stdOut.contains(ERROR_STRING));
	}

	protected abstract String getBaseFilePath();
}

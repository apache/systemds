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

import org.apache.sysds.api.mlcontext.Script;
import org.junit.Test;

public class NNOptimizerMNISTTest extends TestFolder {
	@Test
	public void mnist_optimizer_test() {
		this.inject_optimizer_adapter_module_and_run("sgd");
		this.inject_optimizer_adapter_module_and_run("adam");
	}

	private void inject_optimizer_adapter_module_and_run(String optimizer) {
		Script script = dmlFromFile(getBaseFilePath() + "component/optim/mnist_optimizer_check.dml");
		String moduleImportStatement = String.format("source(\"src/test/scripts/applications/nn/component/optim/adapters/%s.dml\") as optimizer", optimizer);
		String newScriptString = script.getScriptString().replace("# INSERT ADAPTER-MODULE #", moduleImportStatement);
		script.setScriptString(newScriptString);
		String stdOut = executeAndCaptureStdOut(script).getRight();
		LOG.info(stdOut);
		assertTrue(stdOut, !stdOut.contains(BaseTest.ERROR_STRING));
	}
}

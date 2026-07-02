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

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.api.mlcontext.Script;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class NNOptimizerMNISTTest extends TestFolder {
  /*
   * TODO: add instruction how to add optimizer + architecture
   */

	// region: parameters

	private final String optimizer;

	public NNOptimizerMNISTTest(String optimizer) {
		this.optimizer = optimizer;
	}

	@Parameters
	public static Collection<String> data() {
		return Arrays.asList("adagrad", "adam", "adamw", "lars", "rmsprop", "sgd", "sgd_momentum", "sgd_nesterov");
  }

	// endregion

  @Test
	public void mnist_optimizer_test() {
		if (this.optimizer != null)
			this.inject_optimizer_adapter_module_and_run(this.optimizer);
	}

	private void inject_optimizer_adapter_module_and_run(String optimizer) {
		Script script = dmlFromFile(getBaseFilePath() + "component/optim/mnist_optimizer_check.dml");
		String moduleImportStatement = String.format("source(\"src/test/scripts/applications/nn/component/optim/adapters/%s.dml\") as optimizer", optimizer);
		String newScriptString = script.getScriptString().replace("# INSERT ADAPTER-MODULE #", moduleImportStatement);
		script.setScriptString(newScriptString);
		String stdOut = executeAndCaptureStdOut(script).getRight();
		assertTrue(stdOut, !stdOut.contains(BaseTest.ERROR_STRING));
	}
}

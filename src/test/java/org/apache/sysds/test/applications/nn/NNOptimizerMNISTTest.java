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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.mlcontext.Script;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

// This test runs multiple epochs on a 1 hidden layer neural net
// while verifying an increasing accuracy and decreasing loss per epoch.
@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class NNOptimizerMNISTTest extends TestFolder {
  /*
   * To add new optimizer to this test, add an
   * adapter to "src/test/scripts/applications/nn/component/optim/adapters/" 
   * and add it to the parameter Collection. If needed, adjust the 
   * current function interface or make variables adjustable via parameter.
   */

	// region: parameters

	private final String optimizer;
	private final List<Pair<String, Object>> scriptArgs;

	public NNOptimizerMNISTTest(String optimizer, List<Pair<String, Object>> scriptArgs) {
		this.optimizer = optimizer;
		this.scriptArgs = scriptArgs;
	}

	@Parameters(name = "{0}")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{"adagrad", args()},
			{"adam", args()},
			{"adamw", args()},
			{"lars", args("$lr", 0.1)},
			{"rmsprop", args()},
			{"sgd", args()},
			{"sgd_momentum", args()},
			{"sgd_nesterov", args()}
		});
  }

	private static List<Pair<String, Object>> args(Object... args) {
		if(args.length % 2 != 0)
			throw new IllegalArgumentException("args must be given as name/value pairs.");

		List<Pair<String, Object>> pairs = new ArrayList<>(args.length / 2);
		for(int i = 0; i < args.length; i += 2) {
			if(!(args[i] instanceof String))
				throw new IllegalArgumentException("argnames must be strings.");
			pairs.add(Pair.of((String) args[i], args[i + 1]));
		}
		return pairs;
	}

	// endregion

  @Test
	public void mnist_optimizer_test() {
		this.inject_optimizer_adapter_module_and_run(this.optimizer, this.scriptArgs);
	}

  // injects the adapter from "src/test/scripts/applications/nn/component/optim/adapters/"
  // and executes the script while looking out for errors.
	private void inject_optimizer_adapter_module_and_run(String optimizer, List<Pair<String, Object>> scriptArgs) {
		Script script = dmlFromFile(getBaseFilePath() + "component/optim/mnist_optimizer_check.dml");
		String moduleImportStatement = String.format("source(\"src/test/scripts/applications/nn/component/optim/adapters/%s.dml\") as optimizer", optimizer);
		String newScriptString = script.getScriptString().replaceFirst("(?m)^.*# INSERT ADAPTER-MODULE #.*$", moduleImportStatement);
		script.setScriptString(newScriptString);
		for(Pair<String, Object> arg : scriptArgs)
			script.in(arg.getLeft(), arg.getRight());
		String stdOut = executeAndCaptureStdOut(script).getRight();
		assertTrue(stdOut, !stdOut.contains(BaseTest.ERROR_STRING));
	}
}

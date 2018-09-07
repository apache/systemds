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

package org.apache.sysml.test.integration.mlcontext.algorithms;

import static org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromFile;

import org.apache.log4j.Logger;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.test.integration.mlcontext.MLContextTestBase;
import org.junit.Test;

public class MLContextLinregTest extends MLContextTestBase {
	protected static Logger log = Logger.getLogger(MLContextLinregTest.class);

	protected final static String TEST_SCRIPT_CG = "scripts/algorithms/LinearRegCG.dml";
	protected final static String TEST_SCRIPT_DS = "scripts/algorithms/LinearRegDS.dml";
	private final static double sparsity1 = 0.7; // dense
	private final static double sparsity2 = 0.1; // sparse

	public enum LinregType {
		CG, DS,
	}

	@Test
	public void testLinregCGSparse() {
		runLinregTestMLC(LinregType.CG, true);
	}

	@Test
	public void testLinregCGDense() {
		runLinregTestMLC(LinregType.CG, false);
	}

	@Test
	public void testLinregDSSparse() {
		runLinregTestMLC(LinregType.DS, true);
	}

	@Test
	public void testLinregDSDense() {
		runLinregTestMLC(LinregType.DS, false);
	}

	private void runLinregTestMLC(LinregType type, boolean sparse) {

		double[][] X = getRandomMatrix(10, 3, 0, 1, sparse ? sparsity2 : sparsity1, 7);
		double[][] Y = getRandomMatrix(10, 1, 0, 10, 1.0, 3);

		switch (type) {
		case CG:
			Script lrcg = dmlFromFile(TEST_SCRIPT_CG);
			lrcg.in("X", X).in("y", Y).in("$icpt", "0").in("$tol", "0.000001").in("$maxi", "0").in("$reg", "0.000001")
					.out("beta_out");
			ml.execute(lrcg);

			break;

		case DS:
			Script lrds = dmlFromFile(TEST_SCRIPT_DS);
			lrds.in("X", X).in("y", Y).in("$icpt", "0").in("$reg", "0.000001").out("beta_out");
			ml.execute(lrds);

			break;
		}
	}
}

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
package org.apache.sysds.test.functions.federated.io;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.apache.sysds.api.DMLOptions;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedSparsityPropagationTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/io/";
	private final static String TEST_NAME = "FederatedSparsityPropagationTest";
	private final static int NUM_MATRICES = 15;
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedSparsityPropagationTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean rowPartitioned;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// number of rows or cols has to be >= number of federated workers.
		return Arrays.asList(new Object[][] {{100, 130, true}});
	}

	@Test
	public void federatedGetSparseSingleNode() {
		federatedGet(ExecMode.SINGLE_NODE, 0.01);
	}

	@Test
	public void federatedGetDenseSingleNode() {
		federatedGet(ExecMode.SINGLE_NODE, 0.5);
	}

	public void federatedGet(ExecMode execMode, double sparsity) {
		ExecMode platform_old = setExecMode(execMode);
		String HOME = SCRIPT_DIR + TEST_DIR;
		getAndLoadTestConfiguration(TEST_NAME);

		// write input matrices
		int fed_rows = rows / 2;
		int fed_cols = cols;

		MatrixCharacteristics mc = new MatrixCharacteristics(fed_rows, fed_cols, blocksize, fed_rows * fed_cols);
		double[][] X1 = getRandomMatrix(fed_rows, fed_cols, 1, 3, sparsity, 3);
		double[][] X2 = getRandomMatrix(fed_rows, fed_cols, 1, 3, sparsity, 7);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread t2 = startLocalFedWorkerThread(port2);

		getAndLoadTestConfiguration(TEST_NAME);

		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-nvargs", "in_X1=" + input("X1"), "in_X2=" + input("X2"),
			"sparsity=" + Double.toString(sparsity), "out_Dir=" + expectedDir()};
		runTest(true, false, null, -1);

		Map<String, Long> refNNZ = getRefNNZ();

		// Obtain nnz from actual dml script with federated matrix
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		Map<String, String> argVals = new HashMap<>();
		argVals.put("$in_X1", TestUtils.federatedAddress(port1, input("X1")));
		argVals.put("$in_X2", TestUtils.federatedAddress(port2, input("X2")));
		argVals.put("$rows", Integer.toString(fed_rows));
		argVals.put("$cols", Integer.toString(fed_cols));
		argVals.put("$sparsity", Double.toString(sparsity));

		Map<String, Long> fedNNZ = null;
		try {
			fedNNZ = executeFedAndGetNNZ(fullDMLScriptName, argVals);
		} catch(IOException ioe) {
			DMLScript.errorPrint(ioe);
			Assert.fail("IOException when executing federated test script.");
		}

		System.out.println("RefNNZ: " + refNNZ);
		System.out.println("FedNNZ: " + fedNNZ);

		compareNNZ(refNNZ, fedNNZ);

		TestUtils.shutdownThreads(t1, t2);

		resetExecMode(platform_old);
	}

	// NOTE: the body of this function is copied from DMLScript.execute
	private Map<String, Long> executeFedAndGetNNZ(String dmlScriptPath, Map<String, String> argVals)
		throws IOException {
		String dmlScriptStr = "";
		String DML_FILE_PATH_ANTLR_PARSER = DMLOptions.defaultOptions.filePath;
		dmlScriptStr = DMLScript.readDMLScript(true, fullDMLScriptName);

		ParserWrapper parser = ParserFactory.createParser();
		DMLProgram prog = parser.parse(DML_FILE_PATH_ANTLR_PARSER, dmlScriptStr, argVals);

		DMLTranslator dmlt = new DMLTranslator(prog);
		dmlt.liveVariableAnalysis(prog);
		dmlt.validateParseTree(prog);
		dmlt.constructHops(prog);
		dmlt.constructLops(prog);
		Program rtprog = dmlt.getRuntimeProgram(prog, ConfigurationManager.getDMLConfig());
		ArrayList<ProgramBlock> progBlocks = rtprog.getProgramBlocks();

		ExecutionContext ec = ExecutionContextFactory.createContext(rtprog);

		// execute the first program block and obtain the nnz from the federation maps
		progBlocks.get(0).execute(ec);
		Map<String, Long> fedNNZ = getFedNNZ(ec);
		// no need to execute the remaining program blocks
		
		return fedNNZ;
	}

	private Map<String, Long> getRefNNZ() {
		Map<String, Long> refNNZ = new HashMap<>();
		for(int counter = 0; counter < NUM_MATRICES; counter++) {
			String varName = "NNZ_M" + Integer.toString(counter+1);
			refNNZ.put(varName, readDMLScalarFromExpectedDir(varName)
				.entrySet().stream().findAny().get().getValue().longValue());
		}
		return refNNZ;
	}

	private Map<String, Long> getFedNNZ(ExecutionContext ec) {
		Map<String, Long> fedNNZ = new HashMap<>();
		for(String varName : ec.getVariables().keySet()) {
			if(ec.isMatrixObject(varName)) {
				MatrixObject mo = ec.getMatrixObject(varName);
				fedNNZ.put("NNZ_" + varName, mo.getNnz());
			}
		}
		return fedNNZ;
	}

	private void compareNNZ(Map<String, Long> ref, Map<String, Long> fed) {
		for(Map.Entry<String, Long> re : ref.entrySet()) {
			Assert.assertEquals("NNZs of " + re.getKey() + " differ.", re.getValue(), fed.get(re.getKey()));
		}
	}
}

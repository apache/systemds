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

package org.apache.sysds.test.component.resource;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.resource.CloudUtils;
import org.apache.sysds.resource.ResourceCompiler;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.utils.Explain;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.apache.sysds.resource.CloudUtils.getEffectiveExecutorResources;
import static org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext.SparkClusterConfig.RESERVED_SYSTEM_MEMORY_BYTES;

public class RecompilationTest extends AutomatedTestBase {
	static {
		ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.RESOURCE_OPTIMIZATION, true);
	}
	private static final boolean DEBUG_MODE = false;
	private static final String TEST_DIR = "component/resource/";
	private static final String TEST_DATA_DIR = "component/resource/data/";
	private static final String HOME = SCRIPT_DIR + TEST_DIR;
	private static final String HOME_DATA = SCRIPT_DIR + TEST_DATA_DIR;
	// Static Configuration values -------------------------------------------------------------------------------------
	private static final int driverThreads = 4;
	private static final int executorThreads = 2;

	@Override
	public void setUp() {}

	// Tests for setting cluster configurations ------------------------------------------------------------------------

	@Test
	public void testSetSingleNodeResourceConfigs() {
		long nodeMemory = 1024*1024*1024; // 1GB
		long expectedMemory = (long) (0.9 * nodeMemory);
		int expectedThreads = 4;

		ResourceCompiler.setSingleNodeResourceConfigs(nodeMemory, expectedThreads);

		Assert.assertEquals(expectedMemory, InfrastructureAnalyzer.getLocalMaxMemory());
		Assert.assertEquals(expectedThreads, InfrastructureAnalyzer.getLocalParallelism());
	}

	@Test
	public void testSetSparkClusterResourceConfigs() {
		long driverMemory = 4L*1024 * 1024 * 1024; // 1GB
		int driverThreads = 4;
		int numberExecutors = 10;
		long executorMemory = 1024 * 1024 * 1024; // 1GB
		int executorThreads = 4;

		ResourceCompiler.setSparkClusterResourceConfigs(driverMemory, driverThreads, numberExecutors, executorMemory, executorThreads);

		long expectedDriverMemory = CloudUtils.calculateEffectiveDriverMemoryBudget(driverMemory, executorThreads*numberExecutors);
		int[] expectedExecutorValues = getEffectiveExecutorResources(executorMemory, executorThreads, numberExecutors);
		int expectedNumExecutors = expectedExecutorValues[2];
		long expectedExecutorMemoryBudget = (long) (0.6 * (
				expectedNumExecutors * (expectedExecutorValues[0] * 1024 * 1024L - RESERVED_SYSTEM_MEMORY_BYTES)));
		int expectedParallelism = expectedExecutorValues[1] * expectedExecutorValues[2];
		Assert.assertEquals(expectedDriverMemory, InfrastructureAnalyzer.getLocalMaxMemory());
		Assert.assertEquals(driverThreads, InfrastructureAnalyzer.getLocalParallelism());
		Assert.assertEquals(expectedExecutorValues[2], SparkExecutionContext.getNumExecutors());
		Assert.assertEquals(expectedExecutorMemoryBudget, (long) SparkExecutionContext.getDataMemoryBudget(false, false));
		Assert.assertEquals(expectedParallelism, SparkExecutionContext.getDefaultParallelism(false));
	}

	// Tests for regular matrix multiplication (X%*%Y) -----------------------------------------------------------------

	@Test
	public void test_CP_MM_Enforced() throws IOException {
		// Single node cluster with 8GB driver memory -> ba+* operator
		// X = A.csv: (10^5)x(10^4) = 10^9 ~ 8BG
		// Y = B.csv: (10^4)x(10^3) = 10^7 ~ 80MB
		// X %*% Y -> (10^5)x(10^3) = 10^8 ~ 800MB
		runTestMM("A.csv", "B.csv", 8L*1024*1024*1024, 0, 0, Opcodes.MMULT.toString(), false);
	}

	@Test
	public void test_CP_MM_Preferred() throws IOException {
		// Distributed cluster with 16GB driver memory (large enough to host the computation) and any executors
		// X = A.csv: (10^5)x(10^4) = 10^9 ~ 8BG
		// Y = B.csv: (10^4)x(10^3) = 10^7 ~ 80MB
		// X %*% Y -> (10^5)x(10^3) = 10^8 ~ 800MB
		runTestMM("A.csv", "B.csv", 16L*1024*1024*1024, 2, 1024*1024*1024, Opcodes.MMULT.toString(), false);
	}

	@Test
	public void test_SP_MAPMM() throws IOException {
		// Distributed cluster with 4GB driver memory and 4GB executors -> mapmm operator
		// X = A.csv: (10^5)x(10^4) = 10^9 ~ 8BG
		// Y = B.csv: (10^4)x(10^3) = 10^7 ~ 80MB
		// X %*% Y -> (10^5)x(10^3) = 10^8 ~ 800MB
		runTestMM("A.csv", "B.csv", 4L*1024*1024*1024, 2, 4L*1024*1024*1024, Opcodes.MAPMM.toString(), true);
	}

	@Test
	public void test_SP_RMM() throws IOException {
		// Distributed cluster with 1GB driver memory and 500MB executors -> rmm operator
		// X = A.csv: (10^5)x(10^4) = 10^9 ~ 8BG
		// Y = B.csv: (10^4)x(10^3) = 10^7 ~ 80MB
		// X %*% Y -> (10^5)x(10^3) = 10^8 ~ 800MB
		runTestMM("A.csv", "B.csv", 4L*1024*1024*1024, 4, 1024*1024*1024, Opcodes.RMM.toString(), true);
	}

	@Test
	public void test_SP_CPMM() throws IOException {
		// Distributed cluster with 8GB driver memory and 4GB executors -> cpmm operator
		// X = A.csv: (10^5)x(10^4) = 10^9 ~ 8BG
		// Y = C.csv: (10^4)x(10^4) = 10^8 ~ 800MB
		// X %*% Y -> (10^5)x(10^4) = 10^9 ~ 8GB
		runTestMM("A.csv", "C.csv", 8L*1024*1024*1024, 2, 4L*1024*1024*1024, Opcodes.CPMM.toString(), true);
	}

	// Tests for transposed self matrix multiplication (t(X)%*%X) ------------------------------------------------------

	@Test
	public void test_CP_TSMM() throws IOException {
		// Single node cluster with 8GB driver memory -> tsmm operator in CP
		// X = B.csv: (10^4)x(10^3) = 10^7 ~ 80MB
		// t(X) %*% X -> (10^3)x(10^3) = 10^6 ~ 8MB (single block)
		runTestTSMM("B.csv", 8L*1024*1024*1024, 0, -1, Opcodes.TSMM.toString(), false);
	}

	@Test
	public void test_SP_TSMM_as_CPMM() throws IOException {
		// Distributed cluster with 8GB driver memory and 8GB executor memory -> cpmm operator in Spark
		// X = A.csv: (10^5)x(10^4) = 10^9 ~ 8GB
		// t(X) %*% X -> (10^4)x(10^4) = 10^8 ~ 800MB
		runTestTSMM("A.csv", 8L*1024*1024*1024, 2, 8L*1024*1024*1024, Opcodes.CPMM.toString(), true);
	}

	@Test
	public void test_MM_RecompilationSequence() throws IOException {
		runTestMM("A.csv", "B.csv", 8L*1024*1024*1024, 0, -1, Opcodes.MMULT.toString(), false);

		runTestMM("A.csv", "B.csv", 16L*1024*1024*1024, 4, 1024*1024*1024, Opcodes.MMULT.toString(), false);

		runTestMM("A.csv", "B.csv", 4L*1024*1024*1024, 2, 4L*1024*1024*1024, Opcodes.MAPMM.toString(), true);

		runTestMM("A.csv", "B.csv", 4L*1024*1024*1024, 4, 1024*1024*1024, Opcodes.RMM.toString(), true);

		runTestMM("A.csv", "B.csv", 8L*1024*1024*1024, 0, -1, Opcodes.MMULT.toString(), false);
	}

	@Test
	public void test_L2SVM() throws IOException {
		runTestAlgorithm("Algorithm_L2SVM.dml", 8L*1024*1024*1024, 0, -1);
		runTestAlgorithm("Algorithm_L2SVM.dml", 8L*1024*1024*1024, 4, 4L*1024*1024*1024);
	}

	@Test
	public void test_LinReg() throws IOException {
		runTestAlgorithm("Algorithm_Linreg.dml", 8L*1024*1024*1024, 0, -1);
		runTestAlgorithm("Algorithm_Linreg.dml", 8L*1024*1024*1024, 4, 4L*1024*1024*1024);
	}

	@Test
	public void test_PCA() throws IOException {
		runTestAlgorithm("Algorithm_PCA.dml", 8L*1024*1024*1024, 0, -1);
		runTestAlgorithm("Algorithm_PCA.dml", 8L*1024*1024*1024, 4, 8L*1024*1024*1024);
	}

	@Test
	public void test_PCA_Hybrid() throws IOException {
		HashMap<String, String> nvargs = new HashMap<>();
		nvargs.put("$m", "10000000");
		nvargs.put("$n", "100000");
		runTestAlgorithm("Algorithm_PCA.dml", 16L*1024*1024*1024, 4, 16L*1024*1024*1024, nvargs);
	}

	@Test
	public void test_PNMF() throws IOException {
		runTestAlgorithm("Algorithm_PNMF.dml", 8L*1024*1024*1024, 0, -1);
		runTestAlgorithm("Algorithm_PNMF.dml", 8L*1024*1024*1024, 4, 4L*1024*1024*1024);
	}

	@Test
	public void test_PNMF_Hybrid() throws IOException {
		HashMap<String, String> nvargs = new HashMap<>();
		nvargs.put("$m", "10000000");
		nvargs.put("$n", "10000");
		runTestAlgorithm("Algorithm_PNMF.dml", 8L*1024*1024*1024, 4, 4L*1024*1024*1024, nvargs);
	}

	// Helper functions ------------------------------------------------------------------------------------------------
	private Program generateInitialProgram(String filePath, Map<String, String> args) throws IOException {
		ResourceCompiler.setSparkClusterResourceConfigs(4L*1024*1024*1024, 4, ResourceCompiler.DEFAULT_NUMBER_EXECUTORS, ResourceCompiler.DEFAULT_EXECUTOR_MEMORY, ResourceCompiler.DEFAULT_EXECUTOR_THREADS);
		return  ResourceCompiler.compile(filePath, args);
	}

	private void runTestMM(String fileX, String fileY, long driverMemory, int numberExecutors, long executorMemory, String expectedOpcode, boolean expectedSparkExecType) throws IOException {
		Map<String, String> nvargs = new HashMap<>();
		nvargs.put("$X", HOME_DATA+fileX);
		nvargs.put("$Y", HOME_DATA+fileY);

		// pre-compiled program using default values to be used as source for the recompilation
		Program precompiledProgram = generateInitialProgram(HOME+"mm_test.dml", nvargs);

		if (numberExecutors > 0) {
			ResourceCompiler.setSparkClusterResourceConfigs(driverMemory, driverThreads, numberExecutors, executorMemory, executorThreads);
		} else {
			ResourceCompiler.setSingleNodeResourceConfigs(driverMemory, driverThreads);
		}

		// original compilation used for comparison
		Program expectedProgram = ResourceCompiler.compile(HOME+"mm_test.dml", nvargs);
		Program recompiledProgram = runTest(precompiledProgram, expectedProgram, driverMemory, numberExecutors, executorMemory, new StringBuilder());

		Optional<Instruction> mmInstruction = ((BasicProgramBlock) recompiledProgram.getProgramBlocks().get(0)).getInstructions().stream()
				.filter(inst -> (Objects.equals(expectedSparkExecType, inst instanceof SPInstruction) && Objects.equals(inst.getOpcode(), expectedOpcode)))
				.findFirst();
		Assert.assertTrue(mmInstruction.isPresent());
	}

	private void runTestTSMM(String fileX, long driverMemory, int numberExecutors, long executorMemory, String expectedOpcode, boolean expectedSparkExecType) throws IOException {
		Map<String, String> nvargs = new HashMap<>();
		nvargs.put("$X", HOME_DATA+fileX);

		// pre-compiled program using default values to be used as source for the recompilation
		Program precompiledProgram = generateInitialProgram(HOME+"mm_transpose_test.dml", nvargs);

		if (numberExecutors > 0) {
			ResourceCompiler.setSparkClusterResourceConfigs(driverMemory, driverThreads, numberExecutors, executorMemory, executorThreads);
		} else {
			ResourceCompiler.setSingleNodeResourceConfigs(driverMemory, driverThreads);
		}
		// original compilation used for comparison
		Program expectedProgram = ResourceCompiler.compile(HOME+"mm_transpose_test.dml", nvargs);
		Program recompiledProgram = runTest(precompiledProgram, expectedProgram, driverMemory, numberExecutors, executorMemory, new StringBuilder());
		Optional<Instruction> mmInstruction = ((BasicProgramBlock) recompiledProgram.getProgramBlocks().get(0)).getInstructions().stream()
				.filter(inst -> (Objects.equals(expectedSparkExecType, inst instanceof SPInstruction) && Objects.equals(inst.getOpcode(), expectedOpcode)))
				.findFirst();
		Assert.assertTrue(mmInstruction.isPresent());
	}

	private void runTestAlgorithm(String dmlScript, long driverMemory, int numberExecutors, long executorMemory) throws IOException {
		Map<String, String> nvargs = new HashMap<>();
		runTestAlgorithm(dmlScript, driverMemory, numberExecutors, executorMemory, nvargs);
	}

	private void runTestAlgorithm(String dmlScript, long driverMemory, int numberExecutors, long executorMemory,
								  Map<String, String> nvargs) throws IOException {
		// pre-compiled program using default values to be used as source for the recompilation
		Program precompiledProgram = generateInitialProgram(HOME+dmlScript, nvargs);
		StringBuilder sb = new StringBuilder();
		sb.append("\n\nprecompiled\n");
		sb.append(Explain.explain(precompiledProgram));
		if (numberExecutors > 0) {
			ResourceCompiler.setSparkClusterResourceConfigs(driverMemory, driverThreads, numberExecutors, executorMemory, executorThreads);
		} else {
			ResourceCompiler.setSingleNodeResourceConfigs(driverMemory, driverThreads);
		}
		// original compilation used for comparison
		Program expectedProgram = ResourceCompiler.compile(HOME+dmlScript, nvargs);
		sb.append("\n\nexpected\n");
		sb.append(Explain.explain(expectedProgram));
		runTest(precompiledProgram, expectedProgram, driverMemory, numberExecutors, executorMemory, sb);
	}

	private Program runTest(Program precompiledProgram, Program expectedProgram, long driverMemory, int numberExecutors, long executorMemory, StringBuilder sb) {
		if (DEBUG_MODE) sb.append(Explain.explain(expectedProgram));
		Program recompiledProgram;
		if (numberExecutors == 0) {
			ResourceCompiler.setSingleNodeResourceConfigs(driverMemory, driverThreads);
			recompiledProgram = ResourceCompiler.doFullRecompilation(precompiledProgram);
		} else {
			ResourceCompiler.setSparkClusterResourceConfigs(
					driverMemory,
					driverThreads,
					numberExecutors,
					executorMemory,
					executorThreads
			);
			recompiledProgram = ResourceCompiler.doFullRecompilation(precompiledProgram);
		}
		sb.append("\n\nrecompiled\n");
		sb.append(Explain.explain(recompiledProgram));

		if (DEBUG_MODE) sb.append(Explain.explain(recompiledProgram));
		assertEqualPrograms(expectedProgram, recompiledProgram, sb);
		return recompiledProgram;
	}

	private void assertEqualPrograms(Program expected, Program actual, StringBuilder sb) {
		// strip empty blocks basic program blocks
		String expectedProgramExplained = stripGeneralAndReplaceRandoms(Explain.explain(expected));
		String actualProgramExplained = stripGeneralAndReplaceRandoms(Explain.explain(actual));
		Assert.assertEquals(sb.toString(), expectedProgramExplained, actualProgramExplained);
	}

	private String stripGeneralAndReplaceRandoms(String explainedProgram) {
		String[] lines = explainedProgram.split("\\n");
		StringBuilder strippedBuilder = new StringBuilder();

		LinkedList<String> replaceList = new LinkedList<>();
		Pattern patternUnique = Pattern.compile("(_Var|_mVar|_sbcvar)(\\d+)");

		for (String line : lines) {
			String pureLine = line.replaceFirst("^-*", "");
			if (pureLine.startsWith("PROGRAM") || pureLine.startsWith("GENERIC") || pureLine.startsWith("CP rmvar")) {
				continue;
			} else if (pureLine.startsWith("CP mvvar") || pureLine.startsWith("CP cpvar")) {
				String[] parts = pureLine.split(" ");
				String lastPart = parts[parts.length - 1];
				if (!patternUnique.matcher(lastPart).matches()) {
					replaceList.add(lastPart);
				}
				continue;
			}
			if (pureLine.startsWith("CP") || pureLine.startsWith("SPARK")) {
				line = line.replaceFirst("\\b/temp\\d+\\b", "/tempX");
				Matcher matcherUnique = patternUnique.matcher(line);
				StringBuilder newLine = new StringBuilder();
				while (matcherUnique.find()) {
					matcherUnique.appendReplacement(newLine, "testVar");
				}
				matcherUnique.appendTail(newLine);
				line = newLine.toString();
			} else if (pureLine.startsWith("FUNCTION")) {
				line = pureLine.replaceFirst("recompile=true", "recompile=false");
			}
			strippedBuilder.append(line).append("\n");
		}
		String strippedProgram = "\n" + strippedBuilder.toString().trim() + "\n";
		for (String literalVar : replaceList) {
			strippedProgram = strippedProgram.replaceAll("\\b "+literalVar+".\\b", " testVar.");
		}
		return strippedProgram;
	}
}

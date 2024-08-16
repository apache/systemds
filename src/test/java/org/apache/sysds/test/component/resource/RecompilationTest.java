package org.apache.sysds.test.component.resource;

import org.apache.sysds.resource.ResourceCompiler;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.utils.Explain;

import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

import static org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext.SparkClusterConfig.RESERVED_SYSTEM_MEMORY_BYTES;

public class RecompilationTest extends AutomatedTestBase {
    private static final boolean DEBUG_MODE = true;
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
    public void testSetDriverConfigurations() {
        long expectedMemory = 1024*1024*1024; // 1GB
        int expectedThreads = 4;

        ResourceCompiler.setDriverConfigurations(expectedMemory, expectedThreads);

        Assert.assertEquals(expectedMemory, InfrastructureAnalyzer.getLocalMaxMemory());
        Assert.assertEquals(expectedThreads, InfrastructureAnalyzer.getLocalParallelism());
    }

    @Test
    public void testSetExecutorConfigurations() {
        int numberExecutors = 10;
        long executorMemory = 1024*1024*1024; // 1GB
        long expectedMemoryBudget = (long) (numberExecutors*(executorMemory-RESERVED_SYSTEM_MEMORY_BYTES)*0.6);
        int executorThreads = 4;
        int expectedParallelism = numberExecutors*executorThreads;

        ResourceCompiler.setExecutorConfigurations(numberExecutors, executorMemory, executorThreads);

        Assert.assertEquals(numberExecutors, SparkExecutionContext.getNumExecutors());
        Assert.assertEquals(expectedMemoryBudget, (long) SparkExecutionContext.getDataMemoryBudget(false, false));
        Assert.assertEquals(expectedParallelism, SparkExecutionContext.getDefaultParallelism(false));
    }

    // Tests for regular matrix multiplication (X%*%Y) -----------------------------------------------------------------

    @Test
    public void test_CP_MM_Enforced() throws IOException {
        // Single node cluster with 8GB driver memory -> ba+* operator
        // X = A.csv: (10^5)x(10^4) = 10^9 ~ 8BG
        // Y = B.csv: (10^4)x(10^3) = 10^7 ~ 80MB
        // X %*% Y -> (10^5)x(10^3) = 10^8 ~ 800MB
        runTestMM("A.csv", "B.csv", 8L*1024*1024*1024, 0, -1, "ba+*");
    }

    @Test
    public void test_CP_MM_Preferred() throws IOException {
        // Distributed cluster with 16GB driver memory (large enough to host the computation) and any executors
        // X = A.csv: (10^5)x(10^4) = 10^9 ~ 8BG
        // Y = B.csv: (10^4)x(10^3) = 10^7 ~ 80MB
        // X %*% Y -> (10^5)x(10^3) = 10^8 ~ 800MB
        runTestMM("A.csv", "B.csv", 16L*1024*1024*1024, 2, 1024*1024*1024, "ba+*");
    }

    @Test
    public void test_SP_MAPMM() throws IOException {
        // Distributed cluster with 4GB driver memory and 4GB executors -> mapmm operator
        // X = A.csv: (10^5)x(10^4) = 10^9 ~ 8BG
        // Y = B.csv: (10^4)x(10^3) = 10^7 ~ 80MB
        // X %*% Y -> (10^5)x(10^3) = 10^8 ~ 800MB
        runTestMM("A.csv", "B.csv", 4L*1024*1024*1024, 2, 4L*1024*1024*1024, "mapmm");
    }

    @Test
    public void test_SP_RMM() throws IOException {
        // Distributed cluster with 1GB driver memory and 500MB executors -> rmm operator
        // X = A.csv: (10^5)x(10^4) = 10^9 ~ 8BG
        // Y = B.csv: (10^4)x(10^3) = 10^7 ~ 80MB
        // X %*% Y -> (10^5)x(10^3) = 10^8 ~ 800MB
        runTestMM("A.csv", "B.csv", 1024*1024*1024, 2, (long) (0.5*1024*1024*1024), "rmm");
    }

    @Test
    public void test_SP_CPMM() throws IOException {
        // Distributed cluster with 8GB driver memory and 4GB executors -> cpmm operator
        // X = A.csv: (10^5)x(10^4) = 10^9 ~ 8BG
        // Y = C.csv: (10^4)x(10^4) = 10^8 ~ 800MB
        // X %*% Y -> (10^5)x(10^4) = 10^9 ~ 8GB
        runTestMM("A.csv", "C.csv", 8L*1024*1024*1024, 2, 4L*1024*1024*1024, "cpmm");
    }

    // Tests for transposed self matrix multiplication (t(X)%*%X) ------------------------------------------------------

    @Test
    public void test_CP_TSMM() throws IOException {
        // Single node cluster with 8GB driver memory -> tsmm operator in CP
        // X = B.csv: (10^4)x(10^3) = 10^7 ~ 80MB
        // t(X) %*% X -> (10^3)x(10^3) = 10^6 ~ 8MB (single block)
        runTestTSMM("B.csv", 8L*1024*1024*1024, 0, -1, "tsmm", false);
    }

    @Test
    public void test_SP_TSMM() throws IOException {
        // Distributed cluster with 1GB driver memory and 8GB executor memory -> tsmm operator in Spark
        // X = D.csv: (10^5)x(10^3) = 10^8 ~ 800MB
        // t(X) %*% X -> (10^3)x(10^3) = 10^6 ~ 8MB (single block)
        runTestTSMM("D.csv", 1024*1024*1024, 2, 8L*1024*1024*1024, "tsmm", true);
    }

    @Test
    public void test_SP_TSMM_as_CPMM() throws IOException {
        // Distributed cluster with 8GB driver memory and 8GB executor memory -> cpmm operator in Spark
        // X = A.csv: (10^5)x(10^4) = 10^9 ~ 8GB
        // t(X) %*% X -> (10^4)x(10^4) = 10^8 ~ 800MB
        runTestTSMM("A.csv", 8L*1024*1024*1024, 2, 8L*1024*1024*1024, "cpmm", true);
    }

    @Test
    public void test_MM_RecompilationSequence() throws IOException {
        Map<String, String> nvargs = new HashMap<>();
        nvargs.put("$X", HOME_DATA+"A.csv");
        nvargs.put("$Y", HOME_DATA+"B.csv");

        // pre-compiled program using default values to be used as source for the recompilation
        Program precompiledProgram = generateInitialProgram(HOME+"mm_test.dml", nvargs);
        // original compilation used for comparison
        Program expectedProgram;

        ResourceCompiler.setDriverConfigurations(8L*1024*1024*1024, driverThreads);
        ResourceCompiler.setSingleNodeExecution();
        expectedProgram = ResourceCompiler.compile(HOME+"mm_test.dml", nvargs);
        runTest(precompiledProgram, expectedProgram, 8L*1024*1024*1024, 0, -1, "ba+*", false);

        ResourceCompiler.setDriverConfigurations(16L*1024*1024*1024, driverThreads);
        ResourceCompiler.setExecutorConfigurations(2, 1024*1024*1024, executorThreads);
        expectedProgram = ResourceCompiler.compile(HOME+"mm_test.dml", nvargs);
        runTest(precompiledProgram, expectedProgram, 16L*1024*1024*1024, 2, 1024*1024*1024, "ba+*", false);

        ResourceCompiler.setDriverConfigurations(4L*1024*1024*1024, driverThreads);
        ResourceCompiler.setExecutorConfigurations(2, 4L*1024*1024*1024, executorThreads);
        expectedProgram = ResourceCompiler.compile(HOME+"mm_test.dml", nvargs);
        runTest(precompiledProgram, expectedProgram, 4L*1024*1024*1024, 2, 4L*1024*1024*1024, "mapmm", true);

        ResourceCompiler.setDriverConfigurations(1024*1024*1024, driverThreads);
        ResourceCompiler.setExecutorConfigurations(2, (long) (0.5*1024*1024*1024), executorThreads);
        expectedProgram = ResourceCompiler.compile(HOME+"mm_test.dml", nvargs);
        runTest(precompiledProgram, expectedProgram, 1024*1024*1024, 2, (long) (0.5*1024*1024*1024), "rmm", true);

        ResourceCompiler.setDriverConfigurations(8L*1024*1024*1024, driverThreads);
        ResourceCompiler.setSingleNodeExecution();
        expectedProgram = ResourceCompiler.compile(HOME+"mm_test.dml", nvargs);
        runTest(precompiledProgram, expectedProgram, 8L*1024*1024*1024, 0, -1, "ba+*", false);
    }

    // Helper functions ------------------------------------------------------------------------------------------------
    private Program generateInitialProgram(String filePath, Map<String, String> args) throws IOException {
        ResourceCompiler.setDriverConfigurations(ResourceCompiler.DEFAULT_DRIVER_MEMORY, ResourceCompiler.DEFAULT_DRIVER_THREADS);
        ResourceCompiler.setExecutorConfigurations(ResourceCompiler.DEFAULT_NUMBER_EXECUTORS, ResourceCompiler.DEFAULT_EXECUTOR_MEMORY, ResourceCompiler.DEFAULT_EXECUTOR_THREADS);
        return  ResourceCompiler.compile(filePath, args);
    }

    private void runTestMM(String fileX, String fileY, long driverMemory, int numberExecutors, long executorMemory, String expectedOpcode) throws IOException {
        boolean expectedSparkExecType = !Objects.equals(expectedOpcode,"ba+*");
        Map<String, String> nvargs = new HashMap<>();
        nvargs.put("$X", HOME_DATA+fileX);
        nvargs.put("$Y", HOME_DATA+fileY);

        // pre-compiled program using default values to be used as source for the recompilation
        Program precompiledProgram = generateInitialProgram(HOME+"mm_test.dml", nvargs);

        ResourceCompiler.setDriverConfigurations(driverMemory, driverThreads);
        if (numberExecutors > 0) {
            ResourceCompiler.setExecutorConfigurations(numberExecutors, executorMemory, executorThreads);
        } else {
            ResourceCompiler.setSingleNodeExecution();
        }

        // original compilation used for comparison
        Program expectedProgram = ResourceCompiler.compile(HOME+"mm_test.dml", nvargs);
        runTest(precompiledProgram, expectedProgram, driverMemory, numberExecutors, executorMemory, expectedOpcode, expectedSparkExecType);
    }

    private void runTestTSMM(String fileX, long driverMemory, int numberExecutors, long executorMemory, String expectedOpcode, boolean expectedSparkExecType) throws IOException {
        Map<String, String> nvargs = new HashMap<>();
        nvargs.put("$X", HOME_DATA+fileX);

        // pre-compiled program using default values to be used as source for the recompilation
        Program precompiledProgram = generateInitialProgram(HOME+"mm_transpose_test.dml", nvargs);

        ResourceCompiler.setDriverConfigurations(driverMemory, driverThreads);
        if (numberExecutors > 0) {
            ResourceCompiler.setExecutorConfigurations(numberExecutors, executorMemory, executorThreads);
        } else {
            ResourceCompiler.setSingleNodeExecution();
        }
        // original compilation used for comparison
        Program expectedProgram = ResourceCompiler.compile(HOME+"mm_transpose_test.dml", nvargs);
        runTest(precompiledProgram, expectedProgram, driverMemory, numberExecutors, executorMemory, expectedOpcode, expectedSparkExecType);
    }

    private void runTest(Program precompiledProgram, Program expectedProgram, long driverMemory, int numberExecutors, long executorMemory, String expectedOpcode, boolean expectedSparkExecType) {
        String expectedProgramExplained = Explain.explain(expectedProgram);

        Program recompiledProgram;
        if (numberExecutors == 0) {
            recompiledProgram = ResourceCompiler.doFullRecompilation(precompiledProgram, driverMemory, driverThreads);
        } else {
            recompiledProgram = ResourceCompiler.doFullRecompilation(precompiledProgram, driverMemory, driverThreads, numberExecutors, executorMemory, executorThreads);
        }
        String actualProgramExplained = Explain.explain(recompiledProgram);

        if (DEBUG_MODE) System.out.println(actualProgramExplained);
        Assert.assertEquals(expectedProgramExplained, actualProgramExplained);
        Optional<Instruction> mmInstruction = ((BasicProgramBlock) recompiledProgram.getProgramBlocks().get(0)).getInstructions().stream()
                .filter(inst -> (Objects.equals(expectedSparkExecType, inst instanceof SPInstruction) && Objects.equals(inst.getOpcode(), expectedOpcode)))
                .findFirst();
        Assert.assertTrue(mmInstruction.isPresent());
    }
}

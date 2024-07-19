package org.apache.sysds.test.gpu.multigpu;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

public class SingleGPUTestDifferentBratch extends GPUTest {
    @Override
    public void setUp() {
        super.setUp();
    }

    @Test
    public void SingleGPUTest() {
        runMultiGPUsTest(false, 100000); // Example: 100k test images
    }

    @Test
    public void MultiGPUsTest() {
        runMultiGPUsTest(true, 100000); // Example: 100k test images
    }

    @Override
    public void gpuTest() {
        // Not used in this implementation
    }

    /**
     * Run the test with multiple GPUs
     *
     * @param multiGPUs whether to run the test with multiple GPUs
     * @param numTestImages the number of test images
     */
    protected void runMultiGPUsTest(boolean multiGPUs, int numTestImages) {
        getAndLoadTestConfiguration(multiGPUs ? MULTI_GPUS_TEST : SINGLE_GPU_TEST);

        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        programArgs = new String[] { "-args", DATA_SET, output("R"), Integer.toString(numTestImages), "-config",
                multiGPUs ? MULTI_TEST_CONFIG : SINGLE_TEST_CONFIG };
        fullRScriptName = HOME + TEST_NAME + ".R";

        rCmd = null;
        InMemoryAppender appender = configureLog4j();

        runTest(true, false, null, -1);

        List<String> logs = appender.getLogMessages();
        int numRealThread = 0;
        for (String log : logs) {
            if (log.contains("has executed") && extractNumTasks(log) > 0) {
                numRealThread ++;
            }
        }
        if (multiGPUs) {
            assertTrue(numRealThread > 1);
        } else {
            assertEquals(1, numRealThread);
        }

        appender.clearLogMessages();
    }
}

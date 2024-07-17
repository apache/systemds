package org.apache.sysds.test.gpu.multigpu;


import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

public class MultiGPUTest extends AutomatedTestBase {
    private static final String TEST_DIR = "gpu/";
    private static final String TEST_CLASS_DIR = TEST_DIR + MultiGPUTest.class.getSimpleName() +  "/";
    private static final String SINGLE_GPU_TEST = "SingleGPUTest";
    private static final String MULTI_GPUS_TEST = "MultiGPUsTest";

    @Override
    public void setUp() {
        TEST_GPU = false;
        VERBOSE_STATS = true;
        addTestConfiguration(SINGLE_GPU_TEST,
                new TestConfiguration(TEST_CLASS_DIR, "GPUTest", new String[]{"output"}));
        addTestConfiguration(MULTI_GPUS_TEST,
                new TestConfiguration(TEST_CLASS_DIR, "GPUTest", new String[]{"output"}));
        TestConfiguration singleConfig = availableTestConfigurations.get(SINGLE_GPU_TEST);
        singleConfig.addVariable("sysds.gpu.availableGPUs", 0);
    }

    @Test
    public void SingleGPUTest() {
        runMultiGPUsTest(true);
    }

    @Test
    public void MultiGPUsTest() {
        runMultiGPUsTest(true);
    }

    /**
     
Run the test with multiple GPUs
@param multiGPUs whether to run the test with multiple GPUs*/
private void runMultiGPUsTest(boolean multiGPUs) {
    getAndLoadTestConfiguration(multiGPUs ? MULTI_GPUS_TEST : SINGLE_GPU_TEST);

        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + "GPUTest" + ".dml";
        programArgs = new String[]{"-args", output("output") };
        fullRScriptName = HOME + "GPUTest" + ".R";
        rCmd = null;

        runTest(true, false, null, -1);
    }
}
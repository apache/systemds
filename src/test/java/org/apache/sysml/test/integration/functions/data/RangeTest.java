package org.apache.sysml.test.integration.functions.data;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

public class RangeTest extends AutomatedTestBase {

    private static final String TEST_DIR = "functions/data/";
    private static final String TEST_CLASS_DIR = TEST_DIR + RangeTest.class.getSimpleName() + "/";
    private static final String OUTPUT_NAME = "range";

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
    }

    @Test
    public void rangeTest() throws Exception {
        String testName = "Range1";
        String expectedOutput = "1.000\n" +
                "2.000\n" +
                "3.000\n" +
                "4.000\n" +
                "5.000\n";
        addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
        rangeTestHelper(DMLScript.RUNTIME_PLATFORM.SINGLE_NODE, testName, ScriptType.PYDML, expectedOutput);
    }

    @Test
    public void rangeWithTwoParams() throws Exception {
        String testName = "Range2";
        String expectedOutput = "1.000\n" +
                "2.000\n" +
                "3.000\n" +
                "4.000\n" +
                "5.000\n";
        addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
        rangeTestHelper(DMLScript.RUNTIME_PLATFORM.SINGLE_NODE, testName, ScriptType.PYDML, expectedOutput);

    }

    protected void rangeTestHelper(DMLScript.RUNTIME_PLATFORM platform, String testName, ScriptType scriptType, String expectedOutput) {
        DMLScript.RUNTIME_PLATFORM platformOld = rtplatform;

        rtplatform = platform;
        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        if (rtplatform == DMLScript.RUNTIME_PLATFORM.SPARK) {
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;
        }
        try {
            getAndLoadTestConfiguration(testName);

            fullDMLScriptName = SCRIPT_DIR + TEST_DIR + testName + "." + scriptType.toString().toLowerCase();
            programArgs = (scriptType == ScriptType.PYDML) ? new String[]{"-python", "-args", output(OUTPUT_NAME)} : new String[]{"-args", output(OUTPUT_NAME)};

            runTest(true, false, null, -1);

            String output = TestUtils.readDMLString(output(OUTPUT_NAME));
            TestUtils.compareScalars(expectedOutput, output);
        } finally {
            rtplatform = platformOld;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
        }
    }

}

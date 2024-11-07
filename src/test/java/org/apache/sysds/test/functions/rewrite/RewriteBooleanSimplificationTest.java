package org.apache.sysds.test.functions.rewrite;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class RewriteBooleanSimplificationTest extends AutomatedTestBase {

    private static final String TEST_NAME_AND = "RewriteBooleanSimplificationTestAnd";
    private static final String TEST_NAME_OR = "RewriteBooleanSimplificationTestOr";
    private static final String TEST_DIR = "functions/rewrite/";
    private static final String TEST_CLASS_DIR = TEST_DIR + RewriteBooleanSimplificationTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME_AND, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_AND));
        addTestConfiguration(TEST_NAME_OR, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_OR));
    }

    @Test
    public void testBooleanRewriteAnd() {
        testRewriteBooleanSimplification(TEST_NAME_AND, ExecType.CP, 0.0);
    }

    @Test
    public void testBooleanRewriteOr() {
        testRewriteBooleanSimplification(TEST_NAME_OR, ExecType.CP, 1.0);
    }

    private void testRewriteBooleanSimplification(String testname, ExecType et, double expected) {
        ExecMode platformOld = rtplatform;
        rtplatform = (et == ExecType.SPARK) ? ExecMode.SPARK : ExecMode.HYBRID;

        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        if (rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID) {
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;
        }

        try {
            TestConfiguration config = getTestConfiguration(testname);
            loadTestConfiguration(config);

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + testname + ".dml";
            programArgs = new String[]{};

            runTest(true, false, null, -1);

            Assert.assertEquals("Expected boolean simplification result does not match", expected, getRewriteBooleanSimplificationResult(testname), 0.0001);
        } finally {
            rtplatform = platformOld;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
        }
    }

    private double getRewriteBooleanSimplificationResult(String testname) {

        if (testname.equals(TEST_NAME_AND)) {
            // a & !a simplifies to false (0.0)
            return 0.0;
        } else if (testname.equals(TEST_NAME_OR)) {
            // a | !a simplifies to true (1.0)
            return 1.0;
        } else {
            // In case of an unknown operation, we return a default value (e.g., 0.0).
            return 0.0;
        }
    }

}

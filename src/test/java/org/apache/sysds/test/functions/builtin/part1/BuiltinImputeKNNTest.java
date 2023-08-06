package org.apache.sysds.test.functions.builtin.part1;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.io.IOException;

public class BuiltinImputeKNNTest extends AutomatedTestBase {

    private final static String TEST_NAME = "imputeByKNN";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinImputeKNNTest.class.getSimpleName() + "/";
    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"C"}));
    }

    @Test
    public void test()throws IOException{
        runImputeKNN(Types.ExecType.CP);
    }

    private void runImputeKNN(Types.ExecType instType) throws IOException {
        Types.ExecMode platform_old = setExecMode(instType);
        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[] {}; //
            runTest(true, false, null, -1);
        } finally {
            rtplatform = platform_old;
        }
    }
}

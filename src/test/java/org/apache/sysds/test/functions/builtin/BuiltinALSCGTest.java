package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class BuiltinALSCGTest extends AutomatedTestBase {

    private final static String TEST_NAME = "als_cg";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinALSCGTest.class.getSimpleName() + "/";

    private final static int rows = 300;
    private final static int cols = 20;

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
    }

    @Test
    public void testALSCG() {
        runtestALSCG();
    }

    private void runtestALSCG(){

        loadTestConfiguration(getTestConfiguration(TEST_NAME));
        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        List<String> proArgs = new ArrayList<>();

        proArgs.add("-explain");
        proArgs.add("-stats");
        proArgs.add("-args");
        proArgs.add(input("X"));
        proArgs.add(output("U"));
        proArgs.add(output("V"));
        programArgs = proArgs.toArray(new String[proArgs.size()]);

        double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
        writeInputMatrixWithMTD("X", X, true);

        runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

    }
}

package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class BuiltinALSDSTest extends AutomatedTestBase {

    private final static String TEST_NAME = "als_ds";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinALSDSTest.class.getSimpleName() + "/";

    private final static int rows = 300;
    private final static int cols = 20;

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
    }

    @Test
    public void testALSDS() {
        runtestALSDS();
    }

    private void runtestALSDS(){
        loadTestConfiguration(getTestConfiguration(TEST_NAME));
        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        List<String> proArgs = new ArrayList<>();

        proArgs.add("-explain");
        proArgs.add("-stats");
        proArgs.add("-args");
        proArgs.add(input("V"));
        proArgs.add(output("L"));
        proArgs.add(output("R"));
        programArgs = proArgs.toArray(new String[proArgs.size()]);

        double[][] V = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
        writeInputMatrixWithMTD("V", V, true);

        runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);




    }

}

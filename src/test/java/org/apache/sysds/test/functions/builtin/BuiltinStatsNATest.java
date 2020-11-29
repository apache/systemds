package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinStatsNATest extends AutomatedTestBase {
    private final static String TEST_NAME = "split";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinSplitTest.class.getSimpleName() + "/";
    private final static int rows = 10;
    private final static int cols = 10;

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B",}));
    }

    @Test
    public void testStatsNA() {runStatsNA(1, true);}

    @Test
    public void testStatsNA2() {runStatsNA(4, true);}

    @Test
    public void testStatsNA3() {runStatsNA(10, true);}

    @Test
    public void testStatsNA4() {runStatsNA(100, true);}

    @Test
    public void testStatsNAList() {runStatsNA(1, false);}
    @Test
    public void testStatsNA2List() {runStatsNA(4, false);}
    @Test
    public void testStatsNA3List() {runStatsNA(10, false);}
    @Test
    public void testStatsNA4List() {runStatsNA(100, false);}

    private void runStatsNA(int bins, boolean prints_only)
    {
        double[][] A = getRandomMatrix(rows, cols, -10, 10, 0.6, 7);
        writeInputMatrixWithMTD("A", A, true);
    }
}

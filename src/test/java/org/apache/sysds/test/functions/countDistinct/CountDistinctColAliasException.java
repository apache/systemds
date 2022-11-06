package org.apache.sysds.test.functions.countDistinct;

import org.apache.sysds.common.Types;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

public class CountDistinctColAliasException extends CountDistinctBase {

    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    private final static String TEST_NAME = "CountDistinctColAliasException";
    private final static String TEST_DIR = "functions/countDistinct/";
    private final static String TEST_CLASS_DIR = TEST_DIR + CountDistinctColAliasException.class.getSimpleName() + "/";

    private final Types.Direction DIRECTION = Types.Direction.Row;

    @Override
    protected String getTestClassDir() {
        return TEST_CLASS_DIR;
    }

    @Override
    protected String getTestName() {
        return TEST_NAME;
    }

    @Override
    protected String getTestDir() {
        return TEST_DIR;
    }

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName(), new String[] {"A"}));

        this.percentTolerance = 0.2;
    }

    @Test
    public void testCPSparseSmall() {
        exceptionRule.expect(AssertionError.class);
        exceptionRule.expectMessage("Invalid number of arguments for function col_count_distinct(). " +
                "This function only takes 1 or 2 arguments.");

        Types.ExecType execType = Types.ExecType.CP;

        int actualDistinctCount = 10;
        int rows = 1000, cols = 1000;
        double sparsity = 0.1;
        double tolerance = actualDistinctCount * this.percentTolerance;
        countDistinctMatrixTest(DIRECTION, actualDistinctCount, cols, rows, sparsity, execType, tolerance);
    }
}

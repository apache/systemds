package org.apache.sysds.test.functions.builtin.setoperations;

import org.apache.sysds.common.Types;

public class BuiltinSymmetricDifferenceTest extends SetOperationsTestBase {
    private final static String TEST_NAME = "symmetricDifference";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinSymmetricDifferenceTest.class.getSimpleName() + "/";

    public BuiltinSymmetricDifferenceTest(Types.ExecType execType) {
        super(TEST_NAME, TEST_DIR, TEST_CLASS_DIR, execType);
    }
}

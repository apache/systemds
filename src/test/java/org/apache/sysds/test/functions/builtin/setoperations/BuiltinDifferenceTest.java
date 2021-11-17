package org.apache.sysds.test.functions.builtin.setoperations;

import org.apache.sysds.common.Types;

import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;


@RunWith(Parameterized.class)
public class BuiltinDifferenceTest extends SetOperationsTestBase {
    private final static String TEST_NAME = "difference";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDifferenceTest.class.getSimpleName() + "/";

    public BuiltinDifferenceTest(Types.ExecType execType){
        super(TEST_NAME, TEST_DIR, TEST_CLASS_DIR, execType);

    }
}
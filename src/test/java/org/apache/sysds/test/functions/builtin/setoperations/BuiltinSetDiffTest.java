package org.apache.sysds.test.functions.builtin.setoperations;

import org.apache.sysds.common.Types;

import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;


@RunWith(Parameterized.class)
public class BuiltinSetDiffTest extends SetOperationsTestBase {
    private final static String TEST_NAME = "setdiff";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinSetDiffTest.class.getSimpleName() + "/";

    public BuiltinSetDiffTest(Types.ExecType execType){
        super(TEST_NAME, TEST_DIR, TEST_CLASS_DIR, execType);

    }
}
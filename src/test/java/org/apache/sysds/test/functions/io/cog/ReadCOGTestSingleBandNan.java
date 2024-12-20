package org.apache.sysds.test.functions.io.cog;

public class ReadCOGTestSingleBandNan extends ReadCOGTest {
    private final static String TEST_NAME = "ReadCOGTest";
    public final static String TEST_CLASS_DIR = TEST_DIR + ReadCOGTestSingleBandNan.class.getSimpleName() + "/";

    protected String getTestName() {
        return TEST_NAME;
    }

    protected String getTestClassDir() {
        return TEST_CLASS_DIR;
    }

    protected int getScriptId() {
        return 2;
    }

    protected double getResult(){ return 31021228.0; }

    protected int getId() {
        return 3;
    }
}

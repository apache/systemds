package org.apache.sysds.test.functions.io.cog;

public class ReadCOGTestMultipleBandsByte extends ReadCOGTest {
    private final static String TEST_NAME = "ReadCOGTest";
    public final static String TEST_CLASS_DIR = TEST_DIR + ReadCOGTestMultipleBandsByte.class.getSimpleName() + "/";

    protected String getTestName() {
        return TEST_NAME;
    }

    protected String getTestClassDir() {
        return TEST_CLASS_DIR;
    }

    protected double getResult(){ return 1552412.0; }

    protected int getScriptId() {
        return 1;
    }

    protected int getId() {
        return 2;
    }
}
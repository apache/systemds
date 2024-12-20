package org.apache.sysds.test.functions.io.cog;

public class ReadCOGTestSingleBandTileOnly extends ReadCOGTest {
    private final static String TEST_NAME = "ReadCOGTest";
    public final static String TEST_CLASS_DIR = TEST_DIR + ReadCOGTestSingleBandTileOnly.class.getSimpleName() + "/";

    protected String getTestName() {
        return TEST_NAME;
    }

    protected String getTestClassDir() {
        return TEST_CLASS_DIR;
    }

    protected int getScriptId() {
        return 3;
    }

    protected double getResult(){ return 323597181.0; }

    protected int getId() {
        return 1;
    }
}
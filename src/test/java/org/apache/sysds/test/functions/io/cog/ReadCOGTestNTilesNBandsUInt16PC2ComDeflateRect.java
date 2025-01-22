package org.apache.sysds.test.functions.io.cog;

public class ReadCOGTestNTilesNBandsUInt16PC2ComDeflateRect extends ReadCOGTest {
    private final static String TEST_NAME = "ReadCOGTest";
    public final static String TEST_CLASS_DIR = TEST_DIR + ReadCOGTestNTilesNBandsUInt16PC2ComDeflateRect.class.getSimpleName() + "/";

    protected String getTestName() {
        return TEST_NAME;
    }

    protected String getTestClassDir() {
        return TEST_CLASS_DIR;
    }

    protected int getScriptId() {
        return 1;
    }

    protected double getResult(){ return 199571808.0; }

    protected int getId() {
        return 9;
    }
}

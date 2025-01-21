package org.apache.sysds.test.functions.io.cog;

public class ReadCOGTestNTiles1BandFloat32PC1ComNoneSquare extends ReadCOGTest {
    private final static String TEST_NAME = "ReadCOGTest";
    public final static String TEST_CLASS_DIR = TEST_DIR + ReadCOGTestNTiles1BandFloat32PC1ComNoneSquare.class.getSimpleName() + "/";

    protected String getTestName() {
        return TEST_NAME;
    }

    protected String getTestClassDir() {
        return TEST_CLASS_DIR;
    }

    protected double getResult(){ return 126423.0; }

    protected int getScriptId() {
        return 1;
    }

    protected int getId() {
        return 1;
    }
}

package org.apache.sysds.test.functions.io.cog;

public class ReadCOGTest1Tile1BandFloat32PC1ComNoneSquare extends ReadCOGTest {
    private final static String TEST_NAME = "ReadCOGTest";
    public final static String TEST_CLASS_DIR = TEST_DIR + ReadCOGTest1Tile1BandFloat32PC1ComNoneSquare.class.getSimpleName() + "/";

    protected String getTestName() {
        return TEST_NAME;
    }

    protected String getTestClassDir() {
        return TEST_CLASS_DIR;
    }

    protected int getScriptId() {
        return 1;
    }

    protected double getResult(){ return 510226.0; }

    protected int getId() {
        return 3;
    }
}

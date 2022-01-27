package org.apache.sysds.runtime.iogen.exp;

public class GIOMain {
    protected static String sampleRawFileName;
    protected static String sampleFrameFileName;
    protected static Integer sampleNRows;
    protected static String delimiter;
    protected static String schemaFileName;
    protected static String dataFileName;
    protected static String datasetName;
    protected static String cppBaseSrc;
    protected static String LOG_HOME;
    protected static Integer nrows;

    public static void getArgs(){
        sampleRawFileName = System.getProperty("sampleRawFileName");
        sampleFrameFileName = System.getProperty("sampleFrameFileName");
        sampleNRows = Integer.parseInt(System.getProperty("sampleNRows"));
        delimiter = System.getProperty("delimiter");
        if(delimiter.equals("\\t"))
            delimiter = "\t";
        schemaFileName = System.getProperty("schemaFileName");
        dataFileName = System.getProperty("dataFileName");
        datasetName = System.getProperty("datasetName");
        cppBaseSrc = System.getProperty("cppBaseSrc");
        LOG_HOME = System.getProperty("homeLog");
        nrows =  Integer.parseInt(System.getProperty("nrows"));
    }
}

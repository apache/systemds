package org.apache.sysds.runtime.iogen.exp;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderTextCSV;
import org.apache.sysds.runtime.iogen.GIO.Util;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.matrix.data.FrameBlock;

public class GIOFlatFrameExperimentHDFS extends GIOMain {

    public static void main(String[] args) throws Exception {
        getArgs();
        Util util = new Util();
        Types.ValueType[] sampleSchema = util.getSchema(schemaFileName);
        int ncols = sampleSchema.length;

        FileFormatPropertiesCSV csvpro = new FileFormatPropertiesCSV(false, delimiter, false);
        FrameReaderTextCSV csv = new FrameReaderTextCSV(csvpro);
        FrameBlock sampleFrame = csv.readFrameFromHDFS(sampleFrameFileName, sampleSchema, -1, ncols);

        double tmpTime = System.nanoTime();
        String sampleRaw = util.readEntireTextFile(sampleRawFileName);
        GenerateReader.GenerateReaderFrame gr = new GenerateReader.GenerateReaderFrame(sampleRaw, sampleFrame);
        FrameReader fr = gr.getReader();
        double generateTime = (System.nanoTime() - tmpTime) / 1000000000.0;

        tmpTime = System.nanoTime();
        FrameBlock frameBlock = fr.readFrameFromHDFS(dataFileName, sampleSchema, -1, sampleSchema.length);
        double readTime = (System.nanoTime() - tmpTime) / 1000000000.0;

        String log = datasetName + "," + frameBlock.getNumRows() + "," + frameBlock.getNumColumns() + "," + sampleSchema.length + "," + sampleNRows + "," + generateTime + "," + readTime;
        util.addLog(LOG_HOME, log);
    }
}

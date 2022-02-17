package org.apache.sysds.runtime.iogen.EXP;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.util.ArrayList;
import java.util.HashSet;

public class GIOFrame {

	public static void main(String[] args) throws Exception {
		String sampleRawFileName;
		String sampleFrameFileName;
		String sampleRawDelimiter;
		String schemaFileName;
		String dataFileName;
		long nrows;

		sampleRawFileName = System.getProperty("sampleRawFileName");
		sampleFrameFileName = System.getProperty("sampleFrameFileName");
		sampleRawDelimiter = System.getProperty("delimiter");
		if(sampleRawDelimiter.equals("\\t"))
			sampleRawDelimiter = "\t";
		schemaFileName = System.getProperty("schemaFileName");
		dataFileName = System.getProperty("dataFileName");
		nrows = Long.parseLong(System.getProperty("nrows"));


		Util util = new Util();
		Types.ValueType[] sampleSchema = util.getSchema(schemaFileName);
		int ncols = sampleSchema.length;

		String[][] sampleFrameStrings = util.loadFrameData(sampleFrameFileName, sampleRawDelimiter, ncols);
		FrameBlock sampleFrame = new FrameBlock(sampleSchema, sampleFrameStrings);
		String sampleRaw = util.readEntireTextFile(sampleRawFileName);
		GenerateReader.GenerateReaderFrame gr = new GenerateReader.GenerateReaderFrame(sampleRaw, sampleFrame);
		FrameReader fr = gr.getReader();
		FrameBlock frameBlock = fr.readFrameFromHDFS(dataFileName, sampleSchema, nrows, sampleSchema.length);

	}
}

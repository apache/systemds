package org.apache.sysds.test.functions.iogen.experiment;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.iogen.ReaderMappingIndex;

public class GIOFrameIdentificationIndex {

	public static void main(String[] args) throws Exception {
		String sampleRawFileName;
		String sampleFrameFileName;
		String sampleRawDelimiter;
		String schemaFileName;
		boolean parallel;

		sampleRawFileName = System.getProperty("sampleRawFileName");
		sampleFrameFileName = System.getProperty("sampleFrameFileName");
		parallel = Boolean.parseBoolean(System.getProperty("parallel"));
		sampleRawDelimiter = "\t";

		schemaFileName = System.getProperty("schemaFileName");
		Util util = new Util();
		Types.ValueType[] sampleSchema = util.getSchema(schemaFileName);
		int ncols = sampleSchema.length;

		String[][] sampleFrameStrings = util.loadFrameData(sampleFrameFileName, sampleRawDelimiter, ncols);
		FrameBlock sampleFrame = new FrameBlock(sampleSchema, sampleFrameStrings);

		String sampleRaw = util.readEntireTextFile(sampleRawFileName);
		ReaderMappingIndex readerMapping = new ReaderMappingIndex(sampleRaw, sampleFrame);
	}
}

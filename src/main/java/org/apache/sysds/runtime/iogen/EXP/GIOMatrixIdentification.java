package org.apache.sysds.runtime.iogen.EXP;

import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class GIOMatrixIdentification {

	public static void main(String[] args) throws Exception {
		String sampleRawFileName;
		String sampleMatrixFileName;
		String sampleRawDelimiter;

		sampleRawFileName = System.getProperty("sampleRawFileName");
		sampleMatrixFileName = System.getProperty("sampleMatrixFileName");
		sampleRawDelimiter = "\t";

		Util util = new Util();
		MatrixBlock sampleMB = util.loadMatrixData(sampleMatrixFileName, sampleRawDelimiter);
		String sampleRaw = util.readEntireTextFile(sampleRawFileName);

		GenerateReader.GenerateReaderMatrix gr = new GenerateReader.GenerateReaderMatrix(sampleRaw, sampleMB);
		gr.getReader();
	}
}

package org.apache.sysds.runtime.iogen.EXP;

import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class GIOMatrix {

	public static void main(String[] args) throws Exception {
		String sampleRawFileName;
		String sampleMatrixFileName;
		String sampleRawDelimiter;
		String dataFileName;
		long nrows;

		sampleRawFileName = System.getProperty("sampleRawFileName");
		sampleMatrixFileName = System.getProperty("sampleMatrixFileName");
		sampleRawDelimiter = System.getProperty("delimiter");
		if(sampleRawDelimiter.equals("\\t"))
			sampleRawDelimiter = "\t";
		dataFileName = System.getProperty("dataFileName");
		nrows = Long.parseLong(System.getProperty("nrows"));

		Util util = new Util();
		MatrixBlock sampleMB = util.loadMatrixData(sampleMatrixFileName, sampleRawDelimiter);
		String sampleRaw = util.readEntireTextFile(sampleRawFileName);

		GenerateReader.GenerateReaderMatrix gr = new GenerateReader.GenerateReaderMatrix(sampleRaw, sampleMB);
		MatrixReader matrixReader = gr.getReader();
		MatrixBlock matrixBlock = matrixReader.readMatrixFromHDFS(dataFileName, nrows, sampleMB.getNumColumns(), -1, -1);
	}
}

package org.apache.sysds.runtime.iogen.EXP;

import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.wink.json4j.JSONObject;

public class GIOMatrix {

	public static void main(String[] args) throws Exception {
		String sampleRawFileName;
		String sampleMatrixFileName;
		String sampleRawDelimiter;
		String dataFileName;
		boolean parallel;
		long rows = -1;

		sampleRawFileName = System.getProperty("sampleRawFileName");
		sampleMatrixFileName = System.getProperty("sampleMatrixFileName");
		sampleRawDelimiter = "\t";
		dataFileName = System.getProperty("dataFileName");
		parallel = Boolean.parseBoolean(System.getProperty("parallel"));
		Util util = new Util();
		// read and parse mtd file
		String mtdFileName = dataFileName + ".mtd";
		try {
			String mtd = util.readEntireTextFile(mtdFileName);
			mtd = mtd.replace("\n", "").replace("\r", "");
			mtd = mtd.toLowerCase().trim();
			JSONObject jsonObject = new JSONObject(mtd);
			if (jsonObject.containsKey("rows")) rows = jsonObject.getLong("rows");
		} catch (Exception exception) {}


		MatrixBlock sampleMB = util.loadMatrixData(sampleMatrixFileName, sampleRawDelimiter);
		String sampleRaw = util.readEntireTextFile(sampleRawFileName);

		GenerateReader.GenerateReaderMatrix gr = new GenerateReader.GenerateReaderMatrix(sampleRaw, sampleMB, parallel);
		MatrixReader matrixReader = gr.getReader();
		MatrixBlock matrixBlock = matrixReader.readMatrixFromHDFS(dataFileName, rows, sampleMB.getNumColumns(), -1, -1);
	}
}

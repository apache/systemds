package org.apache.sysds.runtime.iogen.exp;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FileFormatPropertiesMM;
import org.apache.sysds.runtime.io.FrameReaderTextCSV;
import org.apache.sysds.runtime.io.FrameReaderTextCell;
import org.apache.sysds.runtime.matrix.data.FrameBlock;

public class SYSDSFrameExperimentHDFS {

	public static void main(String[] args) throws Exception {

		String delimiter = " ";//args[0];
		String schemaFileName = args[1];
		String dataFileName = args[2];
		String datasetName = args[3];
		String LOG_HOME =args[4];
		Integer nrows = Integer.parseInt(args[5]);

		if(delimiter.equals("\\t"))
			delimiter = "\t";

		Util util = new Util();
		Types.ValueType[] schema = util.getSchema(schemaFileName);
		int ncols = schema.length;

		System.out.println("ncols = "+ncols + " >> "+ schema.length);

		double tmpTime = System.nanoTime();
		FrameBlock frameBlock;
		if(datasetName.equals("csv")) {
			FileFormatPropertiesCSV csvpro = new FileFormatPropertiesCSV(false, delimiter, false);
			FrameReaderTextCSV csv = new FrameReaderTextCSV(csvpro);
			frameBlock = csv.readFrameFromHDFS(dataFileName, schema, -1, ncols);
		}
		else if(datasetName.equals("mm")) {
			FileFormatPropertiesMM mmpro = new FileFormatPropertiesMM();
			FrameReaderTextCell mm =new FrameReaderTextCell();
			frameBlock = mm.readFrameFromHDFS(dataFileName, schema, nrows, ncols);
		}
		else
			throw new RuntimeException("Format not support!");

		double readTime = (System.nanoTime() - tmpTime) / 1000000000.0;

		String log= datasetName+","+ frameBlock.getNumRows()+","+ ncols+",1.0,0,0,"+readTime;
		util.addLog(LOG_HOME, log);
	}
}

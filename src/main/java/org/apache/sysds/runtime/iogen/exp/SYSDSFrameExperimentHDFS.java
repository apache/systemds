package org.apache.sysds.runtime.iogen.exp;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReaderTextCSV;
import org.apache.sysds.runtime.io.FrameReaderTextCell;
import org.apache.sysds.runtime.matrix.data.FrameBlock;

public class SYSDSFrameExperimentHDFS extends GIOMain {

	public static void main(String[] args) throws Exception {
		getArgs();

		Util util = new Util();
		Types.ValueType[] schema = util.getSchema(schemaFileName);
		int ncols = schema.length;

		System.out.println(">>>>>>>>>>>>>>>>>>> "+ncols);

		double tmpTime = System.nanoTime();
		FrameBlock frameBlock;
		//if(datasetName.equals("csv")) {
			FileFormatPropertiesCSV csvpro = new FileFormatPropertiesCSV(false, ",", false);
			FrameReaderTextCSV csv = new FrameReaderTextCSV(csvpro);
			frameBlock = csv.readFrameFromHDFS(dataFileName, schema, -1, ncols);
//		}
//		else if(datasetName.equals("mm")) {
//			FrameReaderTextCell mm =new FrameReaderTextCell();
//			frameBlock = mm.readFrameFromHDFS(dataFileName, schema, nrows, ncols);
//		}
//		else
//			throw new RuntimeException("Format not support!");

		double readTime = (System.nanoTime() - tmpTime) / 1000000000.0;

		String log= datasetName+","+ frameBlock.getNumRows()+","+ ncols+",1.0,0,0,"+readTime;
		util.addLog(LOG_HOME, log);
	}
}

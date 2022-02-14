package org.apache.sysds.runtime.iogen.EXP;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReaderJSONGson;
import org.apache.sysds.runtime.io.FrameReaderTextCSV;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.wink.json4j.JSONException;

import java.io.IOException;
import java.util.Map;

public class SystemDSCSV {

	public static void main(String[] args) throws IOException, JSONException {

		String schemaFileName;
		String schemaMapFileName;
		String dataFileName;
		long nrows;

		schemaFileName = System.getProperty("schemaFileName");
		schemaMapFileName = System.getProperty("schemaMapFileName");
		dataFileName = System.getProperty("dataFileName");
		nrows = Long.parseLong(System.getProperty("nrows"));

		Util util = new Util();
		Types.ValueType[] schema = util.getSchema(schemaFileName);
		int ncols = schema.length;
		FileFormatPropertiesCSV propertiesCSV = new FileFormatPropertiesCSV(false, "\t", false);
		FrameReaderTextCSV frameReaderTextCSV = new FrameReaderTextCSV(propertiesCSV);
		FrameBlock readBlock = frameReaderTextCSV.readFrameFromHDFS(dataFileName, schema, nrows, ncols);

	}
}

package org.apache.sysds.test.functions.iogen.experiment;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FrameReaderJSONL;
import org.apache.sysds.runtime.io.FrameReaderJSONLParallel;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.wink.json4j.JSONException;

import java.io.IOException;
import java.util.Map;

public class SystemDSJson4j {

	public static void main(String[] args) throws IOException, JSONException {

		String schemaFileName;
		String schemaMapFileName;
		String dataFileName;
		long nrows;
		boolean parallel;

		schemaFileName = System.getProperty("schemaFileName");
		schemaMapFileName = System.getProperty("schemaMapFileName");
		dataFileName = System.getProperty("dataFileName");
		nrows = Long.parseLong(System.getProperty("nrows"));
		parallel = Boolean.parseBoolean(System.getProperty("parallel"));

		Util util = new Util();
		Types.ValueType[] schema = util.getSchema(schemaFileName);
		int ncols = schema.length;
		Map<String, Integer> schemaMap = util.getSchemaMap(schemaMapFileName);

		if(parallel) {
			FrameReaderJSONL frameReaderJSONL = new FrameReaderJSONL();
			FrameBlock readBlock = frameReaderJSONL.readFrameFromHDFS(dataFileName, schema, schemaMap, nrows, ncols);
		}
		else {
			FrameReaderJSONLParallel frameReaderJSONL = new FrameReaderJSONLParallel();
			FrameBlock readBlock = frameReaderJSONL.readFrameFromHDFS(dataFileName, schema, schemaMap, nrows, ncols);
		}
	}
}

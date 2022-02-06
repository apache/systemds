package org.apache.sysds.runtime.iogen.Baseline;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FrameReaderJSONJackson;
import org.apache.sysds.runtime.io.FrameReaderJSONL;
import org.apache.sysds.runtime.iogen.GIO.Util;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.wink.json4j.JSONException;

import java.io.IOException;
import java.util.Map;

public class SystemDSJSON {

	public static void main(String[] args) throws IOException, JSONException {

		String schemaFileName;
		String schemaMapFileName;
		String dataFileName;
		long nrows;
		String config;

		schemaFileName = System.getProperty("schemaFileName");
		schemaMapFileName = System.getProperty("schemaMapFileName");
		dataFileName = System.getProperty("dataFileName");
		nrows = Long.parseLong(System.getProperty("nrows"));
		config = System.getProperty("config");

		Util util = new Util();
		Types.ValueType[] schema = util.getSchema(schemaFileName);
		int ncols = schema.length;
		Map<String, Integer> schemaMap = util.getSchemaMap(schemaMapFileName);

		FrameBlock readBlock;
		if(config.equals("SystemDS+Jason4j")) {
			FrameReaderJSONL frameReaderJSONL = new FrameReaderJSONL();
			readBlock = frameReaderJSONL.readFrameFromHDFS(dataFileName, schema, schemaMap, nrows, ncols);
		}
		else if(config.equals("SystemDS+Jackson")) {
			FrameReaderJSONJackson frameReaderJSONJackson = new FrameReaderJSONJackson();
			readBlock = frameReaderJSONJackson.readFrameFromHDFS(dataFileName, schema, schemaMap, nrows, ncols);
		}

	}
}

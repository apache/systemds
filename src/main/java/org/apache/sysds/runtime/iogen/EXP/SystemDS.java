package org.apache.sysds.runtime.iogen.EXP;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.*;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.io.IOException;
import java.util.Map;

public class SystemDS {

	public static void main(String[] args) throws IOException, JSONException {

		String schemaFileName;
		String dataFileName;
		String dataType = null;
		String valueType;
		String sep = null;
		String indSep = null;
		boolean header = false;
		long cols = -1;
		long rows = -1;
		String format = null;
		String config = null;
		String schemaMapFileName = null;
		boolean parallel;

		Util util = new Util();
		schemaFileName = System.getProperty("schemaFileName");
		dataFileName = System.getProperty("dataFileName");
		parallel = Boolean.parseBoolean(System.getProperty("parallel"));
		// read and parse mtd file
		String mtdFileName = dataFileName + ".mtd";
		try {
			String mtd = util.readEntireTextFile(mtdFileName);
			mtd = mtd.replace("\n", "").replace("\r", "");
			mtd = mtd.trim();
			JSONObject jsonObject = new JSONObject(mtd);
			if(jsonObject.containsKey("data_type"))
				dataType = jsonObject.getString("data_type");

			if(jsonObject.containsKey("value_type"))
				valueType = jsonObject.getString("value_type");

			if(jsonObject.containsKey("format"))
				format = jsonObject.getString("format");

			if(jsonObject.containsKey("cols"))
				cols = jsonObject.getLong("cols");

			if(jsonObject.containsKey("rows"))
				rows = jsonObject.getLong("rows");

			if(jsonObject.containsKey("header"))
				header = jsonObject.getBoolean("header");

			if(jsonObject.containsKey("schema_path"))
				schemaFileName = jsonObject.getString("schema_path");

			if(jsonObject.containsKey("sep"))
				sep = jsonObject.getString("sep");

			if(jsonObject.containsKey("indSep"))
				indSep = jsonObject.getString("indSep");

		}
		catch(Exception exception) {
		}

		if(dataType.equalsIgnoreCase("matrix")) {
			MatrixReader matrixReader = null;
			if(!parallel) {
				switch(format) {
					case "csv":
						FileFormatPropertiesCSV propertiesCSV = new FileFormatPropertiesCSV(header, sep, false);
						matrixReader = new ReaderTextCSV(propertiesCSV);
						matrixReader.readMatrixFromHDFS(dataFileName, rows, cols, -1, -1);
						break;
					case "libsvm":
						FileFormatPropertiesLIBSVM propertiesLIBSVM = new FileFormatPropertiesLIBSVM(sep, indSep, false);
						matrixReader = new ReaderTextLIBSVM(propertiesLIBSVM);
						matrixReader.readMatrixFromHDFS(dataFileName, rows, cols, -1, -1);
						break;
					case "mm":
						matrixReader = new ReaderTextCell(Types.FileFormat.MM, true);
						break;
				}
			}
			else {
				switch(format) {
					case "csv":
						FileFormatPropertiesCSV propertiesCSV = new FileFormatPropertiesCSV(header, sep, false);
						matrixReader = new ReaderTextCSVParallel(propertiesCSV);
						matrixReader.readMatrixFromHDFS(dataFileName, rows, cols, -1, -1);
						break;
					case "libsvm":
						FileFormatPropertiesLIBSVM propertiesLIBSVM = new FileFormatPropertiesLIBSVM(sep, indSep, false);
						matrixReader = new ReaderTextLIBSVMParallel(propertiesLIBSVM);
						matrixReader.readMatrixFromHDFS(dataFileName, rows, cols, -1, -1);
						break;
					case "mm":
						matrixReader = new ReaderTextCellParallel(Types.FileFormat.MM);
						break;
				}
			}
			if(matrixReader == null)
				throw new IOException("The Matrix Reader is NULL: " + dataFileName + ", format: " + format);
			matrixReader.readMatrixFromHDFS(dataFileName, rows, cols, -1, -1);
		}
		else {
			Types.ValueType[] schema = util.getSchema(schemaFileName);
			cols = schema.length;
			FrameBlock frameBlock = null;

			if(!parallel) {
				switch(format) {
					case "csv":
						FileFormatPropertiesCSV propertiesCSV = new FileFormatPropertiesCSV(header, sep, false);
						FrameReader frameReader = new FrameReaderTextCSV(propertiesCSV);
						frameBlock = frameReader.readFrameFromHDFS(dataFileName, schema, rows, cols);
						break;
					case "json":
						schemaMapFileName = System.getProperty("schemaMapFileName");
						Map<String, Integer> schemaMap = util.getSchemaMap(schemaMapFileName);
						config = System.getProperty("config");
						switch(config.toLowerCase()) {
							case "gson":
								FrameReaderJSONGson frameReaderJSONGson = new FrameReaderJSONGson();
								frameBlock = frameReaderJSONGson.readFrameFromHDFS(dataFileName, schema, schemaMap, rows, cols);
								break;

							case "jackson":
								FrameReaderJSONJackson frameReaderJSONJackson = new FrameReaderJSONJackson();
								frameBlock = frameReaderJSONJackson.readFrameFromHDFS(dataFileName, schema, schemaMap, rows, cols);
								break;
							case "json4j":
								FrameReaderJSONL frameReaderJSONL = new FrameReaderJSONL();
								frameBlock = frameReaderJSONL.readFrameFromHDFS(dataFileName, schema, schemaMap, rows, cols);
								break;
							default:
								throw new IOException("JSON Config don't support!!" + config);
						}
						break;
				}
			}
			else {
				switch(format) {
					case "csv":
						FileFormatPropertiesCSV propertiesCSV = new FileFormatPropertiesCSV(header, sep, false);
						FrameReader frameReader = new FrameReaderTextCSVParallel(propertiesCSV);
						frameBlock = frameReader.readFrameFromHDFS(dataFileName, schema, rows, cols);
						break;
					case "json":
						schemaMapFileName = System.getProperty("schemaMapFileName");
						Map<String, Integer> schemaMap = util.getSchemaMap(schemaMapFileName);
						config = System.getProperty("config");
						switch(config.toLowerCase()) {
							case "gson":
								FrameReaderJSONGsonParallel frameReaderJSONGson = new FrameReaderJSONGsonParallel();
								frameBlock = frameReaderJSONGson.readFrameFromHDFS(dataFileName, schema, schemaMap, rows, cols);
								break;

							case "jackson":
								FrameReaderJSONJacksonParallel frameReaderJSONJackson = new FrameReaderJSONJacksonParallel();
								frameBlock = frameReaderJSONJackson.readFrameFromHDFS(dataFileName, schema, schemaMap, rows, cols);
								break;
							case "json4j":
								FrameReaderJSONLParallel frameReaderJSONL = new FrameReaderJSONLParallel();
								frameBlock = frameReaderJSONL.readFrameFromHDFS(dataFileName, schema, schemaMap, rows, cols);
								break;
							default:
								throw new IOException("JSON Config don't support!!" + config);
						}
						break;
				}
			}
			
		}

	}
}

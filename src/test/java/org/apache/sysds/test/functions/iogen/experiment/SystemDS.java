package org.apache.sysds.test.functions.iogen.experiment;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.*;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.test.functions.iogen.baseline.FileFormatPropertiesAMiner;
import org.apache.sysds.test.functions.iogen.baseline.FileFormatPropertiesHL7;
import org.apache.sysds.test.functions.iogen.baseline.FrameReaderJSONGson;
import org.apache.sysds.test.functions.iogen.baseline.FrameReaderJSONGsonParallel;
import org.apache.sysds.test.functions.iogen.baseline.FrameReaderJSONJackson;
import org.apache.sysds.test.functions.iogen.baseline.FrameReaderJSONJacksonParallel;
import org.apache.sysds.test.functions.iogen.baseline.FrameReaderTextAMiner;
import org.apache.sysds.test.functions.iogen.baseline.FrameReaderTextAMinerParallel;
import org.apache.sysds.test.functions.iogen.baseline.FrameReaderTextHL7;
import org.apache.sysds.test.functions.iogen.baseline.FrameReaderTextHL7Parallel;
import org.apache.sysds.test.functions.iogen.baseline.FrameReaderXMLJackson;
import org.apache.sysds.test.functions.iogen.baseline.FrameReaderXMLJacksonParallel;
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
		Types.ValueType[] schema;
		String beginToken = null;
		String endToken = null;

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

			if(jsonObject.containsKey("begin_token"))
				beginToken = jsonObject.getString("begin_token");

			if(jsonObject.containsKey("end_token"))
				endToken = jsonObject.getString("end_token");

		}
		catch(Exception exception) {}

		if(dataType.equalsIgnoreCase("matrix")) {
			MatrixReader matrixReader = null;
			if(!parallel) {
				switch(format) {
					case "csv":
						FileFormatPropertiesCSV propertiesCSV = new FileFormatPropertiesCSV(header, sep, false);
						matrixReader = new ReaderTextCSV(propertiesCSV);
						break;
					case "libsvm":
						FileFormatPropertiesLIBSVM propertiesLIBSVM = new FileFormatPropertiesLIBSVM(sep, indSep,
							false);
						matrixReader = new ReaderTextLIBSVM(propertiesLIBSVM);
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
						break;
					case "libsvm":
						FileFormatPropertiesLIBSVM propertiesLIBSVM = new FileFormatPropertiesLIBSVM(sep, indSep,
							false);
						matrixReader = new ReaderTextLIBSVMParallel(propertiesLIBSVM);
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
			FrameBlock frameBlock = null;
			if(!parallel) {
				switch(format) {
					case "csv":
						schema = util.getSchema(schemaFileName);
						cols = schema.length;
						FileFormatPropertiesCSV propertiesCSV = new FileFormatPropertiesCSV(header, sep, false);
						FrameReader frameReader = new FrameReaderTextCSV(propertiesCSV);
						frameBlock = frameReader.readFrameFromHDFS(dataFileName, schema, rows, cols);
						break;
					case "json":
						schema = util.getSchema(schemaFileName);
						cols = schema.length;
						schemaMapFileName = System.getProperty("schemaMapFileName");
						Map<String, Integer> schemaMap = util.getSchemaMap(schemaMapFileName);
						config = System.getProperty("config");
						switch(config.toLowerCase()) {
							case "gson":
								FrameReaderJSONGson frameReaderJSONGson = new FrameReaderJSONGson();
								frameBlock = frameReaderJSONGson.readFrameFromHDFS(dataFileName, schema, schemaMap,
									rows, cols);
								break;

							case "jackson":
								FrameReaderJSONJackson frameReaderJSONJackson = new FrameReaderJSONJackson();
								frameBlock = frameReaderJSONJackson.readFrameFromHDFS(dataFileName, schema, schemaMap,
									rows, cols);
								break;
							case "json4j":
								FrameReaderJSONL frameReaderJSONL = new FrameReaderJSONL();
								frameBlock = frameReaderJSONL.readFrameFromHDFS(dataFileName, schema, schemaMap, rows,
									cols);
								break;
							default:
								throw new IOException("JSON Config don't support!!" + config);
						}
						break;

					case "aminer-author":
						FileFormatPropertiesAMiner propertiesAMinerAuthor = new FileFormatPropertiesAMiner("author");
						FrameReader frAuthor = new FrameReaderTextAMiner(propertiesAMinerAuthor);
						frameBlock = frAuthor.readFrameFromHDFS(dataFileName, null, null, -1, -1);
						break;
					case "aminer-paper":
						FileFormatPropertiesAMiner propertiesAMinerPaper = new FileFormatPropertiesAMiner("paper");
						FrameReader frPaper = new FrameReaderTextAMiner(propertiesAMinerPaper);
						frameBlock = frPaper.readFrameFromHDFS(dataFileName, null, null, -1, -1);
						break;
					case "xml":
						schema = util.getSchema(schemaFileName);
						cols = schema.length;
						schemaMapFileName = System.getProperty("schemaMapFileName");
						Map<String, Integer> xmlSchemaMap = util.getSchemaMap(schemaMapFileName);
						FrameReaderXMLJackson jacksonXML = new FrameReaderXMLJackson();
						frameBlock = jacksonXML.readFrameFromHDFS(dataFileName, schema, xmlSchemaMap, beginToken,
							endToken, -1, cols);
						break;

					case "hl7":
						Pair<int[], Integer> pair = getHL7Properties(System.getProperty("schemaMapFileName"));
						FileFormatPropertiesHL7 properties = new FileFormatPropertiesHL7(pair.getKey(), pair.getValue());
						schema = new Types.ValueType[pair.getKey().length];
						for(int i=0; i<pair.getKey().length; i++)
							schema[i] = Types.ValueType.STRING;
						FrameReaderTextHL7 hl7 = new FrameReaderTextHL7(properties);
						frameBlock = hl7.readFrameFromHDFS(dataFileName, schema, -1, pair.getKey().length);
						break;
				}
			}
			else {
				switch(format) {
					case "csv":
						schema = util.getSchema(schemaFileName);
						cols = schema.length;
						FileFormatPropertiesCSV propertiesCSV = new FileFormatPropertiesCSV(header, sep, false);
						FrameReader frameReader = new FrameReaderTextCSVParallel(propertiesCSV);
						frameBlock = frameReader.readFrameFromHDFS(dataFileName, schema, rows, cols);
						break;
					case "json":
						schema = util.getSchema(schemaFileName);
						cols = schema.length;
						schemaMapFileName = System.getProperty("schemaMapFileName");
						Map<String, Integer> schemaMap = util.getSchemaMap(schemaMapFileName);
						config = System.getProperty("config");
						switch(config.toLowerCase()) {
							case "gson":
								FrameReaderJSONGsonParallel frameReaderJSONGson = new FrameReaderJSONGsonParallel();
								frameBlock = frameReaderJSONGson.readFrameFromHDFS(dataFileName, schema, schemaMap,
									rows, cols);
								break;

							case "jackson":
								FrameReaderJSONJacksonParallel frameReaderJSONJackson = new FrameReaderJSONJacksonParallel();
								frameBlock = frameReaderJSONJackson.readFrameFromHDFS(dataFileName, schema, schemaMap,
									rows, cols);
								break;
							case "json4j":
								FrameReaderJSONLParallel frameReaderJSONL = new FrameReaderJSONLParallel();
								frameBlock = frameReaderJSONL.readFrameFromHDFS(dataFileName, schema, schemaMap, rows,
									cols);
								break;
							default:
								throw new IOException("JSON Config don't support!!" + config);
						}
						break;
					case "aminer-author":
						FileFormatPropertiesAMiner propertiesAMinerAuthor = new FileFormatPropertiesAMiner("author");
						FrameReader frAuthor = new FrameReaderTextAMinerParallel(propertiesAMinerAuthor);
						frameBlock = frAuthor.readFrameFromHDFS(dataFileName, null, null, -1, -1);
						break;
					case "aminer-paper":
						FileFormatPropertiesAMiner propertiesAMinerPaper = new FileFormatPropertiesAMiner("paper");
						FrameReader frPaper = new FrameReaderTextAMinerParallel(propertiesAMinerPaper);
						frameBlock = frPaper.readFrameFromHDFS(dataFileName, null, null, -1, -1);
						break;

					case "xml":
						schema = util.getSchema(schemaFileName);
						cols = schema.length;
						schemaMapFileName = System.getProperty("schemaMapFileName");
						Map<String, Integer> xmlSchemaMap = util.getSchemaMap(schemaMapFileName);
						FrameReaderXMLJacksonParallel jacksonXML = new FrameReaderXMLJacksonParallel();
						frameBlock = jacksonXML.readFrameFromHDFS(dataFileName, schema, xmlSchemaMap, beginToken,
							endToken, -1, cols);
						break;

					case "hl7":
						Pair<int[], Integer> pair = getHL7Properties(System.getProperty("schemaMapFileName"));
						FileFormatPropertiesHL7 properties = new FileFormatPropertiesHL7(pair.getKey(), pair.getValue());
						schema = new Types.ValueType[pair.getKey().length];
						for(int i=0; i<pair.getKey().length; i++)
							schema[i] = Types.ValueType.STRING;
						FrameReaderTextHL7Parallel hl7 = new FrameReaderTextHL7Parallel(properties);
						frameBlock = hl7.readFrameFromHDFS(dataFileName, schema, -1, pair.getKey().length);
						break;

				}
			}

		}

	}

	private static Pair<int[], Integer> getHL7Properties(String fileName){
		int[] selectedIndexes;
		Integer maxColumnIndex = -1;
		if(fileName.contains("Q1") || fileName.contains("Q2")){
			if (fileName.contains("Q1"))
				selectedIndexes = new int[] {17,18,19,20};
			else
				selectedIndexes = new int[] {26,27,28,34,35,36,37,38,39,41};
			maxColumnIndex = 0;
		}
		else if(fileName.contains("F1") || fileName.contains("F2") || fileName.contains("F3") ||
			fileName.contains("F4") ||fileName.contains("F5") || fileName.contains("F6") || fileName.contains("F7") ||
			fileName.contains("F8") || fileName.contains("F9") || fileName.contains("F10")) {

			int count = 0;
			if(fileName.contains("F1") && !fileName.contains("F10")){
				count = 10;

			} else if(fileName.contains("F2")){
				count = 20;

			}else if(fileName.contains("F3")){
				count = 30;

			}else if(fileName.contains("F4")){
				count = 40;

			}else if(fileName.contains("F5")){
				count = 50;

			}else if(fileName.contains("F6")){
				count = 60;

			}else if(fileName.contains("F7")){
				count = 70;

			}else if(fileName.contains("F8")){
				count = 80;

			}else if(fileName.contains("F9")){
				count = 90;

			}else if(fileName.contains("F10")){
				count = 100;
			}

			selectedIndexes = new int[count];
			for(int i=0; i< count; i++)
				selectedIndexes[i] = i;
			maxColumnIndex = count;
		}
		else {
			int count = 101;
			selectedIndexes = new int[count];
			for(int i=0; i< count; i++)
				selectedIndexes[i] = i;
			maxColumnIndex = count;
		}

		return new Pair<>(selectedIndexes, maxColumnIndex);

	}
}

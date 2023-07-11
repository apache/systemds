package org.apache.sysds.test.functions.iogen.baseline;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.apache.sysds.runtime.io.FrameReader.checkValidInputFile;
import static org.apache.sysds.runtime.io.FrameReader.createOutputFrameBlock;
import static org.apache.sysds.runtime.io.FrameReader.createOutputSchema;

public class FrameReaderJSONGson
{
	public FrameBlock readFrameFromHDFS(String fname, Types.ValueType[] schema, Map<String, Integer> schemaMap,
		long rlen, long clen) throws IOException, DMLRuntimeException
	{
		//prepare file access
		JobConf jobConf = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fileSystem = IOUtilFunctions.getFileSystem(path, jobConf);
		FileInputFormat.addInputPath(jobConf, path);

		//check existence and non-empty file
		checkValidInputFile(fileSystem, path);

		Types.ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNamesFromSchemaMap(schemaMap);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);

		readJSONLFrameFromHDFS(path, jobConf, fileSystem, ret, schema, schemaMap);
		return ret;
	}


	protected void readJSONLFrameFromHDFS(Path path, JobConf jobConf, FileSystem fileSystem, FrameBlock dest,
		Types.ValueType[] schema, Map<String, Integer> schemaMap) throws IOException
	{
		TextInputFormat inputFormat = new TextInputFormat();
		inputFormat.configure(jobConf);
		InputSplit[] splits = inputFormat.getSplits(jobConf, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);

		for (int i = 0, rowPos = 0; i < splits.length; i++) {
			rowPos = readJSONLFrameFromInputSplit(splits[i], inputFormat, jobConf, schema, schemaMap, dest, rowPos);
		}
	}


	protected static int readJSONLFrameFromInputSplit(InputSplit split, InputFormat<LongWritable, Text> inputFormat,
		JobConf jobConf, Types.ValueType[] schema, Map<String, Integer> schemaMap, FrameBlock dest, int currentRow)
		throws IOException
	{
		RecordReader<LongWritable, Text> reader = inputFormat.getRecordReader(split, jobConf, Reporter.NULL);
		LongWritable key = new LongWritable();
		Text value = new Text();

		int row = currentRow;
		try {
			while (reader.next(key, value)) {
				JsonParser jsonParser = new JsonParser();
				JsonElement root= jsonParser.parse(value.toString());
				Map<String, String> map = new HashMap<>();
				addKeys("", root, map, new ArrayList<>());
				for (Map.Entry<String, Integer> entry : schemaMap.entrySet()) {
					String strCellValue = map.get(entry.getKey());
					if(strCellValue!=null){
						dest.set(row, entry.getValue(), UtilFunctions.stringToObject(schema[entry.getValue()], strCellValue));
					}
				}
				row++;
			}
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
		return row;
	}

	private static void addKeys(String currentPath, JsonElement jsonNode, Map<String, String> map, List<Integer> suffix) {

		if (jsonNode.isJsonObject()) {
			JsonObject jsonObject = (JsonObject) jsonNode;
			Set<Map.Entry<String, JsonElement>> iter = jsonObject.entrySet();
			String pathPrefix = currentPath.isEmpty() ? "" : currentPath + "/";
			for(Map.Entry<String, JsonElement> entry: iter){
				addKeys(pathPrefix + entry.getKey(), entry.getValue(), map, suffix);
			}
		} else if (jsonNode.isJsonArray()) {
			JsonArray arrayNode = (JsonArray) jsonNode;
			for (int i = 0; i < arrayNode.size(); i++) {
				suffix.add(i + 1);
				addKeys(currentPath+"-"+i, arrayNode.get(i), map, suffix);
				if (i + 1 <arrayNode.size()){
					suffix.remove(suffix.size() - 1);
				}
			}

		} else if (jsonNode.isJsonPrimitive()) {
			if (currentPath.contains("/") && !currentPath.contains("-")) {
				for (int i = 0; i < suffix.size(); i++) {
					currentPath += "/" + suffix.get(i);
				}
				suffix = new ArrayList<>();
			}
			JsonPrimitive valueNode = (JsonPrimitive) jsonNode;
			map.put(currentPath, valueNode.getAsString());
		}
	}


	private  String[] createOutputNamesFromSchemaMap(Map<String, Integer> schemaMap) {
		String[] names = new String[schemaMap.size()];
		schemaMap.forEach((key, value) -> names[value] = key);
		return names;
	}
}

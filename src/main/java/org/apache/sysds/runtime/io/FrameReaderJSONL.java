/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.io;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.util.Map;

import static org.apache.sysds.runtime.io.FrameReader.*;


public class FrameReaderJSONL
{
	public FrameBlock readFrameFromHDFS(String fname, Types.ValueType[] schema, Map<String, Integer> schemaMap,
		long rlen, long clen) throws IOException, DMLRuntimeException, JSONException
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
		Types.ValueType[] schema, Map<String, Integer> schemaMap) throws IOException, JSONException
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
			throws IOException, JSONException 
	{
		RecordReader<LongWritable, Text> reader = inputFormat.getRecordReader(split, jobConf, Reporter.NULL);
		LongWritable key = new LongWritable();
		Text value = new Text();

		int row = currentRow;
		try {
			while (reader.next(key, value)) {
				// Potential Problem if JSON/L Object is very large
				JSONObject jsonObject = new JSONObject(value.toString());
				for (Map.Entry<String, Integer> entry : schemaMap.entrySet()) {
					String strCellValue = getStringFromJSONPath(jsonObject, entry.getKey());
					dest.set(row, entry.getValue(), UtilFunctions.stringToObject(schema[entry.getValue()], strCellValue));
				}
				row++;
			}
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
		return row;
	}
	// TODO Needs Optimisation! "split" is inefficient
	private static String getStringFromJSONPath(JSONObject jsonObject, String path) 
		throws IOException 
	{
		String[] splitPath = path.split("/");
		Object temp = null;
		for (String split : splitPath) {
			if(split.equals("")) continue;
			try{
				if (temp == null) {
					temp = jsonObject.get(split);
				} else if (temp instanceof JSONObject) {
					temp = ((JSONObject) temp).get(split);
				} else if (temp instanceof JSONArray) {
					throw new IOException("Cannot traverse JSON Array in a meaningful manner");
				} else {
					return null;
				}
			}
			catch (JSONException e){
				// Value not in JsonObject
				return null;
			}

		}
		if(temp == null){
			throw new IOException("Could not traverse the JSON path: '" + path + "'!");
		}
		return temp.toString();
	}


	private static String[] createOutputNamesFromSchemaMap(Map<String, Integer> schemaMap) {
		String[] names = new String[schemaMap.size()];
		schemaMap.forEach((key, value) -> names[value] = key);
		return names;
	}
}

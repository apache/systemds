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

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Iterator;
import java.util.Map;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.iterators.IteratorFactory;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class FrameWriterJSONL
{
	public void writeFrameToHDFS(FrameBlock src, String fname, Map<String, Integer> schemaMap, long rlen, long clen)
		throws IOException, DMLRuntimeException, JSONException
	{
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		HDFSTool.deleteFileIfExistOnHDFS( fname );

		//validity check frame dimensions
		if( src.getNumRows() != rlen || src.getNumColumns() != clen ) {
			throw new IOException("Frame dimensions mismatch with metadata: " +
				src.getNumRows()+"x"+src.getNumColumns()+" vs "+rlen+"x"+clen+".");
		}

		//core write (sequential/parallel)
		writeJSONLFrameToHDFS(path, job, src, rlen, clen, schemaMap);
	}

	protected void writeJSONLFrameToHDFS(Path path, JobConf jobConf, FrameBlock src,
		long rlen, long clen, Map<String, Integer> schemaMap) throws IOException, JSONException
	{
		FileSystem fileSystem = IOUtilFunctions.getFileSystem(path, jobConf);

		//sequential write to single text file
		writeJSONLFrameToFile(path, fileSystem, src, 0, (int)rlen, schemaMap);
		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fileSystem, path);
	}

	protected void writeJSONLFrameToFile(Path path, FileSystem fileSystem, FrameBlock src, int lowerRowBound,
		int upperRowBound, Map<String, Integer> schemaMap) throws IOException, JSONException
	{
		BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(fileSystem.create(path, true)));

		try {
			Iterator<String[]> stringRowIterator = IteratorFactory.getStringRowIterator(src, lowerRowBound, upperRowBound);
			while (stringRowIterator.hasNext()) {
				String[] row = stringRowIterator.next();
				bufferedWriter.write(formatToJSONString(row, schemaMap) + "\n");
			}
		}
		finally {
			IOUtilFunctions.closeSilently(bufferedWriter);
		}
	}

	protected String formatToJSONString(String[] values, Map<String, Integer> schemaMap) throws IOException, JSONException {
		if(schemaMap.size() != values.length){
			throw new IOException("Schema Map and row mismatch. Cannot map "
				+ values.length + " values to " + schemaMap.size() + " JSON Objects");
		}
		JSONObject jsonObject = new JSONObject();
		for (Map.Entry<String, Integer> entry : schemaMap.entrySet()) {
			String[] splits = entry.getKey().split("/");
			Integer value = entry.getValue();
			gernerateJSONObjectFromPath(splits, 1, values[value],jsonObject);
		}
		return jsonObject.toString();
	}

	protected JSONObject gernerateJSONObjectFromPath(String[] path, int index, Object value, JSONObject jsonObject) throws JSONException {
		JSONObject temp = new JSONObject();
		if(index == path.length - 1){
			if(jsonObject != null){
				jsonObject.put(path[index], value);
				return jsonObject;
			}
			temp.put(path[index], value);
			return temp;
		}
		JSONObject newJsonObject = (jsonObject == null)? null : jsonObject.optJSONObject(path[index]);
		JSONObject ret = gernerateJSONObjectFromPath(path, index + 1, value, newJsonObject);

		if(newJsonObject == null && jsonObject != null){
			jsonObject.put(path[index], ret);
			return null;
		} else if( ret == null){
			return null;
		}
		temp.put(path[index], ret);
		return temp;
	}
}

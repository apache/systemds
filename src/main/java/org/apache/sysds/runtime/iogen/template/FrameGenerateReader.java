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

package org.apache.sysds.runtime.iogen.template;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.RowIndexStructure;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.util.InputStreamInputFormat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public abstract class FrameGenerateReader extends FrameReader {

	protected CustomProperties _props;

	public FrameGenerateReader(CustomProperties _props) {
		this._props = _props;
	}

	private int getNumRows(List<Path> files, FileSystem fs) throws IOException, DMLRuntimeException {
		int rows = 0;
		for(int fileNo = 0; fileNo < files.size(); fileNo++) {
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));
			try {
				// Row Identify
				if(_props.getRowIndexStructure().getProperties().equals(RowIndexStructure.IndexProperties.Identity)) {
					while(br.readLine() != null)
						rows++;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(br);
			}
		}
		return rows;
	}

	@Override
	public FrameBlock readFrameFromHDFS(String fname, Types.ValueType[] schema, String[] names, long rlen,
		long clen) throws IOException, DMLRuntimeException {

		// prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		FileInputFormat.addInputPath(job, path);

		// check existence and non-empty file
		checkValidInputFile(fs, path);

		// compute size if necessary
		if(rlen <= 0) {
			ArrayList<Path> paths = new ArrayList<>();
			paths.add(path);
			rlen = getNumRows(paths, fs);
		}

		// allocate output frame block
		Types.ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNames(names, clen);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);

		// core read (sequential/parallel)
		readFrameFromHDFS(path, job, fs, ret, lschema, lnames, rlen, clen);

		return ret;

	}

	@Override
	public FrameBlock readFrameFromInputStream(InputStream is, Types.ValueType[] schema, String[] names,
		long rlen, long clen) throws IOException, DMLRuntimeException {

		// allocate output frame block
		Types.ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNames(names, clen);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);

		// core read (sequential/parallel)
		InputStreamInputFormat informat = new InputStreamInputFormat(is);
		InputSplit split = informat.getSplits(null, 1)[0];
		readFrameFromInputSplit(split, informat, null, ret, schema, names, rlen, clen, 0, true);

		return ret;
	}

	protected void readFrameFromHDFS(Path path, JobConf job, FileSystem fs, FrameBlock dest, Types.ValueType[] schema,
		String[] names, long rlen, long clen) throws IOException {

		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);
		for(int i = 0, rpos = 0; i < splits.length; i++)
			rpos = readFrameFromInputSplit(splits[i], informat, job, dest, schema, names, rlen, clen, rpos, i == 0);
	}

	protected abstract int readFrameFromInputSplit(InputSplit split, InputFormat<LongWritable, Text> informat,
		JobConf job, FrameBlock dest, Types.ValueType[] schema, String[] names, long rlen, long clen, int rl,
		boolean first) throws IOException;

	protected int getEndPos(String str, int strLen, int currPos, HashSet<String> endWithValueString) {
		int endPos = strLen;
		for(String d : endWithValueString) {
			int pos = d.length()> 0 ? str.indexOf(d, currPos): strLen;
			if(pos != -1)
				endPos = Math.min(endPos, pos);
		}
		return endPos;
	}

}

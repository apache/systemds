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

package org.apache.sysds.runtime.iogen;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.util.InputStreamInputFormat;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public abstract class FrameGenerateReader extends FrameReader {

	protected CustomProperties _props;
	protected final FastStringTokenizer fastStringTokenizerDelim;

	public FrameGenerateReader(CustomProperties _props) {
		this._props = _props;
		fastStringTokenizerDelim = new FastStringTokenizer(_props.getDelim());
	}

	private int getNumRows(List<Path> files, FileSystem fs) throws IOException, DMLRuntimeException {
		int rows = 0;
		String value;
		for(int fileNo = 0; fileNo < files.size(); fileNo++) {
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));
			try {
				// Row Regular
				if(_props.getRowPattern().equals(CustomProperties.GRPattern.Regular)) {
					// TODO: check the file has header?
					while(br.readLine() != null)
						rows++;
				}
				// Row Irregular
				else {
					FastStringTokenizer st = new FastStringTokenizer(_props.getDelim());
					while((value = br.readLine()) != null) {
						st.reset(value);
						int row = st.nextInt();
						rows = Math.max(rows, row);
					}
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

	public static class FrameReaderRowRegularColRegular extends FrameGenerateReader {

		public FrameReaderRowRegularColRegular(CustomProperties _props) {
			super(_props);
		}

		@Override
		protected int readFrameFromInputSplit(InputSplit split, InputFormat<LongWritable, Text> informat,
			JobConf job, FrameBlock dest, Types.ValueType[] schema, String[] names, long rlen, long clen, int rl,
			boolean first) throws IOException {

			String cellValue;
			fastStringTokenizerDelim.setNaStrings(_props.getNaStrings());

			// create record reader
			RecordReader<LongWritable, Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			int row = rl;
			int col = 0;
			Set<String> naValues = _props.getNaStrings();

			// Read the data
			try {
				while(reader.next(key, value)) // foreach line
				{
					String cellStr = value.toString();
					fastStringTokenizerDelim.reset(cellStr);
					while(col != -1) {
						cellValue = fastStringTokenizerDelim.nextToken();
						col = fastStringTokenizerDelim.getIndex();
						if(col != -1 && cellValue != null && (naValues == null || !naValues.contains(cellValue))) {
							dest.set(row, col, UtilFunctions.stringToObject(schema[col], cellValue));
						}
					}
					row++;
					col = 0;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
			return row;
		}
	}

	public static class FrameReaderRowRegularColIrregular extends FrameGenerateReader {

		public FrameReaderRowRegularColIrregular(CustomProperties _props) {
			super(_props);
		}

		@Override
		protected int readFrameFromInputSplit(InputSplit split, InputFormat<LongWritable, Text> informat,
			JobConf job, FrameBlock dest, Types.ValueType[] schema, String[] names, long rlen, long clen, int rl,
			boolean first) throws IOException {

			String cellValue;
			FastStringTokenizer fastStringTokenizerIndexDelim = new FastStringTokenizer(_props.getIndexDelim());

			// create record reader
			RecordReader<LongWritable, Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			int row = rl;
			int col = 0;

			// Read the data
			try {
				while(reader.next(key, value)) // foreach line
				{
					String cellStr = value.toString();
					fastStringTokenizerDelim.reset(cellStr);
					String cellValueString = fastStringTokenizerDelim.nextToken();
					dest.set(row, (int) clen - 1 - _props.getFirstColIndex(),
						UtilFunctions.stringToObject(schema[(int) clen - 1 - _props.getFirstColIndex()], cellValueString));

					while(col != -1) {
						String nt = fastStringTokenizerDelim.nextToken();
						if(fastStringTokenizerDelim.getIndex() == -1)
							break;
						fastStringTokenizerIndexDelim.reset(nt);
						col = fastStringTokenizerIndexDelim.nextInt();
						cellValue = fastStringTokenizerIndexDelim.nextToken();
						if(col != -1 && cellValue != null) {
							dest.set(row, col - _props.getFirstColIndex(),
								UtilFunctions.stringToObject(schema[col - _props.getFirstColIndex()], cellValue));
						}
					}
					row++;
					col = 0;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
			return row;
		}
	}

	public static class FrameReaderRowIrregular extends FrameGenerateReader {

		public FrameReaderRowIrregular(CustomProperties _props) {
			super(_props);
		}

		@Override
		protected int readFrameFromInputSplit(InputSplit split, InputFormat<LongWritable, Text> informat,
			JobConf job, FrameBlock dest, Types.ValueType[] schema, String[] names, long rlen, long clen, int rl,
			boolean first) throws IOException {

			String cellValue;
			fastStringTokenizerDelim.setNaStrings(_props.getNaStrings());

			// create record reader
			RecordReader<LongWritable, Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			int row = rl;
			int col = 0;

			// Read the data
			try {
				while(reader.next(key, value)) // foreach line
				{
					String cellStr = value.toString();
					fastStringTokenizerDelim.reset(cellStr);
					int ri = fastStringTokenizerDelim.nextInt();
					col = fastStringTokenizerDelim.nextInt();
					cellValue = fastStringTokenizerDelim.nextToken();

					if(col != -1 && cellValue != null) {
						dest.set(ri-_props.getFirstRowIndex(), col - _props.getFirstColIndex(),
							UtilFunctions.stringToObject(schema[col - _props.getFirstColIndex()], cellValue));
					}
					row = Math.max(row, ri);
				}
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
			return row;
		}
	}
}

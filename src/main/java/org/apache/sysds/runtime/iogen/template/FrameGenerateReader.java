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
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.InputStreamInputFormat;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public abstract class FrameGenerateReader extends FrameReader {

	protected CustomProperties _props;
	protected TemplateUtil.SplitOffsetInfos _offsets;

	public FrameGenerateReader(CustomProperties _props) {
		this._props = _props;
	}

	@Override
	public FrameBlock readFrameFromHDFS(String fname, Types.ValueType[] schema, String[] names, long rlen, long clen) throws IOException, DMLRuntimeException {

		// prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		FileInputFormat.addInputPath(job, path);

		// check existence and non-empty file
		checkValidInputFile(fs, path);

		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);

		// allocate output frame block
		Types.ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNames(names, clen);

		FrameBlock ret;
		if(rlen <= 0 || _props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.SeqScatter)
			ret = computeSizeAndCreateOutputFrameBlock(informat,job, schema, names, splits, path);
		else
			ret = createOutputFrameBlock(lschema, lnames, rlen);

		readFrameFromHDFS(informat, splits, job, ret);
		return ret;

	}

	private FrameBlock computeSizeAndCreateOutputFrameBlock(TextInputFormat informat, JobConf job, Types.ValueType[] schema, String[] names,
		InputSplit[] splits, Path path)
		throws IOException, DMLRuntimeException {

		int row = 0;
		// count rows in parallel per split
		try {
			if(_props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.Identity) {
				// compute number of rows
				for(InputSplit inputSplit : splits) {
					RecordReader<LongWritable, Text> reader = informat.getRecordReader(inputSplit, job, Reporter.NULL);
					LongWritable key = new LongWritable();
					Text value = new Text();
					try {
						// count remaining number of rows, ignore meta data
						while(reader.next(key, value)) {
							row++;
						}
					}
					finally {
						IOUtilFunctions.closeSilently(reader);
					}
				}
			}
			else if(_props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.SeqScatter) {
				_offsets = new TemplateUtil.SplitOffsetInfos(splits.length);
				int splitIndex = 0;
				for(InputSplit inputSplit : splits) {
					int nrows = 0;
					TemplateUtil.SplitInfo splitInfo = new TemplateUtil.SplitInfo();
					ArrayList<Pair<Integer, Integer>> beginIndexes = TemplateUtil.getTokenIndexOnMultiLineRecords(inputSplit, informat, job,
						_props.getRowIndexStructure().getSeqBeginString());

					ArrayList<Pair<Integer, Integer>> endIndexes;
					int tokenLength = 0;
					boolean diffBeginEndToken = false;
					if(!_props.getRowIndexStructure().getSeqBeginString().equals(_props.getRowIndexStructure().getSeqEndString())) {
						endIndexes = TemplateUtil.getTokenIndexOnMultiLineRecords(inputSplit, informat, job, _props.getRowIndexStructure().getSeqEndString());
						tokenLength = _props.getRowIndexStructure().getSeqEndString().length();
						diffBeginEndToken = true;
					}
					else {
						endIndexes = new ArrayList<>();
						for(int i = 1; i < beginIndexes.size(); i++)
							endIndexes.add(beginIndexes.get(i));
					}
					beginIndexes.remove(beginIndexes.size()-1);
					int i = 0;
					int j = 0;
					while(i < beginIndexes.size() && j < endIndexes.size()) {
						Pair<Integer, Integer> p1 = beginIndexes.get(i);
						Pair<Integer, Integer> p2 = endIndexes.get(j);
						int n = 0;
						while(p1.getKey() < p2.getKey() || (p1.getKey() == p2.getKey() && p1.getValue() < p2.getValue())) {
							n++;
							i++;
							if(i == beginIndexes.size())
								break;
							p1 = beginIndexes.get(i);
						}
						j += n - 1;
						splitInfo.addIndexAndPosition(beginIndexes.get(i - n).getKey(), endIndexes.get(j).getKey(), beginIndexes.get(i - n).getValue(),
							endIndexes.get(j).getValue() + tokenLength);
						j++;
						nrows++;
					}
					if(!diffBeginEndToken && i == beginIndexes.size() && j < endIndexes.size())
						nrows++;
					if(beginIndexes.get(0).getKey() == 0 && beginIndexes.get(0).getValue() == 0)
						splitInfo.setRemainString("");
					else {
						RecordReader<LongWritable, Text> reader = informat.getRecordReader(inputSplit, job, Reporter.NULL);
						LongWritable key = new LongWritable();
						Text value = new Text();

						StringBuilder sb = new StringBuilder();
						for(int ri = 0; ri < beginIndexes.get(0).getKey(); ri++) {
							reader.next(key, value);
							String raw = value.toString();
							sb.append(raw);
						}
						if(beginIndexes.get(0).getValue() != 0) {
							reader.next(key, value);
							sb.append(value.toString().substring(0, beginIndexes.get(0).getValue()));
						}
						splitInfo.setRemainString(sb.toString());
					}
					splitInfo.setNrows(nrows);
					_offsets.setSeqOffsetPerSplit(splitIndex, splitInfo);
					_offsets.setOffsetPerSplit(splitIndex, row);
					row += nrows;
					splitIndex++;
				}
			}
		}
		catch(Exception e) {
			throw new IOException("Thread pool Error " + e.getMessage(), e);
		}
		FrameBlock ret = createOutputFrameBlock(schema, names, row);
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
		//readFrameFromInputSplit(split, informat, null, ret, schema, names, rlen, clen, 0, true);

		return ret;
	}

	protected void readFrameFromHDFS(TextInputFormat informat, InputSplit[] splits, JobConf job, FrameBlock dest) throws IOException {
		int rpos = 0;
		for(int i = 0; i < splits.length; i++) {
			RecordReader<LongWritable, Text> reader = informat.getRecordReader(splits[i], job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			TemplateUtil.SplitInfo splitInfo = null;
			if(_props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.SeqScatter){
				splitInfo = _offsets.getSeqOffsetPerSplit(i);
				rpos = _offsets.getOffsetPerSplit(i);
			}
			readFrameFromHDFS(reader, key, value, dest, rpos, splitInfo);
		}
	}

	protected abstract int readFrameFromHDFS(RecordReader<LongWritable, Text> reader, LongWritable key, Text value, FrameBlock dest,
		int rowPos, TemplateUtil.SplitInfo splitInfo) throws IOException;
}

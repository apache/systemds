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

import org.apache.commons.lang.mutable.MutableInt;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.RowIndexStructure;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.matrix.data.Pair;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public abstract class MatrixGenerateReader extends MatrixReader {

	protected static CustomProperties _props;
	protected TemplateUtil.SplitOffsetInfos _offsets;

	public MatrixGenerateReader(CustomProperties _props) {
		MatrixGenerateReader._props = _props;
	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {

		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		FileInputFormat.addInputPath(job, path);

		checkValidInputFile(fs, path);

		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);

		MatrixBlock ret;
		if(rlen >= 0 && clen >= 0 && _props.getRowIndexStructure().getProperties() != RowIndexStructure.IndexProperties.SeqScatter) {
//			if(_props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.RowWiseExist ||
//				_props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.CellWiseExist ){
//				//clen++;
//				//rlen ++;
//			}
			ret = createOutputMatrixBlock(rlen, clen, (int) rlen, estnnz, !_props.isSparse(), _props.isSparse());
		}
		else
			ret = computeSizeAndCreateOutputMatrixBlock(informat,job, splits, estnnz);

		//core read
		readMatrixFromHDFS(informat, splits, job, ret);

		return ret;
	}


	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {

		MatrixBlock ret = null;
		if(rlen >= 0 && clen >= 0) //otherwise allocated on read
			ret = createOutputMatrixBlock(rlen, clen, (int) rlen, estnnz, true, false);

		return ret;
	}

	private MatrixBlock computeSizeAndCreateOutputMatrixBlock(TextInputFormat informat, JobConf job, InputSplit[] splits, long estnnz) throws IOException, DMLRuntimeException {
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
		MatrixBlock ret = createOutputMatrixBlock(row, _props.getNcols(), (int) row, estnnz, !_props.isSparse(), _props.isSparse());
		return ret;
	}

	@SuppressWarnings("unchecked")
	protected void readMatrixFromHDFS(TextInputFormat informat, InputSplit[] splits, JobConf job, MatrixBlock dest) throws IOException {
		MutableInt row = new MutableInt(0);
		long lnnz = 0;
		for(int i = 0; i < splits.length; i++) {
			RecordReader<LongWritable, Text> reader = informat.getRecordReader(splits[i], job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			TemplateUtil.SplitInfo splitInfo = null;
			if(_props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.SeqScatter){
				splitInfo = _offsets.getSeqOffsetPerSplit(i);
				row.setValue(_offsets.getOffsetPerSplit(i));
			}
			lnnz += readMatrixFromHDFS(reader, key, value, dest, row, splitInfo);
		}
		//post processing
		dest.setNonZeros(lnnz);
	}

	protected abstract long readMatrixFromHDFS(RecordReader<LongWritable, Text> reader, LongWritable key, Text value, MatrixBlock dest,
		MutableInt rowPos, TemplateUtil.SplitInfo splitInfo) throws IOException;
}

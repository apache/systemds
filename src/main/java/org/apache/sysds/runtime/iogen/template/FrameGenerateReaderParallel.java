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
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.RowIndexStructure;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.InputStreamInputFormat;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public abstract class FrameGenerateReaderParallel extends FrameReader {

	protected CustomProperties _props;
	protected int _numThreads;
	protected JobConf job;
	protected TemplateUtil.SplitOffsetInfos _offsets;
	protected int _rLen;
	protected int _cLen;

	public FrameGenerateReaderParallel(CustomProperties _props) {
		this._numThreads = OptimizerUtils.getParallelTextReadParallelism();
		this._props = _props;
	}

	@Override public FrameBlock readFrameFromHDFS(String fname, Types.ValueType[] schema, String[] names, long rlen,
		long clen) throws IOException, DMLRuntimeException {

		//prepare file access
		job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		InputSplit[] splits = informat.getSplits(job, _numThreads);
		splits = IOUtilFunctions.sortInputSplits(splits);

		// check existence and non-empty file
		checkValidInputFile(fs, path);

		// allocate output frame block
		FrameBlock ret = computeSizeAndCreateOutputFrameBlock(schema, names, splits, path, rlen, clen);

		// core read (sequential/parallel)
		readFrameFromHDFS(splits, path, job, ret);
		return ret;
	}

	private FrameBlock computeSizeAndCreateOutputFrameBlock(Types.ValueType[] schema, String[] names,
		InputSplit[] splits, Path path, long rlen, long clen) throws IOException, DMLRuntimeException {
		_rLen = 0;
		_cLen = _props.getNcols();

		Types.ValueType[] lschema = createOutputSchema(schema, _cLen);
		String[] lnames = createOutputNames(names, _cLen);

		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		// count rows in parallel per split
		try {
			ExecutorService pool = CommonThreadPool.get(_numThreads);
			if(_props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.Identity) {
				ArrayList<IOUtilFunctions.CountRowsTask> tasks = new ArrayList<>();
				for(InputSplit split : splits)
					tasks.add(new IOUtilFunctions.CountRowsTask(split, informat, job, false));

				// collect row counts for offset computation
				// early error notify in case not all tasks successful
				_offsets = new TemplateUtil.SplitOffsetInfos(tasks.size());
				int i = 0;
				for(Future<Long> rc : pool.invokeAll(tasks)) {
					int lnrow = (int) rc.get().longValue(); // incl error handling
					_offsets.setOffsetPerSplit(i, _rLen);
					_offsets.setLenghtPerSplit(i, lnrow);
					_rLen = _rLen + lnrow;
					i++;
				}
				pool.shutdown();
			}
			if(_props.getRowIndexStructure()
				.getProperties() == RowIndexStructure.IndexProperties.CellWiseExist || _props.getRowIndexStructure()
				.getProperties() == RowIndexStructure.IndexProperties.RowWiseExist) {
				ArrayList<IOUtilFunctions.CountRowsTask> tasks = new ArrayList<>();
				for(InputSplit split : splits)
					tasks.add(new IOUtilFunctions.CountRowsTask(split, informat, job, false));

				// collect row counts for offset computation
				// early error notify in case not all tasks successful
				_offsets = new TemplateUtil.SplitOffsetInfos(tasks.size());
				int i = 0;
				for(Future<Long> rc : pool.invokeAll(tasks)) {
					int lnrow = (int) rc.get().longValue(); // incl error handling
					_offsets.setOffsetPerSplit(i, _rLen);
					_offsets.setLenghtPerSplit(i, lnrow);
					i++;
				}
				pool.shutdown();
			}
			else if(_props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.SeqScatter) {
				_offsets = new TemplateUtil.SplitOffsetInfos(splits.length);
				for(int i = 0; i < splits.length; i++) {
					TemplateUtil.SplitInfo splitInfo = new TemplateUtil.SplitInfo();
					_offsets.setSeqOffsetPerSplit(i, splitInfo);
					_offsets.setOffsetPerSplit(i, 0);
				}

				ArrayList<CountSeqScatteredRowsTask> tasks = new ArrayList<>();
				int splitIndex = 0;
				for(InputSplit split : splits) {
					Integer nextOffset = splitIndex + 1 == splits.length ? null : splitIndex + 1;
					tasks.add(new CountSeqScatteredRowsTask(_offsets, splitIndex, nextOffset, split, informat, job,
						_props.getRowIndexStructure().getSeqBeginString(),
						_props.getRowIndexStructure().getSeqEndString()));
					splitIndex++;
				}

				// collect row counts for offset computation
				int i = 0;
				for(Future<Integer> rc : pool.invokeAll(tasks)) {
					Integer nrows = rc.get();
					_offsets.setOffsetPerSplit(i, _rLen);
					_rLen += nrows;
					i++;
				}
				pool.shutdown();
			}
		}
		catch(Exception e) {
			throw new IOException("Thread pool Error " + e.getMessage(), e);
		}

		// robustness for wrong dimensions which are already compiled into the plan
		if(rlen != -1 && _rLen != rlen) {
			String msg = "Read frame dimensions differ from meta data: [" + _rLen + "x" + _cLen + "] vs. [" + rlen + "x" + clen + "].";
			if(rlen < _rLen || clen < _cLen) {
				// a) specified matrix dimensions too small
				throw new DMLRuntimeException(msg);
			}
			else {
				// b) specified matrix dimensions too large -> padding and warning
				LOG.warn(msg);
				_rLen = (int) rlen;
				_cLen = (int) clen;
			}
		}
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, _rLen);
		return ret;
	}

	@Override public FrameBlock readFrameFromInputStream(InputStream is, Types.ValueType[] schema, String[] names,
		long rlen, long clen) throws IOException, DMLRuntimeException {

		// allocate output frame block
		InputStreamInputFormat informat = new InputStreamInputFormat(is);
		InputSplit[] splits = informat.getSplits(null, 1);
		FrameBlock ret = computeSizeAndCreateOutputFrameBlock(schema, names, splits, null, rlen, clen);
		// TODO: implement parallel reader for input stream
		return ret;
	}

	protected void readFrameFromHDFS(InputSplit[] splits, Path path, JobConf job, FrameBlock dest) throws IOException {

		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		ExecutorService pool = CommonThreadPool.get(_numThreads);
		try {
			// create read tasks for all splits
			ArrayList<ReadTask> tasks = new ArrayList<>();
			int splitCount = 0;
			for(InputSplit split : splits) {
				tasks.add(new ReadTask(split, informat, dest, splitCount++));
			}
			pool.invokeAll(tasks);
			pool.shutdown();

		}
		catch(Exception e) {
			throw new IOException("Threadpool issue, while parallel read.", e);
		}
	}

	private class ReadTask implements Callable<Long> {

		private final InputSplit _split;
		private final TextInputFormat _informat;
		private final FrameBlock _dest;
		private final int _splitCount;

		public ReadTask(InputSplit split, TextInputFormat informat, FrameBlock dest, int splitCount) {
			_split = split;
			_informat = informat;
			_dest = dest;
			_splitCount = splitCount;
		}

		@Override public Long call() throws IOException {
			RecordReader<LongWritable, Text> reader = _informat.getRecordReader(_split, job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			int row = _offsets.getOffsetPerSplit(_splitCount);
			TemplateUtil.SplitInfo _splitInfo = _offsets.getSeqOffsetPerSplit(_splitCount);
			readFrameFromHDFS(reader, key, value, _dest, row, _splitInfo);
			return 0L;
		}
	}

	private static class CountSeqScatteredRowsTask implements Callable<Integer> {
		private final TemplateUtil.SplitOffsetInfos _offsets;
		private final Integer _curOffset;
		private final Integer _nextOffset;
		private final InputSplit _split;
		private final TextInputFormat _inputFormat;
		private final JobConf _job;
		private final String _beginToken;
		private final String _endToken;

		public CountSeqScatteredRowsTask(TemplateUtil.SplitOffsetInfos offsets, Integer curOffset, Integer nextOffset,
			InputSplit split, TextInputFormat inputFormat, JobConf job, String beginToken, String endToken) {
			_offsets = offsets;
			_curOffset = curOffset;
			_nextOffset = nextOffset;
			_inputFormat = inputFormat;
			_split = split;
			_job = job;
			_beginToken = beginToken;
			_endToken = endToken;
		}

		@Override public Integer call() throws Exception {
			int nrows = 0;
			TemplateUtil.SplitInfo splitInfo = _offsets.getSeqOffsetPerSplit(_curOffset);

			long lastLineIndex;
			Pair<ArrayList<Pair<Long, Integer>>, Long> tokenPair =  TemplateUtil.getTokenIndexOnMultiLineRecords(
				_split, _inputFormat, _job, _endToken);

			ArrayList<Pair<Long, Integer>> beginIndexes = tokenPair.getKey();
			lastLineIndex = tokenPair.getValue();

			ArrayList<Pair<Long, Integer>> endIndexes;
			int tokenLength = 0;
			if(!_beginToken.equals(_endToken)) {
				endIndexes = TemplateUtil.getTokenIndexOnMultiLineRecords(_split, _inputFormat, _job, _endToken).getKey();
				tokenLength = _endToken.length();
				lastLineIndex = -1;
			}
			else {
				endIndexes = new ArrayList<>();
				for(int i = 1; i < beginIndexes.size(); i++)
					endIndexes.add(beginIndexes.get(i));
			}
			int i = 0;
			int j = 0;
			if(endIndexes.size() > 0) {
				if(beginIndexes.get(0).getKey() > endIndexes.get(0).getKey()) {
					nrows++;
					for(; j < endIndexes.size() && beginIndexes.get(0).getKey() > endIndexes.get(j).getKey(); j++)
						;
				}
				else if(_curOffset != 0 && _beginToken.equals(_endToken)) {
//					splitInfo.addIndexAndPosition(0l, endIndexes.get(0).getKey(), 0,
//						endIndexes.get(0).getValue() + tokenLength);
					//System.out.println(_curOffset+" || ["+0+","+endIndexes.get(0).getKey()+"]");
					nrows++;
				}
			}

			while(i < beginIndexes.size() && j < endIndexes.size()) {
				Pair<Long, Integer> p1 = beginIndexes.get(i);
				Pair<Long, Integer> p2 = endIndexes.get(j);
				int n = 0;
				while(p1.getKey() < p2.getKey() || (p1.getKey() == p2.getKey() && p1.getValue() < p2.getValue())) {
					n++;
					i++;
					if(i == beginIndexes.size()) {
						break;
					}
					p1 = beginIndexes.get(i);
				}
				j += n - 1;
				splitInfo.addIndexAndPosition(beginIndexes.get(i - n).getKey(), endIndexes.get(j).getKey(),
					beginIndexes.get(i - n).getValue(), endIndexes.get(j).getValue() + tokenLength);
				j++;
				nrows++;
			}
			if(_nextOffset != null) {
				if(beginIndexes.size() == 1){
					splitInfo.addIndexAndPosition( beginIndexes.get(0).getKey(), beginIndexes.get(0).getKey(),
						0, beginIndexes.get(0).getValue());
					nrows++;
				}
				RecordReader<LongWritable, Text> reader = _inputFormat.getRecordReader(_split, _job, Reporter.NULL);
				LongWritable key = new LongWritable();
				Text value = new Text();
				StringBuilder sb = new StringBuilder();
				for(long ri = 0; ri < beginIndexes.get(beginIndexes.size() - 1).getKey(); ri++) {
					reader.next(key, value);
				}
				if(reader.next(key, value)) {
					String strVar = value.toString();
					sb.append(strVar.substring(beginIndexes.get(beginIndexes.size() - 1).getValue()));
					while(reader.next(key, value)) {
						sb.append(value.toString());
					}
					_offsets.getSeqOffsetPerSplit(_nextOffset).setRemainString(sb.toString());
				}
			}
			else if(lastLineIndex !=-1) {
				splitInfo.addIndexAndPosition(endIndexes.get(endIndexes.size()-1).getKey(), lastLineIndex,
					endIndexes.get(endIndexes.size()-1).getValue(), 0);
				nrows++;
			}

			splitInfo.setNrows(nrows);
			_offsets.getSeqOffsetPerSplit(_curOffset).setNrows(nrows);
			_offsets.setOffsetPerSplit(_curOffset, nrows);

			return nrows;
		}
	}

	protected abstract int readFrameFromHDFS(RecordReader<LongWritable, Text> reader, LongWritable key, Text value,
		FrameBlock dest, int rowPos, TemplateUtil.SplitInfo splitInfo) throws IOException;
}

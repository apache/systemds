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
import org.apache.sysds.runtime.matrix.data.FrameBlock;
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

	@Override public FrameBlock readFrameFromHDFS(String fname, Types.ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {

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

	private FrameBlock computeSizeAndCreateOutputFrameBlock(Types.ValueType[] schema, String[] names, InputSplit[] splits, Path path, long rlen,
		long clen) throws IOException, DMLRuntimeException {
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
			if(_props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.CellWiseExist ||
				_props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.RowWiseExist) {
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
				ArrayList<CountSeqScatteredRowsTask> tasks = new ArrayList<>();
				for(InputSplit split : splits)
					tasks.add(new CountSeqScatteredRowsTask(split, informat, job, _props.getRowIndexStructure().getSeqBeginString(),
						_props.getRowIndexStructure().getSeqEndString()));

				// collect row counts for offset computation
				// early error notify in case not all tasks successful
				_offsets = new TemplateUtil.SplitOffsetInfos(tasks.size());
				int i = 0;
				for(Future<TemplateUtil.SplitInfo> rc : pool.invokeAll(tasks)) {
					TemplateUtil.SplitInfo splitInfo = rc.get();
					_offsets.setSeqOffsetPerSplit(i, splitInfo);
					_offsets.setOffsetPerSplit(i, _rLen);
					_rLen = _rLen + splitInfo.getNrows();
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

	@Override public FrameBlock readFrameFromInputStream(InputStream is, Types.ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {

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

		@Override
		public Long call() throws IOException {
			RecordReader<LongWritable, Text> reader = _informat.getRecordReader(_split, job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			int row = _offsets.getOffsetPerSplit(_splitCount);
			TemplateUtil.SplitInfo _splitInfo = _offsets.getSeqOffsetPerSplit(_splitCount);
			readFrameFromHDFS(reader, key, value, _dest, row, _splitInfo);
			return 0L;
		}
	}

	private static class CountSeqScatteredRowsTask implements Callable<TemplateUtil.SplitInfo> {
		private final InputSplit _split;
		private final TextInputFormat _inputFormat;
		private final JobConf _jobConf;
		private final String _beginString;
		private final String _endString;

		public CountSeqScatteredRowsTask(InputSplit split, TextInputFormat inputFormat, JobConf jobConf, String beginString, String endString) {
			_split = split;
			_inputFormat = inputFormat;
			_jobConf = jobConf;
			_beginString = beginString;
			_endString = endString;
		}

		@Override
		public TemplateUtil.SplitInfo call() throws Exception {
			TemplateUtil.SplitInfo splitInfo = new TemplateUtil.SplitInfo();
			int nrows = 0;
			ArrayList<Pair<Integer, Integer>> beginIndexes = TemplateUtil.getTokenIndexOnMultiLineRecords(_split, _inputFormat, _jobConf,
				_beginString);
			ArrayList<Pair<Integer, Integer>> endIndexes;
			int tokenLength = 0;
			boolean diffBeginEndToken = false;
			if(!_beginString.equals(_endString)) {
				endIndexes = TemplateUtil.getTokenIndexOnMultiLineRecords(_split, _inputFormat, _jobConf, _endString);
				tokenLength = _endString.length();
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
				RecordReader<LongWritable, Text> reader = _inputFormat.getRecordReader(_split, _jobConf, Reporter.NULL);
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
			return splitInfo;
		}
	}

	protected abstract int readFrameFromHDFS(RecordReader<LongWritable, Text> reader, LongWritable key, Text value, FrameBlock dest, int rowPos,
		TemplateUtil.SplitInfo splitInfo) throws IOException;
}

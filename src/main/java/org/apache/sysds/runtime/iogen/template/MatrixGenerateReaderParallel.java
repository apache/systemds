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
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.RowIndexStructure;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public abstract class MatrixGenerateReaderParallel extends MatrixReader {

	protected static CustomProperties _props;
	protected int _numThreads = 1;
	protected JobConf job;
	protected SplitOffsetInfos _offsets;
	protected int _rLen;
	protected int _cLen;

	public MatrixGenerateReaderParallel(CustomProperties _props) {
		_numThreads = OptimizerUtils.getParallelTextReadParallelism();
		MatrixGenerateReaderParallel._props = _props;
	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz) throws IOException, DMLRuntimeException {

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

		//allocate output matrix block
		MatrixBlock ret = computeSizeAndCreateOutputMatrixBlock(splits, path, rlen, _props.getNcols(), blen, estnnz);

		// Second Read Pass (read, parse strings, append to matrix block)
		readMatrixFromHDFS(splits, path, job, ret, rlen, clen, blen);

		return ret;
	}

	private MatrixBlock computeSizeAndCreateOutputMatrixBlock(InputSplit[] splits, Path path, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {
		_rLen = 0;
		_cLen = _props.getNcols();

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
				_offsets = new SplitOffsetInfos(tasks.size());
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
			else if(_props.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.SeqScatter) {
				ArrayList<CountSeqScatteredRowsTask> tasks = new ArrayList<>();
				for(InputSplit split : splits)
					tasks.add(new CountSeqScatteredRowsTask(split, informat, job, _props.getRowIndexStructure().getSeqBeginString(),
						_props.getRowIndexStructure().getSeqEndString()));

				// collect row counts for offset computation
				// early error notify in case not all tasks successful
				_offsets = new SplitOffsetInfos(tasks.size());
				int i = 0;
				for(Future<SplitInfo> rc : pool.invokeAll(tasks)) {
					SplitInfo splitInfo = rc.get();
					_offsets.setSeqOffsetPerSplit(i, splitInfo);
					_offsets.setOffsetPerSplit(i, _rLen);
					_rLen = _rLen + splitInfo.nrows;
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
			String msg = "Read matrix dimensions differ from meta data: [" + _rLen + "x" + _cLen + "] vs. [" + rlen+ "x" + clen + "].";
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

		// allocate target matrix block based on given size;
		// need to allocate sparse as well since lock-free insert into target
		long estnnz2 = (estnnz < 0) ? (long) _rLen * _cLen : estnnz;
		return createOutputMatrixBlock(_rLen, _cLen, blen, estnnz2, !_props.isSparse(), _props.isSparse());
	}

	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {

		MatrixBlock ret = null;
		if(rlen >= 0 && clen >= 0) //otherwise allocated on read
			ret = createOutputMatrixBlock(rlen, clen, (int) rlen, estnnz, true, false);

		return ret;
	}

	private void readMatrixFromHDFS(InputSplit[] splits, Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int blen) throws IOException
	{
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		ExecutorService pool = CommonThreadPool.get(_numThreads);
		try{
			// create read tasks for all splits
			ArrayList<ReadTask> tasks = new ArrayList<>();
			int splitCount = 0;
			for (InputSplit split : splits) {
				tasks.add( new ReadTask(split, informat, dest, splitCount++) );
			}
			pool.invokeAll(tasks);
			pool.shutdown();

			// check return codes and aggregate nnz
			long lnnz = 0;
			for (ReadTask rt : tasks)
				lnnz += rt.getNnz();
			dest.setNonZeros(lnnz);
		}
		catch (Exception e) {
			throw new IOException("Threadpool issue, while parallel read.", e);
		}
	}

	private static class SplitOffsetInfos {
		// offset & length info per split
		private int[] offsetPerSplit = null;
		private int[] lenghtPerSplit = null;
		private SplitInfo[] seqOffsetPerSplit = null;

		public SplitOffsetInfos(int numSplits) {
			lenghtPerSplit = new int[numSplits];
			offsetPerSplit = new int[numSplits];
			seqOffsetPerSplit = new SplitInfo[numSplits];
		}

		public int getLenghtPerSplit(int split) {
			return lenghtPerSplit[split];
		}

		public void setLenghtPerSplit(int split, int r) {
			lenghtPerSplit[split] = r;
		}

		public int getOffsetPerSplit(int split) {
			return offsetPerSplit[split];
		}

		public void setOffsetPerSplit(int split, int o) {
			offsetPerSplit[split] = o;
		}

		public SplitInfo getSeqOffsetPerSplit(int split) {
			return seqOffsetPerSplit[split];
		}

		public void setSeqOffsetPerSplit(int split, SplitInfo splitInfo) {
			seqOffsetPerSplit[split] = splitInfo;
		}
	}

	private class ReadTask implements Callable<Long> {

		private final InputSplit _split;
		private final TextInputFormat _informat;
		private final MatrixBlock _dest;
		private final int _splitCount;
		private int _row = 0;
		private long _nnz = 0;

		public ReadTask(InputSplit split, TextInputFormat informat, MatrixBlock dest, int splitCount) {
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
			_row = _offsets.getOffsetPerSplit(_splitCount);
			SplitInfo _splitInfo = _offsets.getSeqOffsetPerSplit(_splitCount);
			_nnz = readMatrixFromHDFS(reader, key, value, _dest, _row, _splitInfo);
			return _nnz;
		}

		public long getNnz() {
			return _nnz;
		}
	}

	private static class CountSeqScatteredRowsTask implements Callable<SplitInfo> {
		private final InputSplit _split;
		private final TextInputFormat _inputFormat;
		private final JobConf _jobConf;
		private final String _beginString;
		private final String _endString;

		public CountSeqScatteredRowsTask(InputSplit split, TextInputFormat inputFormat, JobConf jobConf, String beginString, String endString){
			_split = split;
			_inputFormat = inputFormat;
			_jobConf = jobConf;
			_beginString = beginString;
			_endString = endString;
		}

		@Override
		public SplitInfo call() throws Exception {
			SplitInfo splitInfo = new SplitInfo();
			int nrows = 0;
			ArrayList<Pair<Integer, Integer>> beginIndexes = getTokenIndexOnMultiLineRecords(_split, _inputFormat, _jobConf, _beginString);
			ArrayList<Pair<Integer, Integer>> endIndexes;
			int tokenLength = 0;
			if(!_beginString.equals(_endString)) {
				endIndexes = getTokenIndexOnMultiLineRecords(_split, _inputFormat, _jobConf, _endString);
				tokenLength = _endString.length();
			}
			else {
				endIndexes = new ArrayList<>();
				for(int i = 1; i < beginIndexes.size(); i++)
					endIndexes.add(beginIndexes.get(i));
			}

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
				j += n-1;
				splitInfo.addIndexAndPosition(beginIndexes.get(i - n).getKey(), endIndexes.get(j).getKey(), beginIndexes.get(i - n).getValue(),
					endIndexes.get(j).getValue()+tokenLength);
				j++;
				nrows++;
			}
			if(i == beginIndexes.size() && j < endIndexes.size())
				nrows++;
			if(beginIndexes.get(0).getKey() == 0 && beginIndexes.get(0).getValue() == 0)
				splitInfo.setRemainString("");
			else{
				RecordReader<LongWritable, Text> reader = _inputFormat.getRecordReader(_split, _jobConf, Reporter.NULL);
				LongWritable key = new LongWritable();
				Text value = new Text();

				StringBuilder sb = new StringBuilder();
				for(int ri = 0; ri< beginIndexes.get(0).getKey(); ri++){
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

	protected static class SplitInfo{
		private int nrows;
		private ArrayList<Integer> recordIndexBegin;
		private ArrayList<Integer> recordIndexEnd;
		private ArrayList<Integer> recordPositionBegin;
		private ArrayList<Integer> recordPositionEnd;
		private String remainString;

		public SplitInfo() {
			recordIndexBegin = new ArrayList<>();
			recordIndexEnd = new ArrayList<>();
			recordPositionBegin = new ArrayList<>();
			recordPositionEnd = new ArrayList<>();
		}

		public void addIndexAndPosition(int beginIndex, int endIndex, int beginPos, int endPos){
			recordIndexBegin.add(beginIndex);
			recordIndexEnd.add(endIndex);
			recordPositionBegin.add(beginPos);
			recordPositionEnd.add(endPos);
		}

		public int getNrows() {
			return nrows;
		}

		public void setNrows(int nrows) {
			this.nrows = nrows;
		}

		public String getRemainString() {
			return remainString;
		}

		public void setRemainString(String remainString) {
			this.remainString = remainString;
		}

		public int getRecordIndexBegin(int index) {
			return recordIndexBegin.get(index);
		}

		public int getRecordIndexEnd(int index) {
			return recordIndexEnd.get(index);
		}

		public int getRecordPositionBegin(int index) {
			return recordPositionBegin.get(index);
		}

		public int getRecordPositionEnd(int index) {
			return recordPositionEnd.get(index);
		}
	}

	private static ArrayList<Pair<Integer, Integer>> getTokenIndexOnMultiLineRecords(InputSplit split, TextInputFormat inputFormat, JobConf job,
		String token) throws IOException {
		RecordReader<LongWritable, Text> reader = inputFormat.getRecordReader(split, job, Reporter.NULL);
		LongWritable key = new LongWritable();
		Text value = new Text();
		ArrayList<Pair<Integer, Integer>> result = new ArrayList<>();

		int ri = 0;
		while (reader.next(key, value)){
			String raw = value.toString();
			int index;
			int fromIndex = 0;
			do {
				index = raw.indexOf(token, fromIndex);
				if(index !=-1){
					result.add(new Pair<>(ri, index));
					fromIndex = index+token.length();
				}
				else
					break;
			}while(true);
			ri++;
		}
		return result;
	}

	protected abstract long readMatrixFromHDFS(RecordReader<LongWritable, Text> reader, LongWritable key, Text value, MatrixBlock dest,
		int rowPos, SplitInfo splitInfo) throws IOException;

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

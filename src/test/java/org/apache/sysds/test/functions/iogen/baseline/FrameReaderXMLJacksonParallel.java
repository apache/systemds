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

package org.apache.sysds.test.functions.iogen.baseline;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
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
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.iogen.template.TemplateUtil;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import static org.apache.sysds.runtime.io.FrameReader.checkValidInputFile;
import static org.apache.sysds.runtime.io.FrameReader.createOutputFrameBlock;

public class FrameReaderXMLJacksonParallel extends FrameReaderXMLJackson {

	protected static final Log LOG = LogFactory.getLog(FrameReaderXMLJacksonParallel.class.getName());
	protected int _numThreads;
	protected JobConf job;
	protected TemplateUtil.SplitOffsetInfos _offsets;
	protected int _rLen;
	protected int _cLen;

	public FrameReaderXMLJacksonParallel() {
		this._numThreads = OptimizerUtils.getParallelTextReadParallelism();
	}

	@Override public FrameBlock readFrameFromHDFS(String fname, Types.ValueType[] schema,
		Map<String, Integer> schemaMap, String beginToken, String endToken, long rlen, long clen)
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

		String[] names = createOutputNamesFromSchemaMap(schemaMap);
		// allocate output frame block
		FrameBlock ret = computeSizeAndCreateOutputFrameBlock(informat, job, schema, names, splits, beginToken, endToken);

		// core read (sequential/parallel)
		readXMLFrameFromHDFS(splits, informat, job, schema, schemaMap, ret);
		return ret;
	}

	protected void readXMLFrameFromHDFS(InputSplit[] splits, TextInputFormat informat, JobConf jobConf,
		Types.ValueType[] schema, Map<String, Integer> schemaMap, FrameBlock dest) throws IOException {

		ExecutorService pool = CommonThreadPool.get(_numThreads);
		try {
			// create read tasks for all splits
			ArrayList<ReadTask> tasks = new ArrayList<>();
			int splitCount = 0;
			for(InputSplit split : splits) {
				tasks.add(new ReadTask(split, informat, dest, splitCount++, schema, schemaMap));
			}
			pool.invokeAll(tasks);
			pool.shutdown();

		}
		catch(Exception e) {
			throw new IOException("Threadpool issue, while parallel read.", e);
		}
	}

	@Override protected FrameBlock computeSizeAndCreateOutputFrameBlock(TextInputFormat informat, JobConf job,
		Types.ValueType[] schema, String[] names, InputSplit[] splits, String beginToken, String endToken)
		throws IOException, DMLRuntimeException {
		_rLen = 0;
		_cLen = names.length;

		// count rows in parallel per split
		try {
			ExecutorService pool = CommonThreadPool.get(_numThreads);

			_offsets = new TemplateUtil.SplitOffsetInfos(splits.length);
			for(int i = 0; i < splits.length; i++) {
				TemplateUtil.SplitInfo splitInfo = new TemplateUtil.SplitInfo();
				_offsets.setSeqOffsetPerSplit(i, splitInfo);
				_offsets.setOffsetPerSplit(i, 0);
			}

			ArrayList<CountRowsTask> tasks = new ArrayList<>();
			int splitIndex = 0;
			for(InputSplit split : splits) {
				Integer nextOffset = splitIndex + 1 == splits.length ? null : splitIndex + 1;
				tasks.add(new CountRowsTask(_offsets, splitIndex, nextOffset, split, informat, job, beginToken, endToken));
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
		catch(Exception e) {
			throw new IOException("Thread pool Error " + e.getMessage(), e);
		}
		FrameBlock ret = createOutputFrameBlock(schema, names, _rLen);
		return ret;
	}

	private static class CountRowsTask implements Callable<Integer> {
		private final TemplateUtil.SplitOffsetInfos _offsets;
		private final Integer _curOffset;
		private final Integer _nextOffset;
		private final InputSplit _split;
		private final TextInputFormat _inputFormat;
		private final JobConf _job;
		private final String _beginToken;
		private final String _endToken;

		public CountRowsTask(TemplateUtil.SplitOffsetInfos offsets, Integer curOffset, Integer nextOffset,
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

			ArrayList<Pair<Long, Integer>> beginIndexes = TemplateUtil.getTokenIndexOnMultiLineRecords(_split,
				_inputFormat, _job, _beginToken).getKey();
			ArrayList<Pair<Long, Integer>> endIndexes = TemplateUtil.getTokenIndexOnMultiLineRecords(_split,
				_inputFormat, _job, _endToken).getKey();
			int tokenLength = _endToken.length();

			int i = 0;
			int j = 0;

			if(beginIndexes.get(0).getKey() > endIndexes.get(0).getKey()) {
				nrows++;
				for(; j < endIndexes.size() && beginIndexes.get(0).getKey() > endIndexes.get(j).getKey(); j++)
					;
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
				_offsets.getSeqOffsetPerSplit(_curOffset)
					.addIndexAndPosition(beginIndexes.get(i - n).getKey(), endIndexes.get(j).getKey(),
						beginIndexes.get(i - n).getValue(), endIndexes.get(j).getValue() + tokenLength);
				j++;
				nrows++;
			}
			if(_nextOffset != null) {
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
			_offsets.getSeqOffsetPerSplit(_curOffset).setNrows(nrows);
			_offsets.setOffsetPerSplit(_curOffset, nrows);

			return nrows;
		}
	}

	private class ReadTask implements Callable<Long> {

		private final InputSplit _split;
		private final TextInputFormat _informat;
		private final FrameBlock _dest;
		private final int _splitCount;
		private final Types.ValueType[] _schema;
		private final Map<String, Integer> _schemaMap;

		public ReadTask(InputSplit split, TextInputFormat informat, FrameBlock dest, int splitCount,
			Types.ValueType[] schema, Map<String, Integer> schemaMap) {
			_split = split;
			_informat = informat;
			_dest = dest;
			_splitCount = splitCount;
			_schema = schema;
			_schemaMap = schemaMap;
		}

		@Override public Long call() throws IOException {
			RecordReader<LongWritable, Text> reader = _informat.getRecordReader(_split, job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			int row = _offsets.getOffsetPerSplit(_splitCount);
			TemplateUtil.SplitInfo _splitInfo = _offsets.getSeqOffsetPerSplit(_splitCount);
			readXMLFrameFromInputSplit(reader, _splitInfo, key, value, row, _schema, _schemaMap, _dest);
			return 0L;
		}
	}
}

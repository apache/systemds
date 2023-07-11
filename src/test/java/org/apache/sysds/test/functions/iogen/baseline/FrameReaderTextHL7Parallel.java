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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.iogen.template.TemplateUtil;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

/**
 * Multi-threaded frame text HL7 reader.
 */
public class FrameReaderTextHL7Parallel extends FrameReaderTextHL7 {
	public FrameReaderTextHL7Parallel(FileFormatPropertiesHL7 props) {
		super(props);
		this._numThreads = OptimizerUtils.getParallelTextReadParallelism();
	}

	@Override
	public FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {

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
		String[] lnames = createOutputNames(names, clen);
		FrameBlock ret = computeSizeAndCreateOutputFrameBlock(informat, job, schema, lnames, splits, "MSH|");

		readHL7FrameFromHDFS(splits, informat, job, schema, ret);

		return ret;
	}

	protected void readHL7FrameFromHDFS(InputSplit[] splits, TextInputFormat informat, JobConf jobConf,
		Types.ValueType[] schema, FrameBlock dest) throws IOException {

		ExecutorService pool = CommonThreadPool.get(_numThreads);
		try {
			// create read tasks for all splits
			ArrayList<ReadTask> tasks = new ArrayList<>();
			int splitCount = 0;
			for(InputSplit split : splits) {
				tasks.add(new ReadTask(split, informat, dest, splitCount++, schema));
			}
			pool.invokeAll(tasks);
			pool.shutdown();

		}
		catch(Exception e) {
			throw new IOException("Threadpool issue, while parallel read.", e);
		}
	}

	@Override
	protected FrameBlock computeSizeAndCreateOutputFrameBlock(TextInputFormat informat, JobConf job,
		Types.ValueType[] schema, String[] names, InputSplit[] splits, String beginToken)
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
				tasks.add(new CountRowsTask(_offsets, splitIndex, nextOffset, split, informat, job, beginToken));
				splitIndex++;
			}

			// collect row counts for offset computation
			int i = 0;
			for(Future<Integer> rc : pool.invokeAll(tasks)) {
				Integer nrows = rc.get();
				_offsets.setOffsetPerSplit(i, _rLen);
				_offsets.setLenghtPerSplit(i, _rLen+nrows);
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

	private class ReadTask implements Callable<Long> {

		private final InputSplit _split;
		private final TextInputFormat _informat;
		private final FrameBlock _dest;
		private final int _splitCount;
		private final Types.ValueType[] _schema;

		public ReadTask(InputSplit split, TextInputFormat informat, FrameBlock dest, int splitCount,
			Types.ValueType[] schema) {
			_split = split;
			_informat = informat;
			_dest = dest;
			_splitCount = splitCount;
			_schema = schema;
		}

		@Override
		public Long call() throws IOException {
			readHL7FrameFromInputSplit(_informat, _split, _splitCount, _schema, _dest);
			return 0L;
		}
	}

	protected static class CountRowsTask implements Callable<Integer> {
		private final TemplateUtil.SplitOffsetInfos _offsets;
		private final Integer _curOffset;
		private final Integer _nextOffset;
		private final InputSplit _split;
		private final TextInputFormat _inputFormat;
		private final JobConf _job;
		private final String _beginToken;

		public CountRowsTask(TemplateUtil.SplitOffsetInfos offsets, Integer curOffset, Integer nextOffset,
			InputSplit split, TextInputFormat inputFormat, JobConf job, String beginToken) {
			_offsets = offsets;
			_curOffset = curOffset;
			_nextOffset = nextOffset;
			_inputFormat = inputFormat;
			_split = split;
			_job = job;
			_beginToken = beginToken;
		}

		@Override
		public Integer call() throws Exception {
			return countRows(_offsets, _curOffset, _nextOffset, _split, _inputFormat, _job, _beginToken);
		}
	}
}

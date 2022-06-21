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
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

/**
 * Multi-threaded frame text AMiner reader.
 */
public class FrameReaderTextAMinerParallel extends FrameReaderTextAMiner {
	protected int _numThreads;

	public FrameReaderTextAMinerParallel(FileFormatPropertiesAMiner props) {
		super(props);
		this._numThreads = OptimizerUtils.getParallelTextReadParallelism();
	}

	@Override public FrameBlock readFrameFromHDFS(String fname, Types.ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {

		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		InputSplit[] splits = informat.getSplits(job, _numThreads);
		splits = IOUtilFunctions.sortInputSplits(splits);

		// check existence and non-empty file
		checkValidInputFile(fs, path);

		FrameBlock ret = readAMinerHDFS(splits, informat, job);

		return ret;
	}

	protected FrameBlock readAMinerHDFS(InputSplit[] splits, TextInputFormat informat, JobConf job) throws IOException {
		try {
			ExecutorService pool = CommonThreadPool.get(Math.min(_numThreads, splits.length));
			this.rowIndexs = new ArrayList[splits.length];
			this.colBeginIndexs = new ArrayList[splits.length];

			//compute num rows per split
			ArrayList<CountRowsColsTask> tasks = new ArrayList<>();
			for(int i = 0; i < splits.length; i++) {
				rowIndexs[i] = new ArrayList<>();
				if(_props.getType().equals("author"))
					tasks.add(new CountRowsColsTaskAuthor(splits[i], informat, job, rowIndexs[i], i));
				else {
					colBeginIndexs[i] = new ArrayList<>();
					tasks.add(new CountRowsColsTaskPaper(splits[i], informat, job, rowIndexs[i], colBeginIndexs[i], i));
				}
			}
			List<Future<DatasetMetaData>> cret = pool.invokeAll(tasks);

			for(Future<DatasetMetaData> count : cret)
				while(!count.isDone())
					;

			//compute row offset per split via cumsum on row counts
			int offset = 0;
			int maxAffiliation = 0;
			int maxResearchInterest = 0;
			int maxReference = 0;
			int maxAuthor = 0;
			int ncol;
			ValueType[] lschema = null;
			String[] lnames = null;

			for(Future<DatasetMetaData> count : cret) {
				DatasetMetaData metaData = count.get();
				ArrayList<Integer> ri = rowIndexs[metaData.getIndex()];
				if(_props.getType().equals("author")) {
					maxResearchInterest = Math.max(maxResearchInterest, metaData.maxResearchInterest);
				}
				else {
					int negativeBeginPos = -1;
					int negativeEndPos = -1;
					for(int i = 0; i < ri.size(); i++) {
						int valIndex = ri.get(i);
						if(valIndex == -1) {
							if(negativeBeginPos == -1) {
								negativeBeginPos = i;
							}
							negativeEndPos = i;
						}
					}
					if(negativeBeginPos != -1) {
						int bcIndex = colBeginIndexs[metaData.getIndex() - 1].get(colBeginIndexs[metaData.getIndex() - 1].size() - 1);
						for(int i = negativeBeginPos; i <= negativeEndPos; i++) {
							colBeginIndexs[metaData.getIndex()].set(i, i - negativeBeginPos + bcIndex + 1);
						}
						int tMax = Math.max(bcIndex + negativeEndPos - negativeBeginPos + 1, metaData.maxReference);
						metaData.setMaxReference(tMax);
					}
					maxReference = Math.max(maxReference, metaData.maxReference);
					maxAuthor = Math.max(maxAuthor, metaData.maxAuthor);
				}

				for(int i = 0; i < ri.size(); i++)
					ri.set(i, ri.get(i) + offset);

				maxAffiliation = Math.max(maxAffiliation, metaData.maxAffiliation);
				offset += metaData.getNrow();
			}
			if(_props.getType().equals("paper")) {
				ncol = 5 + maxAuthor + maxAffiliation + maxReference;
				this.paperMetaData = new DatasetMetaDataPaper(ncol, offset, maxAuthor, maxAffiliation, maxReference);
				lschema = this.paperMetaData.schema;
				lnames = this.paperMetaData.names;
			}
			else {
				ncol = 7 + maxAffiliation + maxResearchInterest;
				this.authorMetaData = new DatasetMetaDataAuthor(ncol, offset, maxAffiliation, maxResearchInterest);
				lschema = this.authorMetaData.schema;
				lnames = this.authorMetaData.names;
			}
			FrameBlock ret = createOutputFrameBlock(lschema, lnames, offset + 1);
			//read individual splits
			ArrayList<ReadRowsTask> tasks2 = new ArrayList<>();
			for(int i = 0; i < splits.length; i++)
				tasks2.add(new ReadRowsTask(splits[i], rowIndexs[i], colBeginIndexs[i], informat, job, ret, lschema));
			CommonThreadPool.invokeAndShutdown(pool, tasks2);
			return ret;
		}
		catch(Exception e) {
			throw new IOException("Failed parallel read of text AMiner input.", e);
		}
	}

	private static abstract class CountRowsColsTask implements Callable<DatasetMetaData> {
		protected InputSplit _split = null;
		protected Integer _splitIndex;
		protected TextInputFormat _informat = null;
		protected JobConf _job = null;
		protected ArrayList<Integer> _rowIndex;
		protected ArrayList<Integer> _colBeginIndex;

		public CountRowsColsTask(InputSplit split, TextInputFormat informat, JobConf job, ArrayList<Integer> rowIndex,
			ArrayList<Integer> colBeginIndex, int splitIndex) {
			_split = split;
			_informat = informat;
			_job = job;
			_rowIndex = rowIndex;
			_colBeginIndex = colBeginIndex;
			_splitIndex = splitIndex;
		}

		@Override public DatasetMetaData call() throws Exception {
			return null;
		}
	}

	private static class CountRowsColsTaskAuthor extends CountRowsColsTask {

		public CountRowsColsTaskAuthor(InputSplit split, TextInputFormat informat, JobConf job, ArrayList<Integer> rowIndex, int splitIndex) {
			super(split, informat, job, rowIndex, null, splitIndex);
		}

		@Override public DatasetMetaDataAuthor call() throws Exception {
			RecordReader<LongWritable, Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();

			int ncol = 7;
			int maxAffiliations = 0;
			int maxResearchInterest = 0;
			int row = -1;

			while(reader.next(key, value)) {
				String raw = value.toString().trim();
				if(raw.startsWith("#index "))
					row++;
				else if(raw.startsWith("#a")) //affiliations  (separated by semicolons)
					maxAffiliations = Math.max(maxAffiliations, IOUtilFunctions.countTokensCSV(raw, ";"));

				else if(raw.startsWith("#t")) // research interests of this author  (separated by semicolons)
					maxResearchInterest = Math.max(maxResearchInterest, IOUtilFunctions.countTokensCSV(raw, ";"));

				this._rowIndex.add(row);
			}

			ncol += maxAffiliations + maxResearchInterest;

			DatasetMetaDataAuthor datasetMetaDataAuthor = new DatasetMetaDataAuthor(ncol, row + 1, maxAffiliations, maxResearchInterest);
			datasetMetaDataAuthor.setIndex(_splitIndex);
			return datasetMetaDataAuthor;
		}
	}

	private static class CountRowsColsTaskPaper extends CountRowsColsTask {

		public CountRowsColsTaskPaper(InputSplit split, TextInputFormat informat, JobConf job, ArrayList<Integer> rowIndex,
			ArrayList<Integer> colBeginIndex, int splitIndex) {
			super(split, informat, job, rowIndex, colBeginIndex, splitIndex);
		}

		@Override public DatasetMetaDataPaper call() throws Exception {
			RecordReader<LongWritable, Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			int ncol = 5;
			int maxAuthors = 0;
			int maxAffiliations = 0;
			int maxReferences = 0;
			int row = -1;
			int refCount = 0;

			while(reader.next(key, value)) {
				String raw = value.toString().trim();
				if(raw.startsWith("#index ")) {
					row++;
					maxReferences = Math.max(maxReferences, refCount);
					refCount = 0;
					this._colBeginIndex.add(0);
				}
				else if(raw.startsWith("#@")) { //authors (separated by semicolons)
					maxAuthors = Math.max(maxAuthors, IOUtilFunctions.countTokensCSV(raw, ";"));
					this._colBeginIndex.add(0);
				}
				else if(raw.startsWith("#o")) { //(separated by semicolons, and each affiliaiton corresponds to an author in order)
					maxAffiliations = Math.max(maxAffiliations, IOUtilFunctions.countTokensCSV(raw, ";"));
					this._colBeginIndex.add(0);
				}
				else if(raw.startsWith("#%")) { // the id of references of this paper (there are multiple lines, with each indicating a reference)
					this._colBeginIndex.add(refCount);
					refCount++;
				}
				else
					this._colBeginIndex.add(0);

				this._rowIndex.add(row);
			}

			ncol += maxAuthors + maxAffiliations + maxReferences;
			DatasetMetaDataPaper datasetMetaDataPaper = new DatasetMetaDataPaper(ncol, row + 1, maxAuthors, maxAffiliations, maxReferences);
			datasetMetaDataPaper.setIndex(_splitIndex);
			return datasetMetaDataPaper;
		}
	}

	private class ReadRowsTask implements Callable<Object> {
		private final InputSplit _split;
		private final TextInputFormat _informat;
		private final JobConf _job;
		private final FrameBlock _dest;
		private final ArrayList<Integer> _rowIndex;
		private final ArrayList<Integer> _colBeginIndex;
		private final ValueType[] _schema;

		public ReadRowsTask(InputSplit split, ArrayList<Integer> rowIndex, ArrayList<Integer> colBeginIndex, TextInputFormat informat, JobConf job,
			FrameBlock dest, ValueType[] schema) {
			_split = split;
			_informat = informat;
			_job = job;
			_dest = dest;
			_rowIndex = rowIndex;
			_colBeginIndex = colBeginIndex;
			_schema = schema;
		}

		@Override public Object call() throws Exception {
			if(_props.getType().equals("paper"))
				readAMinerPaperFrameFromInputSplit(_split, _rowIndex, _colBeginIndex, _informat, _job, _dest, _schema);
			else
				readAMinerAuthorFrameFromInputSplit(_split, _rowIndex, _informat, _job, _dest, _schema);
			return null;
		}
	}
}

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

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang.StringUtils;
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
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.io.IOUtilFunctions.CountRowsTask;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Parallel version of ReaderTextCSV.java. To summarize, we do two passes in order to compute row offsets and the actual
 * read. We accordingly create count and read tasks and use fixed-size thread pools to execute these tasks. If the
 * target matrix is dense, the inserts are done lock-free. In contrast to textcell parallel read, we also do lock-free
 * inserts. If the matrix is sparse, because splits contain row partitioned lines and hence there is no danger of lost
 * updates. Note, there is also no sorting of sparse rows required because data comes in sorted order per row.
 * 
 */
public class ReaderTextCSVParallel extends MatrixReader {
	final private int _numThreads;

	protected final FileFormatPropertiesCSV _props;
	protected SplitOffsetInfos _offsets;
	protected int _bLen;
	protected int _rLen;
	protected int _cLen;
	protected JobConf _job;

	public ReaderTextCSVParallel(FileFormatPropertiesCSV props) {
		_numThreads = OptimizerUtils.getParallelTextReadParallelism();
		_props = props;
	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {
		_bLen = blen;

		// prepare file access
		_job = new JobConf(ConfigurationManager.getCachedJobConf());

		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, _job);

		FileInputFormat.addInputPath(_job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(_job);

		InputSplit[] splits = informat.getSplits(_job, _numThreads);
		splits = IOUtilFunctions.sortInputSplits(splits);

		// check existence and non-empty file
		checkValidInputFile(fs, path);

		// allocate output matrix block
		// First Read Pass (count rows/cols, determine offsets, allocate matrix block)
		MatrixBlock ret = computeCSVSizeAndCreateOutputMatrixBlock(splits, path, rlen, clen, blen, estnnz);

		// Second Read Pass (read, parse strings, append to matrix block)
		readCSVMatrixFromHDFS(splits, path, ret);

		// post-processing (representation-specific, change of sparse/dense block representation)
		// - no sorting required for CSV because it is read in sorted order per row
		// - nnz explicitly maintained in parallel for the individual splits
		ret.examSparsity();

		// sanity check for parallel row count (since determined internally)
		if(rlen >= 0 && rlen != ret.getNumRows())
			throw new DMLRuntimeException("Read matrix inconsistent with given meta data: " + "expected nrow=" + rlen
				+ ", real nrow=" + ret.getNumRows());

		return ret;
	}

	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {
		// not implemented yet, fallback to sequential reader
		return new ReaderTextCSV(_props).readMatrixFromInputStream(is, rlen, clen, blen, estnnz);
	}

	private void readCSVMatrixFromHDFS(InputSplit[] splits, Path path, MatrixBlock dest) throws IOException {

		FileInputFormat.addInputPath(_job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(_job);

		ExecutorService pool = CommonThreadPool.get(_numThreads);

		try {
			// create read tasks for all splits
			ArrayList<Callable<Long>> tasks = new ArrayList<>();
			int splitCount = 0;
			for(InputSplit split : splits) {
				if(dest.isInSparseFormat() && _props.getNAStrings() != null)
					tasks.add(new CSVReadSparseNanTask(split, informat, dest, splitCount++));
				else if(dest.isInSparseFormat() && _props.getFillValue() == 0)
					tasks.add(new CSVReadSparseNoNanTaskAndFill(split, informat, dest, splitCount++));
				else if(dest.isInSparseFormat())
					tasks.add(new CSVReadSparseNoNanTask(split, informat, dest, splitCount++));
				else if(_props.getNAStrings() != null)
					tasks.add(new CSVReadDenseNanTask(split, informat, dest, splitCount++));
				else
					tasks.add(new CSVReadDenseNoNanTask(split, informat, dest, splitCount++));
			}

			// check return codes and aggregate nnz
			long lnnz = 0;
			for(Future<Long> rt : pool.invokeAll(tasks)) {
				lnnz += rt.get();
			}
			pool.shutdown();
			dest.setNonZeros(lnnz);
		}
		catch(Exception e) {
			throw new IOException("Thread pool issue, while parallel read.", e);
		}
	}

	private MatrixBlock computeCSVSizeAndCreateOutputMatrixBlock(InputSplit[] splits,
		Path path, long rlen, long clen, int blen, long estnnz) throws IOException, DMLRuntimeException {
		_rLen = 0;
		_cLen = 0;

		FileInputFormat.addInputPath(_job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(_job);

		// count number of entities in the first non-header row
		LongWritable key = new LongWritable();
		Text oneLine = new Text();
		RecordReader<LongWritable, Text> reader = informat.getRecordReader(splits[0], _job, Reporter.NULL);
		try {
			if(reader.next(key, oneLine)) {
				String cellStr = oneLine.toString().trim();
				_cLen = StringUtils.countMatches(cellStr, _props.getDelim()) + 1;
			}
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}

		// count rows in parallel per split
		try {
			ExecutorService pool = CommonThreadPool.get(_numThreads);
			ArrayList<CountRowsTask> tasks = new ArrayList<>();
			boolean hasHeader = _props.hasHeader();
			for(InputSplit split : splits) {
				tasks.add(new CountRowsTask(split, informat, _job, hasHeader));
				hasHeader = false;
			}

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
		catch(Exception e) {
			throw new IOException("Thread pool Error " + e.getMessage(), e);
		}

		// robustness for wrong dimensions which are already compiled into the plan
		if((rlen != -1 && _rLen != rlen) || (clen != -1 && _cLen != clen)) {
			String msg = "Read matrix dimensions differ from meta data: [" + _rLen + "x" + _cLen + "] vs. [" + rlen
				+ "x" + clen + "].";
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
		return createOutputMatrixBlock(_rLen, _cLen, blen, estnnz2, true, true);
	}

	private static class SplitOffsetInfos {
		// offset & length info per split
		private int[] offsetPerSplit = null;
		private int[] lenghtPerSplit = null;

		public SplitOffsetInfos(int numSplits) {
			lenghtPerSplit = new int[numSplits];
			offsetPerSplit = new int[numSplits];
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
	}

	private abstract class CSVReadTask implements Callable<Long> {
		protected final InputSplit _split;
		protected final TextInputFormat _informat;
		protected final MatrixBlock _dest;
		protected final boolean _isFirstSplit;
		protected final int _splitCount;

		protected int _row = 0;
		protected int _col = 0;

		public CSVReadTask(InputSplit split, TextInputFormat informat, MatrixBlock dest, int splitCount) {
			_split = split;
			_informat = informat;
			_dest = dest;
			_isFirstSplit = (splitCount == 0);
			_splitCount = splitCount;
		}

		@Override
		public Long call() throws Exception {

			try {
				RecordReader<LongWritable, Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
				LongWritable key = new LongWritable();
				Text value = new Text();

				// skip the header line
				if(_isFirstSplit && _props.hasHeader()) {
					reader.next(key, value);
				}

				_row = _offsets.getOffsetPerSplit(_splitCount);

				long nnz = 0;
				try {
					nnz = parse(reader, key, value);
					verifyRows(value);
				}
				finally {
					IOUtilFunctions.closeSilently(reader);
				}
				return nnz;
			}
			catch(Exception ex) {
				// post-mortem error handling and bounds checking
				if(_row < 0 || _row + 1 > _rLen || _col < 0 || _col + 1 > _cLen) {
					String errMsg = "CSV cell [" + (_row + 1) + "," + (_col + 1) + "] "
						+ "out of overall matrix range [1:" + _rLen + ",1:" + _cLen + "]. " + ex.getMessage();
					throw new IOException(errMsg, ex);
				}
				else {
					String errMsg = "Unable to read matrix in text CSV format. " + ex.getMessage();
					throw new IOException(errMsg, ex);
				}
			}

		}

		protected abstract long parse(RecordReader<LongWritable, Text> reader, LongWritable key, Text value)
			throws IOException;

		protected void verifyRows(Text value) throws IOException {
			if(_row != (_offsets.getOffsetPerSplit(_splitCount) + _offsets.getLenghtPerSplit(_splitCount))) {
				throw new IOException("Incorrect number of rows (" + _row + ") found in delimited file ("
					+ (_offsets.getOffsetPerSplit(_splitCount) + _offsets.getLenghtPerSplit(_splitCount)) + "): "
					+ value);
			}
		}
	}

	private class CSVReadDenseNoNanTask extends CSVReadTask {

		public CSVReadDenseNoNanTask(InputSplit split, TextInputFormat informat, MatrixBlock dest, int splitCount) {
			super(split, informat, dest, splitCount);
		}

		protected long parse(RecordReader<LongWritable, Text> reader, LongWritable key, Text value) throws IOException {
			DenseBlock a = _dest.getDenseBlock();
			double cellValue = 0;
			long nnz = 0;
			boolean noFillEmpty = false;

			while(reader.next(key, value)) { // foreach line
				final String cellStr = value.toString().trim();
				final String[] parts = IOUtilFunctions.split(cellStr, _props.getDelim());
				double[] avals = a.values(_row);
				int apos = a.pos(_row);
				for(int j = 0; j < _cLen; j++) { // foreach cell
					String part = parts[j].trim();
					if(part.isEmpty()) {
						noFillEmpty |= !_props.isFill();
						cellValue = _props.getFillValue();
					}
					else {
						cellValue = Double.parseDouble(part);
					}
					if(cellValue != 0) {
						avals[apos+j] = cellValue;
						nnz++;
					}
				}
				// sanity checks (number of columns, fill values)
				IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(cellStr, _props.isFill(), noFillEmpty);
				IOUtilFunctions.checkAndRaiseErrorCSVNumColumns(_split.toString(), cellStr, parts, _cLen);
				_row++;
			}

			return nnz;
		}

	}

	private class CSVReadDenseNanTask extends CSVReadTask {

		public CSVReadDenseNanTask(InputSplit split, TextInputFormat informat, MatrixBlock dest, int splitCount) {
			super(split, informat, dest, splitCount);
		}

		protected long parse(RecordReader<LongWritable, Text> reader, LongWritable key, Text value) throws IOException {
			DenseBlock a = _dest.getDenseBlock();
			double cellValue = 0;
			boolean noFillEmpty = false;
			long nnz = 0;
			while(reader.next(key, value)) { // foreach line
				String cellStr = value.toString().trim();
				String[] parts = IOUtilFunctions.split(cellStr, _props.getDelim());
				double[] avals = a.values(_row);
				int apos = a.pos(_row);
				for(int j = 0; j < _cLen; j++) { // foreach cell
					String part = parts[j].trim();
					if(part.isEmpty()) {
						noFillEmpty |= !_props.isFill();
						cellValue = _props.getFillValue();
					}
					else
						cellValue = UtilFunctions.parseToDouble(part, _props.getNAStrings());

					if(cellValue != 0) {
						avals[apos+j] = cellValue;
						nnz++;
					}
				}
				// sanity checks (number of columns, fill values)
				IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(cellStr, _props.isFill(), noFillEmpty);
				IOUtilFunctions.checkAndRaiseErrorCSVNumColumns(_split.toString(), cellStr, parts, _cLen);
				_row++;
			}
			return nnz;
		}
	}

	private class CSVReadSparseNanTask extends CSVReadTask {

		public CSVReadSparseNanTask(InputSplit split, TextInputFormat informat, MatrixBlock dest, int splitCount) {
			super(split, informat, dest, splitCount);
		}

		protected long parse(RecordReader<LongWritable, Text> reader, LongWritable key, Text value) throws IOException {
			boolean noFillEmpty = false;
			double cellValue = 0;
			final SparseBlock sb = _dest.getSparseBlock();
			long nnz = 0;
			while(reader.next(key, value)) {

				final String cellStr = value.toString().trim();
				final String[] parts = IOUtilFunctions.split(cellStr, _props.getDelim());
				_col = 0;
				sb.allocate(_row);
				SparseRow r = sb.get(_row);

				for(String part : parts) {
					part = part.trim();
					if(part.isEmpty()) {
						noFillEmpty |= !_props.isFill();
						cellValue = _props.getFillValue();
					}
					else {
						cellValue = UtilFunctions.parseToDouble(part, _props.getNAStrings());
					}

					if(cellValue != 0) {
						r.append(_col, cellValue);
						nnz++;
					}
					_col++;
				}

				// sanity checks (number of columns, fill values)
				IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(cellStr, _props.isFill(), noFillEmpty);
				IOUtilFunctions.checkAndRaiseErrorCSVNumColumns(_split.toString(), cellStr, parts, _cLen);

				_row++;
			}
			return nnz;
		}
	}

	private class CSVReadSparseNoNanTask extends CSVReadTask {
		public CSVReadSparseNoNanTask(InputSplit split, TextInputFormat informat, MatrixBlock dest, int splitCount) {
			super(split, informat, dest, splitCount);
		}

		protected long parse(RecordReader<LongWritable, Text> reader, LongWritable key, Text value) throws IOException {
			final SparseBlock sb = _dest.getSparseBlock();
			long nnz = 0;
			double cellValue = 0;
			boolean noFillEmpty = false;
			while(reader.next(key, value)) {
				_col = 0;
				final String cellStr = value.toString().trim();
				final String[] parts = IOUtilFunctions.split(cellStr, _props.getDelim());
				sb.allocate(_row);
				SparseRow r = sb.get(_row);
				for(String part : parts) {
					part = part.trim();
					if(part.isEmpty()) {
						noFillEmpty |= !_props.isFill();
						cellValue = _props.getFillValue();
					}
					else {
						cellValue = Double.parseDouble(part);
					}

					if(cellValue != 0) {
						r.append(_col, cellValue);
						nnz++;
					}
					_col++;
				}

				// sanity checks (number of columns, fill values)
				IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(cellStr, _props.isFill(), noFillEmpty);
				IOUtilFunctions.checkAndRaiseErrorCSVNumColumns(_split.toString(), cellStr, parts, _cLen);

				_row++;
			}
			return nnz;
		}
	}

	private class CSVReadSparseNoNanTaskAndFill extends CSVReadTask {
		public CSVReadSparseNoNanTaskAndFill(InputSplit split, TextInputFormat informat, MatrixBlock dest,
			int splitCount) {
			super(split, informat, dest, splitCount);
		}

		protected long parse(RecordReader<LongWritable, Text> reader, LongWritable key, Text value) throws IOException {
			final SparseBlock sb = _dest.getSparseBlock();
			long nnz = 0;
			double cellValue = 0;
			while(reader.next(key, value)) {
				_col = 0;
				final String cellStr = value.toString().trim();
				final String[] parts = IOUtilFunctions.split(cellStr, _props.getDelim());
				sb.allocate(_row);
				SparseRow r = sb.get(_row);
				for(String part : parts) {
					if(!part.isEmpty()) {
						cellValue = Double.parseDouble(part);
						if(cellValue != 0) {
							r.append(_col, cellValue);
							nnz++;
						}
					}
					_col++;
				}

				IOUtilFunctions.checkAndRaiseErrorCSVNumColumns(_split.toString(), cellStr, parts, _cLen);

				_row++;
			}
			return nnz;
		}
	}
}

/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.io;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;

/**
 * Parallel version of ReaderTextCSV.java. To summarize, we do two passes in
 * order to compute row offsets and the actual read. We accordingly create count
 * and read tasks and use fixed-size thread pools to execute these tasks. If the
 * target matrix is dense, the inserts are done lock-free. In contrast to
 * textcell parallel read, we also do lock-free inserts. If the matrix is
 * sparse, because splits contain row partitioned lines and hence there is no
 * danger of lost updates. Note, there is also no sorting of sparse rows
 * required because data comes in sorted order per row.
 * 
 */
public class ReaderTextCSVParallel extends MatrixReader {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n"
			+ "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private CSVFileFormatProperties _props = null;
	private int _numThreads = 1;

	private SplitOffsetInfos _offsets = null;

	public ReaderTextCSVParallel(CSVFileFormatProperties props) {
		_numThreads = OptimizerUtils.getParallelTextReadParallelism();
		_props = props;
	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen,
			int brlen, int bclen, long estnnz) throws IOException,
			DMLRuntimeException 
	{
		// prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		FileSystem fs = FileSystem.get(job);
		Path path = new Path(fname);

		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		InputSplit[] splits = informat.getSplits(job, _numThreads);

		if (splits[0] instanceof FileSplit) {

			// The splits do not always arrive in order by file name.
			// Sort the splits lexicographically by path so that the header will
			// be in the first split.
			// Note that we're assuming that the splits come in order by offset
			Arrays.sort(splits, new Comparator<InputSplit>() {

				@Override
				public int compare(InputSplit o1, InputSplit o2) {
					Path p1 = ((FileSplit) o1).getPath();
					Path p2 = ((FileSplit) o2).getPath();
					return p1.toString().compareTo(p2.toString());
				}

			});
		}

		// check existence and non-empty file
		checkValidInputFile(fs, path);

		// allocate output matrix block
		// First Read Pass (count rows/cols, determine offsets, allocate matrix
		// block)
		MatrixBlock ret = computeCSVSizeAndCreateOutputMatrixBlock(splits,
				path, job, _props.hasHeader(), _props.getDelim(), estnnz);
		rlen = ret.getNumRows();
		clen = ret.getNumColumns();

		// Second Read Pass (read, parse strings, append to matrix block)
		readCSVMatrixFromHDFS(splits, path, job, ret, rlen, clen, brlen, bclen,
				_props.hasHeader(), _props.getDelim(), _props.isFill(),
				_props.getFillValue());

		// post-processing (representation-specific, change of sparse/dense
		// block representation)
		// - no sorting required for CSV because it is read in sorted order per
		// row
		// - always recompute non zeros (not maintained for dense, potential
		// lost updates for sparse)
		ret.recomputeNonZeros();
		ret.examSparsity();

		// sanity check for parallel row count (since determined internally)
		if (rlen > 0 && rlen != ret.getNumRows())
			throw new DMLRuntimeException(
					"Read matrix inconsistent with given meta data: expected nrow="
							+ rlen + ", real nrow=" + ret.getNumRows());

		return ret;
	}

	/**
	 * 
	 * @param path
	 * @param job
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @param fillValue
	 * @return
	 * @throws IOException
	 */
	private void readCSVMatrixFromHDFS(InputSplit[] splits, Path path,
			JobConf job, MatrixBlock dest, long rlen, long clen, int brlen,
			int bclen, boolean hasHeader, String delim, boolean fill,
			double fillValue) throws IOException {
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		ExecutorService pool = Executors.newFixedThreadPool(_numThreads);

		try {
			// create read tasks for all splits
			ArrayList<CSVReadTask> tasks = new ArrayList<CSVReadTask>();
			int splitCount = 0;

			for (InputSplit split : splits) {
				CSVReadTask t = new CSVReadTask(split, _offsets, informat, job,
						dest, rlen, clen, hasHeader, delim, fill, fillValue,
						splitCount++);
				tasks.add(t);
			}

			pool.invokeAll(tasks);
			pool.shutdown();

			// check status of every thread and report only error-messages per
			// thread
			for (CSVReadTask rt : tasks) {
				if (!rt.getReturnCode()) {
					Exception err = rt.getException();
					throw new IOException("Read task for csv input failed: "
							+ err.toString(), err);
				}
			}
		} catch (Exception e) {
			throw new IOException("Threadpool issue, while parallel read.", e);
		}
	}

	/**
	 * 
	 * @param path
	 * @param job
	 * @param hasHeader
	 * @param delim
	 * @return
	 * @throws IOException
	 */
	private MatrixBlock computeCSVSizeAndCreateOutputMatrixBlock(
			InputSplit[] splits, Path path, JobConf job, boolean hasHeader,
			String delim, long estnnz) throws IOException 
	{
		int nrow = 0;
		int ncol = 0;

		MatrixBlock dest = null;
		String cellStr = null;

		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		ExecutorService pool = Executors.newFixedThreadPool(_numThreads);

		// count no of entities in the first non-header row
		LongWritable key = new LongWritable();
		Text oneLine = new Text();
		RecordReader<LongWritable, Text> reader = informat.getRecordReader(
				splits[0], job, Reporter.NULL);
		try {
			if (reader.next(key, oneLine)) {
				cellStr = oneLine.toString().trim();
				ncol = StringUtils.countMatches(cellStr, delim) + 1;
			}
		} finally {
			if (reader != null)
				reader.close();
		}

		// count rows in parallel per split
		try {
			ArrayList<CountRowsTask> tasks = new ArrayList<CountRowsTask>();
			for (InputSplit split : splits) {
				CountRowsTask t = new CountRowsTask(split, informat, job,
						hasHeader);
				hasHeader = false;
				tasks.add(t);
			}

			pool.invokeAll(tasks);
			pool.shutdown();

			// collect row counts for offset computation
			// early error notify in case not all tasks successful
			_offsets = new SplitOffsetInfos(tasks.size());
			for (CountRowsTask rt : tasks) {
				if (!rt.getReturnCode())
					throw new IOException(
							"Thread Error, while counting the rows "
									+ rt.getErrMsg());

				_offsets.setOffsetPerSplit(tasks.indexOf(rt), nrow);
				_offsets.setLenghtPerSplit(tasks.indexOf(rt), rt.getRowCount());
				nrow = nrow + rt.getRowCount();
			}

			// allocate target matrix block based on given size
			// need to allocate sparse as well since lock-free insert into
			// target
			dest = createOutputMatrixBlock(nrow, ncol, estnnz, true, true);
		} catch (Exception e) {
			throw new IOException("Threadpool Error " + e.getMessage(), e);
		}

		return dest;
	}

	/**
	 * 
	 * 
	 */
	private static class SplitOffsetInfos {
		// offset & length info per split
		private int[] offsetPerSplit = null;
		private int[] lenghtPerSplit = null;

		public SplitOffsetInfos(int numSplits) {
			lenghtPerSplit = new int[numSplits];
			offsetPerSplit = new int[numSplits];
		}

		@SuppressWarnings("unused")
		public SplitOffsetInfos(SplitOffsetInfos offsets) {
			lenghtPerSplit = offsets.lenghtPerSplit.clone();
			offsetPerSplit = offsets.offsetPerSplit.clone();
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

	/**
	 * 
	 * 
	 */
	private static class CountRowsTask implements Callable<Object> {
		private InputSplit _split = null;
		private TextInputFormat _informat = null;
		private JobConf _job = null;
		private boolean _rc = true;
		private String _errMsg = null;
		private int _nrows = -1;
		private boolean _hasHeader = false;

		public CountRowsTask(InputSplit split, TextInputFormat informat,
				JobConf job, boolean hasHeader) {
			_split = split;
			_informat = informat;
			_job = job;
			_nrows = 0;
			_hasHeader = hasHeader;
		}

		public boolean getReturnCode() {
			return _rc;
		}

		public String getErrMsg() {
			return _errMsg;
		}

		public int getRowCount() {
			return _nrows;
		}

		@Override
		public Object call() throws Exception {
			LongWritable key = new LongWritable();
			Text oneLine = new Text();

			RecordReader<LongWritable, Text> reader = _informat
					.getRecordReader(_split, _job, Reporter.NULL);

			try {
				// count rows from the first non-header row
				if (_hasHeader) {
					reader.next(key, oneLine);
				}
				while (reader.next(key, oneLine)) {
					_nrows++;
				}
			} catch (Exception e) {
				_rc = false;
				_errMsg = "RecordReader error CSV format. split: "
						+ _split.toString() + e.getMessage();
				throw new IOException(_errMsg);
			} finally {
				if (reader != null)
					reader.close();
			}

			return null;
		}
	}

	/**
	 * 
	 * 
	 */
	private static class CSVReadTask implements Callable<Object> {
		private InputSplit _split = null;
		private SplitOffsetInfos _splitoffsets = null;
		private boolean _sparse = false;
		private TextInputFormat _informat = null;
		private JobConf _job = null;
		private MatrixBlock _dest = null;
		private long _rlen = -1;
		private long _clen = -1;
		private boolean _isFirstSplit = false;
		private boolean _hasHeader = false;
		private boolean _fill = false;
		private double _fillValue = 0;
		private String _delim = null;
		private int _splitCount = 0;

		private boolean _rc = true;
		// private String _errMsg = null;
		private Exception _exception = null;

		public CSVReadTask(InputSplit split, SplitOffsetInfos offsets,
				TextInputFormat informat, JobConf job, MatrixBlock dest,
				long rlen, long clen, boolean hasHeader, String delim,
				boolean fill, double fillValue, int splitCount) {
			_split = split;
			_splitoffsets = offsets; // new SplitOffsetInfos(offsets);
			_sparse = dest.isInSparseFormat();
			_informat = informat;
			_job = job;
			_dest = dest;
			_rlen = rlen;
			_clen = clen;
			_isFirstSplit = (splitCount == 0);
			_hasHeader = hasHeader;
			_fill = fill;
			_fillValue = fillValue;
			_delim = delim;
			_rc = true;
			_splitCount = splitCount;
		}

		public boolean getReturnCode() {
			return _rc;
		}

		public Exception getException() {
			return _exception;
		}

		@Override
		public Object call() throws Exception {
			LongWritable key = new LongWritable();
			Text value = new Text();

			int row = 0;
			int col = 0;
			double cellValue = 0;

			String cellStr = null;

			try {
				RecordReader<LongWritable, Text> reader = _informat
						.getRecordReader(_split, _job, Reporter.NULL);

				// skip the header line
				if (_isFirstSplit && _hasHeader) {
					reader.next(key, value);
				}

				boolean noFillEmpty = false;
				row = _splitoffsets.getOffsetPerSplit(_splitCount);

				try {
					if (_sparse) // SPARSE<-value
					{
						while (reader.next(key, value)) // foreach line
						{
							cellStr = value.toString().trim();
							String[] parts = IOUtilFunctions.split(cellStr, _delim);
							col = 0;

							for (String part : parts) // foreach cell
							{
								part = part.trim();
								if (part.isEmpty()) {
									noFillEmpty |= !_fill;
									cellValue = _fillValue;
								} else {
									cellValue = IOUtilFunctions
											.parseDoubleParallel(part);
								}

								if (Double.compare(cellValue, 0.0) != 0)
									_dest.appendValue(row, col, cellValue);
								col++;
							}

							// sanity checks (number of columns, fill values)
							IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(cellStr, _fill, noFillEmpty);
							if (parts.length != _clen) {
								throw new IOException(
										"Invalid number of columns (" + col + ") found in delimited file "
										+ "(" + _split.toString() + "). Expecting (" + _clen + "): " + value);
							}
							row++;
						}
					} 
					else // DENSE<-value
					{
						while (reader.next(key, value)) // foreach line
						{
							cellStr = value.toString().trim();
							String[] parts = IOUtilFunctions.split(cellStr, _delim);
							col = 0;

							for (String part : parts) // foreach cell
							{
								part = part.trim();
								if (part.isEmpty()) {
									noFillEmpty |= !_fill;
									cellValue = _fillValue;
								} else {
									cellValue = IOUtilFunctions
											.parseDoubleParallel(part);
								}
								_dest.setValueDenseUnsafe(row, col, cellValue);
								col++;
							}

							// sanity checks (number of columns, fill values)
							IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(cellStr, _fill, noFillEmpty);
							if (parts.length != _clen) {
								throw new IOException(
										"Invalid number of columns (" + col + ") found in delimited file "
										+ "(" + _split.toString() + "). Expecting (" + _clen + "): " + value);
							}
							row++;
						}
					}

					// sanity checks (number of rows)
					if (row != (_splitoffsets.getOffsetPerSplit(_splitCount) + _splitoffsets
							.getLenghtPerSplit(_splitCount))) {
						throw new IOException(
								"Incorrect number of rows ("
										+ row
										+ ") found in delimited file ("
										+ (_splitoffsets
												.getOffsetPerSplit(_splitCount) + _splitoffsets
												.getLenghtPerSplit(_splitCount))
										+ "): " + value);
					}
				} finally {
					if (reader != null)
						reader.close();
				}
			} catch (Exception ex) {
				// central error handling (return code, message)
				_rc = false;
				_exception = ex;
				// _errMsg = ex.getMessage();

				// post-mortem error handling and bounds checking
				if (row < 0 || row + 1 > _rlen || col < 0 || col + 1 > _clen) {
					String errMsg = "CSV cell [" + (row + 1) + "," + (col + 1)
							+ "] " + "out of overall matrix range [1:" + _rlen
							+ ",1:" + _clen + "]. " + ex.getMessage();
					throw new IOException(errMsg, _exception);
				} else {
					String errMsg = "Unable to read matrix in text CSV format. "
							+ ex.getMessage();
					throw new IOException(errMsg, _exception);
				}
			}

			return null;
		}
	}
}

/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;

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

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.io.ReaderTextCellParallel.CellBuffer;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;


public class ReaderTextCSVParallel extends MatrixReader {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n"
			+ "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	CSVFileFormatProperties _props = null;
	private int _numThreads = 1;

	public ReaderTextCSVParallel(CSVFileFormatProperties props) {
		_props = props;
		_numThreads = InfrastructureAnalyzer.getLocalParallelism();
	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen,
			int brlen, int bclen, long estnnz) throws IOException,
			DMLRuntimeException {
		
		// prepare file access
		JobConf job = new JobConf();
		FileSystem fs = FileSystem.get(job);
		Path path = new Path(fname);

		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		InputSplit[] splits = informat.getSplits(job, _numThreads);
		
		// check existence and non-empty file
		checkValidInputFile(fs, path);

		// allocate output matrix block
		MatrixBlock ret = null;
		// walk-thru the file, count rows+cols and then create MatrixBlock
		ret = computeCSVSizeAndCreateOutputMatrixBlock(splits, path, job, _props.hasHeader(), _props.getDelim(), estnnz);
		rlen = ret.getNumRows();
		clen = ret.getNumColumns();
		
		// core read
		readCSVMatrixFromHDFS(splits, path, job, ret, rlen, clen, brlen,
				bclen, _props.hasHeader(), _props.getDelim(), _props.isFill(),
				_props.getFillValue());

		// finally check if change of sparse/dense block representation required
		if( ret.isInSparseFormat() ) {
			ret.sortSparseRows();
		} 
		else {
			ret.recomputeNonZeros();
		}
		ret.examSparsity();

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
	private void readCSVMatrixFromHDFS(InputSplit[] splits, Path path, JobConf job,
			MatrixBlock dest, long rlen, long clen, int brlen,
			int bclen, boolean hasHeader, String delim, boolean fill,
			double fillValue) throws IOException {

		boolean sparse = dest.isInSparseFormat();
		boolean isFirstSplit = true;
		int splitCount = 0;

		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		ExecutorService pool = Executors.newFixedThreadPool(_numThreads);

		try 
		{
			//create read tasks for all splits
			ArrayList<CSVReadTask> tasks = new ArrayList<CSVReadTask>();
			for( InputSplit split : splits ){
				CSVReadTask t = new CSVReadTask(split, sparse, informat, job, dest, rlen, clen, isFirstSplit, hasHeader, delim, fill, fillValue, splitCount);
				isFirstSplit = false;
				tasks.add(t);
				splitCount++;
			}

			pool.invokeAll(tasks);	
			pool.shutdown();
			
			// check status of every thread and report only error-messages per thread
			for(CSVReadTask rt : tasks) {
				if (rt.getReturnCode() == false) {
					throw new IOException("ReadTask issue " + rt.getErrMsg());
				}
			}

		} 
		catch (Exception e) {
			throw new IOException("Threadpool issue, while parallel read " + e.getMessage());
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
	private MatrixBlock computeCSVSizeAndCreateOutputMatrixBlock(InputSplit[] splits, Path path, JobConf job,
			boolean hasHeader, String delim, long estnnz) throws IOException {
		int nrow = 0;
		int ncol = 0;
		int cumuRow = 0;
		int r = 0;

		String escapedDelim = Pattern.quote(delim);
		Pattern compiledDelim = Pattern.compile(escapedDelim);
		MatrixBlock dest = null;
		String cellStr = null;

		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		
		ExecutorService pool = Executors.newFixedThreadPool(_numThreads);
		
		// count no of entities in the first non-header row
		LongWritable key = new LongWritable();
		Text oneLine = new Text();
		try {
			RecordReader<LongWritable,Text> reader = informat.getRecordReader(splits[0], job, Reporter.NULL);
			try {
				if (hasHeader) {
					reader.next(key, oneLine); // ignore header
				}
				reader.next(key, oneLine);
				cellStr = oneLine.toString().trim();
				ncol = compiledDelim.split(cellStr,-1).length;
			} 
			catch(Exception ex) {
				throw new IOException("CSV parse error " + ex.getMessage());
			}
			finally {
				if( reader != null )
					reader.close();
			}
		} 
		catch (Exception e) {
			throw new IOException("RecordReader error " + e.getMessage());
		}

		// count rows in parallel per split
		try 
		{
			ArrayList<CountRowsTask> tasks = new ArrayList<CountRowsTask>();
			for( InputSplit split : splits ){
				CountRowsTask t = new CountRowsTask(split, informat, job, hasHeader);
				hasHeader = false;
				tasks.add(t);
			}

			pool.invokeAll(tasks);	
			pool.shutdown();
			
			for(CountRowsTask rt : tasks) {
				if (rt.getReturnCode()) {
					nrow = nrow + rt.getRowCount();
				} 
				else {
					throw new IOException("Thread Error, while counting the rows " + rt.getErrMsg());
				}
			}

			try {
				dest = createOutputMatrixBlock(nrow, ncol, estnnz, true);
				dest.allocateOffsetLenghtPerSplit(tasks.size());
			} catch (DMLRuntimeException e) {
				throw new IOException("Unable to allocate memory " + e.getMessage());
			}
			
			for(CountRowsTask rt : tasks) {
				dest.setOffsetPerSplit(tasks.indexOf(rt) , cumuRow);
				r = rt.getRowCount();
				dest.setLenghtPerSplit(tasks.indexOf(rt), r);
				cumuRow = cumuRow + r;
			}
			
			if (cumuRow != nrow) {
				throw new IOException("The sum of rowCounts per split is not adding up. Expected: " + nrow + " Actual: " + cumuRow);
			}
			
		} catch (Exception e) {
			throw new IOException("Threadpool Error " + e.getMessage());
		}

		return dest;
	}
}

/**
 * 
 * 
 */
class CountRowsTask implements Callable<Object> {
	private InputSplit _split = null;
	private TextInputFormat _informat = null;
	private JobConf _job = null;
	private boolean _rc = true;
	private String _errMsg = null;
	private int _nrows = -1;
	private boolean _hasHeader = false;

	public CountRowsTask(InputSplit split, TextInputFormat informat, JobConf job, boolean hasHeader) {
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
	public Object call() throws Exception 
	{
		LongWritable key = new LongWritable();
		Text oneLine = new Text();
		try {
			RecordReader<LongWritable,Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
			
			// count rows from the first non-header row
			try {
				if (_hasHeader) reader.next(key, oneLine);
				
				while(reader.next(key, oneLine)) {
					_nrows++;
				}
			} catch(Exception ex) {
				_rc = false;
				_errMsg = new String("Unable to read rows in CSV format. split: " + _split.toString() + ex.getMessage());
				throw new IOException(_errMsg);
			}
			finally {
				if( reader != null )
					reader.close();
			}
		} catch (Exception e) {
			_rc = false;
			_errMsg = new String("RecordReader error CSV format. split: " + _split.toString() + e.getMessage());
			throw new IOException(_errMsg);
		}
		return null;
	}
}

/**
 * 
 * 
 */
class CSVReadTask implements Callable<Object> 
{
	private InputSplit _split = null;
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
	private String _errMsg = null;
	
	public CSVReadTask( InputSplit split, boolean sparse, TextInputFormat informat, JobConf job, MatrixBlock dest, 
			long rlen, long clen, boolean isFirstSplit, boolean hasHeader, String delim, boolean fill, double fillValue, int splitCount)
	{
		_split = split;
		_sparse = sparse;
		_informat = informat;
		_job = job;
		_dest = dest;
		_rlen = rlen;
		_clen = clen;
		_isFirstSplit = isFirstSplit;
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

	public String getErrMsg() {
		return _errMsg;
	}

	@Override
	public Object call() throws Exception 
	{
		LongWritable key = new LongWritable();
		Text value = new Text();
		
		int row = 0;
		int col = 0;
		double cellValue = 0;
		
		String escapedDelim = Pattern.quote(_delim);
		Pattern compiledDelim = Pattern.compile(escapedDelim);
		String cellStr = null;
		
		try
		{
			RecordReader<LongWritable,Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
			
			// skip the header line
			if (_isFirstSplit && _hasHeader) {
				reader.next(key, value);
			}

			boolean emptyValuesFound = false;
			row = _dest.getOffsetPerSplit(_splitCount);
			try{
				if( _sparse ) //SPARSE<-value
				{
					CellBuffer buff = new CellBuffer();
					
					while( reader.next(key, value) )
					{
						col = 0;
						cellStr = value.toString().trim();
						emptyValuesFound = false;

						for(String part : compiledDelim.split(cellStr, -1)) {
							part = part.trim();
							if ( part.isEmpty() ) {
								emptyValuesFound = true;
								cellValue = _fillValue;
							}
							else {
								cellValue = IOUtilFunctions.parseDoubleParallel(part);
							}
							
							if ( Double.compare(cellValue, 0.0) != 0 ) {
								buff.addCell(row, col, cellValue);

								if( buff.size() >= (CellBuffer.CAPACITY) ) {
									synchronized( _dest ){ //sparse requires lock
										buff.flushCellBufferToMatrixBlock(_dest);
									}
								}
							}
							col++;
						}
						
						if ( !_fill && emptyValuesFound) {
							_rc = false;
							_errMsg = new String("Empty fields found in delimited file (" + _split.toString() + "). Use \"fill\" option to read delimited files with empty fields." + cellStr);
							throw new IOException(_errMsg);
						}
						
						if ( col != _clen ) {
							_rc = false;
							_errMsg = new String("Invalid number of columns (" + col + ") found in delimited file (" + _split.toString() + "). Expecting (" + _clen + "): " + value);
							throw new IOException(_errMsg);
						}
						row++;
					}

					if (row != (_dest.getOffsetPerSplit(_splitCount) + _dest.getLenghtPerSplit(_splitCount))) {
						_rc = false;
						_errMsg = new String("Incorrect number of rows (" + row + ") found in delimited file (" + _split.toString() + "). Expecting (" + (_dest.getOffsetPerSplit(_splitCount) + _dest.getLenghtPerSplit(_splitCount)) + "): " + value);
						throw new IOException(_errMsg);						
					}

					synchronized( _dest ){ //sparse requires lock
						buff.flushCellBufferToMatrixBlock(_dest);
					}

				} 
				else //DENSE<-value
				{
					while( reader.next(key, value) )
					{
						cellStr = value.toString().trim();
						col = 0;
						for(String part : compiledDelim.split(cellStr, -1)) {
							part = part.trim();
							if ( part.isEmpty() ) {
								if ( !_fill ) {
									throw new IOException("Empty fields found in delimited file (" + _split.toString() + "). Use \"fill\" option to read delimited files with empty fields.");
								}
								else {
									cellValue = _fillValue;
								}
							}
							else {
								cellValue = IOUtilFunctions.parseDoubleParallel(part);
							}
							_dest.setValueDenseUnsafe(row, col, cellValue);
							col++;
						}
						if ( col != _clen ) {
							_rc = false;
							_errMsg = new String("Invalid number of columns (" + col + ") found in delimited file (" + _split.toString() + "). Expecting (" + _clen + "): " + value);
							throw new IOException(_errMsg);
						}
						row++;
					}


					if (row != (_dest.getOffsetPerSplit(_splitCount) + _dest.getLenghtPerSplit(_splitCount))) {
						_rc = false;
						_errMsg = new String("Incorrect number of rows (" + row + ") found in delimited file (" + (_dest.getOffsetPerSplit(_splitCount) + _dest.getLenghtPerSplit(_splitCount)) + "): " + value);
						throw new IOException(_errMsg);						
					}
				}
			}
			finally
			{
				if( reader != null )
					reader.close();
			}
		}
		catch(Exception ex)
		{
			_rc = false;

			//post-mortem error handling and bounds checking
			if( row < 0 || row + 1 > _rlen || col < 0 || col + 1 > _clen )
			{
				_errMsg = new String("CSV cell ["+(row+1)+","+(col+1)+"] " +
						  "out of overall matrix range [1:"+_rlen+",1:"+_clen+"]. " + ex.getMessage());
				throw new IOException(_errMsg);
			}
			else {
				_errMsg = new String("Unable to read matrix in text CSV format. " + ex.getMessage());
				throw new IOException(_errMsg);
			}
		}
		
		return null;
	}
}

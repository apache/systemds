/**
* IBM Confidential
* OCO Source Materials
* (C) Copyright IBM Corp. 2010, 2015
* The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
*/

package com.ibm.bi.dml.runtime.io;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.BinaryBlockInputFormat;

public class ReaderBinaryBlockParSplits extends MatrixReader {

	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private boolean _localFS = false;
	private static int _numThreads = 1;
	
	public ReaderBinaryBlockParSplits( boolean localFS )
	{
		_localFS = localFS;
		_numThreads = OptimizerUtils.getParallelTextReadParallelism();
	}
	
	public void setLocalFS(boolean flag) {
		_localFS = flag;
	}
	
	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, estnnz, false, false);
		
		//prepare file access
		JobConf job = new JobConf();	
		FileSystem fs = _localFS ? FileSystem.getLocal(job) : FileSystem.get(job);
		Path path = new Path( (_localFS ? "file:///" : "") + fname); 
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		readBinaryBlockMatrixFromHDFS(path, job, fs, ret, rlen, clen, brlen, bclen);
		
		//finally check if change of sparse/dense block representation required
		ret.recomputeNonZeros();
		ret.examSparsity();
		
		return ret;
	}
	
	/**
	 * 
	 * @param fname
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param estnnz
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	public ArrayList<IndexedMatrixValue> readIndexedMatrixBlocksFromHDFS(String fname, long rlen, long clen, int brlen, int bclen) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block collection
		ArrayList<IndexedMatrixValue> ret = new ArrayList<IndexedMatrixValue>();
		
		//prepare file access
		JobConf job = new JobConf();	
		FileSystem fs = _localFS ? FileSystem.getLocal(job) : FileSystem.get(job);
		Path path = new Path( (_localFS ? "file:///" : "") + fname); 
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		readBinaryBlockMatrixBlocksFromHDFS(path, job, fs, ret, rlen, clen, brlen, bclen);
		
		return ret;
	}


	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param fs 
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 * @throws DMLRuntimeException 
	 */
	private static void readBinaryBlockMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, DMLRuntimeException
	{
		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );
		
		FileInputFormat.addInputPath(job, path);
/*		
		BinaryBlockInputFormat informat = new BinaryBlockInputFormat();
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, _numThreads);

		SequenceFileInputFormat<MatrixIndexes, MatrixBlock> informat = new SequenceFileInputFormat<MatrixIndexes, MatrixBlock>();
		InputSplit[] seqsplits = informat.getSplits(job, _numThreads);
*/
		BinaryBlockInputFormat informat = new BinaryBlockInputFormat();
		InputSplit[] seqsplits = informat.getSplits(job, _numThreads);
		
		ExecutorService pool = Executors.newFixedThreadPool(_numThreads);
		try 
		{
			//create read tasks for all splits
			ArrayList<ReadMatrixPerSplitTask> tasks = new ArrayList<ReadMatrixPerSplitTask>();

			for( InputSplit split : seqsplits ){
				ReadMatrixPerSplitTask t = new ReadMatrixPerSplitTask(split, informat, job, dest, rlen, clen, brlen, bclen);
				tasks.add(t);
			}

			//wait until all tasks have been executed
			pool.invokeAll(tasks);	
			pool.shutdown();
			
			//early error notify in case not all tasks successful
			for(ReadMatrixPerSplitTask rt : tasks) {
				if( !rt.getReturnCode() ) {
					throw new IOException("Read task for text input failed: " + rt.getErrMsg());
				}
			}

		} 
		catch (Exception e) {
			throw new IOException("Threadpool issue, while parallel read.", e);
		}
	}
	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param fs
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private void readBinaryBlockMatrixBlocksFromHDFS( Path path, JobConf job, FileSystem fs, Collection<IndexedMatrixValue> dest, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );
		
		ExecutorService pool = Executors.newFixedThreadPool(_numThreads);
		try 
		{
			//create read tasks for all splits
			ArrayList<ReadMatrixBlockTask> tasks = new ArrayList<ReadMatrixBlockTask>();
			for( Path lpath : getSequenceFilePaths(fs, path) ){
				ReadMatrixBlockTask t = new ReadMatrixBlockTask(lpath, job, fs, dest, rlen, clen, brlen, bclen);
				tasks.add(t);
			}
			
			//wait until all tasks have been executed
			pool.invokeAll(tasks);	
			pool.shutdown();
			
			//early error notify in case not all tasks successful
			for(ReadMatrixBlockTask rt : tasks) {
				if( !rt.getReturnCode() ) {
					throw new IOException("Read task for text input failed: " + rt.getErrMsg());
				}
			}

		} 
		catch (Exception e) {
			throw new IOException("Threadpool issue, while parallel read.", e);
		}

	}

	/**
	 * 
	 * 	 * Note: For efficiency, we directly use SequenceFile.Reader instead of SequenceFileInputFormat-
	 * InputSplits-RecordReader (SequenceFileRecordReader). First, this has no drawbacks since the
	 * SequenceFileRecordReader internally uses SequenceFile.Reader as well. Second, it is 
	 * advantageous if the actual sequence files are larger than the file splits created by   
	 * informat.getSplits (which is usually aligned to the HDFS block size) because then there is 
	 * overhead for finding the actual split between our 1k-1k blocks. This case happens
	 * if the read matrix was create by CP or when jobs directly write to large output files 
	 * (e.g., parfor matrix partitioning).
	 * 
	 */
	public static class ReadMatrixPerSplitTask implements Callable<Object> 
	{

		private boolean _sparse = false;
		private InputSplit _split = null;
		private BinaryBlockInputFormat _in = null;
		private JobConf _job = null;
		private MatrixBlock _dest = null;
		private long _rlen = -1;
		private long _clen = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		
		private boolean _rc = true;
		private String _errMsg = null;
		
		public ReadMatrixPerSplitTask( InputSplit split, BinaryBlockInputFormat in, JobConf job, MatrixBlock dest, long rlen, long clen, int brlen, int bclen)
		{
			_split = split;
			_in = in;
			_sparse = dest.isInSparseFormat();
			_job = job;
			_dest = dest;
			_rlen = rlen;
			_clen = clen;
			_brlen = brlen;
			_bclen = bclen;
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
			MatrixIndexes key = new MatrixIndexes(); 
			MatrixBlock value = new MatrixBlock();

			//directly read from each split
			RecordReader<MatrixIndexes, MatrixBlock> reader = _in.getRecordReader(_split, _job, Reporter.NULL);
			
			try
			{
				//note: next(key, value) does not yet exploit the given serialization classes, record reader does but is generally slower.
				while( reader.next(key, value) )
				{	
					//empty block filter (skip entire block)
					if( value.isEmptyBlock(false) )
						continue;
					
					int row_offset = (int)(key.getRowIndex()-1)*_brlen;
					int col_offset = (int)(key.getColumnIndex()-1)*_bclen;
					
					int rows = value.getNumRows();
					int cols = value.getNumColumns();
					
					//bound check per block
					if( row_offset + rows < 0 || row_offset + rows > _rlen || col_offset + cols<0 || col_offset + cols > _clen )
					{
						throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
								              "out of overall matrix range [1:"+_rlen+",1:"+_clen+"].");
					}
			
					//copy block to result
					if( _sparse )
					{
						if (cols < _clen ) {
							synchronized( _dest ){ //sparse requires lock
								_dest.appendToSparse(value, row_offset, col_offset);
								//note: append requires final sort
							}
						}
						else {
							_dest.copy( row_offset, row_offset+rows-1, 
									   col_offset, col_offset+cols-1,
									   value, false );
						}
					} 
					else
					{
						_dest.copy( row_offset, row_offset+rows-1, 
								   col_offset, col_offset+cols-1,
								   value, false );
					}
				}
			}
			finally
			{
				if( reader != null )
					reader.close();
			}
			return null;
		}
	}

	
	/**
	 * 
	 * 
	 */
	public static class ReadMatrixBlockTask implements Callable<Object> 
	{
		private Path _path = null;
		private JobConf _job = null;
		private FileSystem _fs = null;
		private Collection<IndexedMatrixValue> _dest = null;
		private long _rlen = -1;
		private long _clen = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		
		private boolean _rc = true;
		private String _errMsg = null;
		
		public ReadMatrixBlockTask( Path path, JobConf job, FileSystem fs, Collection<IndexedMatrixValue> dest, long rlen, long clen, int brlen, int bclen)
		{
			_path = path;
			_fs = fs;
			_job = job;
			_dest = dest;
			_rlen = rlen;
			_clen = clen;
			_brlen = brlen;
			_bclen = bclen;
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
			MatrixIndexes key = new MatrixIndexes(); 
			MatrixBlock value = new MatrixBlock();


			//directly read from sequence files (individual partfiles)
			@SuppressWarnings("deprecation")
			SequenceFile.Reader reader = new SequenceFile.Reader(_fs,_path,_job);
			
			try
			{
				while( reader.next(key, value) )
				{	
					int row_offset = (int)(key.getRowIndex()-1)*_brlen;
					int col_offset = (int)(key.getColumnIndex()-1)*_bclen;
					int rows = value.getNumRows();
					int cols = value.getNumColumns();
					
					//bound check per block
					if( row_offset + rows < 0 || row_offset + rows > _rlen || col_offset + cols<0 || col_offset + cols > _clen )
					{
						throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
								              "out of overall matrix range [1:"+_rlen+",1:"+_clen+"].");
					}
			
					//copy block to result
					_dest.add(new IndexedMatrixValue(new MatrixIndexes(key), new MatrixBlock(value)));
				}
			}
			finally
			{
				if( reader != null )
					reader.close();
			}
			
			return null;
		}
	}

}

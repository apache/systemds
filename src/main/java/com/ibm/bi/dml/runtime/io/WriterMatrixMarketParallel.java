/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.matrix.data.IJV;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.SparseRowsIterator;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

public class WriterMatrixMarketParallel extends WriterMatrixMarket
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int brlen, int bclen, long nnz) 
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//validity check block dimensions
		if( src.getNumRows() != rlen || src.getNumColumns() != clen ) {
			throw new IOException("Matrix block dimensions mismatch with metadata: "+src.getNumRows()+"x"+src.getNumColumns()+" vs "+rlen+"x"+clen+".");
		}
		
		//prepare file access
		JobConf job = new JobConf();
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );
			
		//core write
		writeMatrixMarketMatrixToHDFS(path, job, src, rlen, clen, nnz);
	}

	/**
	 * 
	 * @param fileName
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @throws IOException
	 */
	@Override
	protected void writeMatrixMarketMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, long nnz )
		throws IOException
	{
		//estimate output size and number of output blocks (min 1)
		int numPartFiles = (int)(OptimizerUtils.estimateSizeTextOutput(src.getNumRows(), src.getNumColumns(), src.getNonZeros(), 
				              OutputInfo.MatrixMarketOutputInfo)  / InfrastructureAnalyzer.getHDFSBlockSize());
		numPartFiles = Math.max(numPartFiles, 1);
		
		//determine degree of parallelism
		int _numThreads = OptimizerUtils.getParallelTextWriteParallelism();
		_numThreads = Math.min(_numThreads, numPartFiles);
		
		//create thread pool
		ExecutorService pool = Executors.newFixedThreadPool(_numThreads);
		
		try 
		{
			if (_numThreads > 1) {
				MapReduceTool.createDirIfNotExistOnHDFS(path.toString(), DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
			}
			
			//create write tasks for all splits
			ArrayList<WriteMMTask> tasks = new ArrayList<WriteMMTask>();
			long offset = rlen/_numThreads;
			long rowStart = 0;
			WriteMMTask t = null;

			for( int i=0; i < _numThreads; i++ ){
				if (i == (_numThreads-1)) {
					offset = rlen;
				}
				if (_numThreads > 1) {
					Path newPath = new Path(path, String.format("0-m-%05d",i));
					t = new WriteMMTask(newPath, job, src, rowStart, offset);
				}
				else {
					t = new WriteMMTask(path, job, src, rowStart, offset);
				}
				
				tasks.add(t);
				rowStart = rowStart + offset;
				rlen = rlen - offset;
			}
			
			//wait until all tasks have been executed
			pool.invokeAll(tasks);	
			pool.shutdown();
			
			//early error notify in case not all tasks successful
			for(WriteMMTask rt : tasks) {
				if( !rt.getReturnCode() ) {
					throw new IOException("Parallel write task failed: " + rt.getErrMsg());
				}
			}
		} 
		catch (Exception e) {
			throw new IOException("Parallel write of matrixmarket output failed.", e);
		}
	}
	
	/**
	 * 
	 * 
	 */
	private static class WriteMMTask implements Callable<Object> 
	{
		private JobConf _job = null;
		private MatrixBlock _src = null;
		private Path _path =null;
		private long _rowStart = -1;
		private long _rowNum = -1;

		private boolean _rc = true;
		private String _errMsg = null;
		
		public WriteMMTask(Path path, JobConf job, MatrixBlock src, long rowStart, long rowNum)
		{
			_path = path;
			_job = job;
			_src = src;
			_rowStart = rowStart;
			_rowNum = rowNum;
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
			boolean entriesWritten = false;
			FileSystem fs = FileSystem.get(_job);
			BufferedWriter bw = null;
			
			int rows = _src.getNumRows();
	    	int cols = _src.getNumColumns();
	    	long nnz = _src.getNonZeros();
	    	
			try
			{
				//for obj reuse and preventing repeated buffer re-allocations
				StringBuilder sb = new StringBuilder();
		        bw = new BufferedWriter(new OutputStreamWriter(fs.create(_path,true)));
				
		        if (_rowStart == 0) {
					// First output MM header
					sb.append ("%%MatrixMarket matrix coordinate real general\n");
				
					// output number of rows, number of columns and number of nnz
					sb.append (rows + " " + cols + " " + nnz + "\n");
		            bw.write( sb.toString());
		            sb.setLength(0);		            
		        }
		        
				if( _src.isInSparseFormat() ) //SPARSE
				{			   
					SparseRowsIterator iter = _src.getSparseRowsIterator((int)_rowStart, (int)_rowNum);

					while( iter.hasNext() )
					{
						IJV cell = iter.next();

						sb.append(cell.i+1);
						sb.append(' ');
						sb.append(cell.j+1);
						sb.append(' ');
						sb.append(cell.v);
						sb.append('\n');
						bw.write( sb.toString() );
						sb.setLength(0); 
						entriesWritten = true;
					}
				}
				else //DENSE
				{
					for( int i=(int)_rowStart; i<(_rowStart+_rowNum); i++ )
					{
						String rowIndex = Integer.toString(i+1);					
						for( int j=0; j<cols; j++ )
						{
							double lvalue = _src.getValueDenseUnsafe(i, j);
							if( lvalue != 0 ) //for nnz
							{
								sb.append(rowIndex);
								sb.append(' ');
								sb.append( j+1 );
								sb.append(' ');
								sb.append( lvalue );
								sb.append('\n');
								bw.write( sb.toString() );
								sb.setLength(0); 
								entriesWritten = true;
							}
						}
					}
				}
				
				//handle empty result
				if ( !entriesWritten ) {
			        bw.write("1 1 0\n");
				}
			}
			catch(Exception ex)
			{
				//central error handling (return code, message) 
				_rc = false;
				_errMsg = ex.getMessage();
				throw new RuntimeException(_errMsg, ex );
			}
			finally
			{
				IOUtilFunctions.closeSilently(bw);
			}
			
			return null;
		}
	}
}

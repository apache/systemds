/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.io.WriterTextCellParallel.WriteTask;
import com.ibm.bi.dml.runtime.matrix.data.IJV;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.SparseRowsIterator;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

public class WriterMatrixMarketParallel extends MatrixWriter
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int brlen, int bclen, long nnz) 
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException 
	{
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
	private static void writeMatrixMarketMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, long nnz )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		boolean entriesWritten = false;
		FileSystem fs = FileSystem.get(job);
        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
        
    	int rows = src.getNumRows();
		int cols = src.getNumColumns();

		//bound check per block
		if( rows > rlen || cols > clen )
		{
			throw new IOException("Matrix block [1:"+rows+",1:"+cols+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		
		
		int _numThreads = OptimizerUtils.getParallelTextReadParallelism();
		int numPartFiles = (int) ((src.getExactSizeOnDisk()) / (fs.getFileStatus(path).getBlockSize()));
		
		if (numPartFiles == 0) {  // if data is less than DFS BlockSize
			numPartFiles = 1;
		}
		
		_numThreads = Math.min(_numThreads, numPartFiles);
		ExecutorService pool = Executors.newFixedThreadPool(_numThreads);
		

		try 
		{
			if (_numThreads > 1) {
				fs.close();
				MapReduceTool.deleteFileIfExistOnHDFS(path.toString());
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
					t = new WriteMMTask(newPath, job, src, rowStart, offset, clen, nnz);
				}
				else {
					t = new WriteMMTask(path, job, src, rowStart, offset, clen, nnz);
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
					throw new IOException("Read task for text input failed: " + rt.getErrMsg());
				}
			}
		} 
		catch (Exception e) {
			throw new IOException("Threadpool issue, while parallel write.", e);
		}
	}
	
	
	/**
	 * 
	 * 
	 */
	public static class WriteMMTask implements Callable<Object> 
	{
		private boolean _sparse = false;
		private JobConf _job = null;
		private MatrixBlock _src = null;
		private Path _path =null;
		private long _rlen = -1;
		private long _rowStart = -1;
		private long _clen = -1;
		private long _nnz = -1;
		private BufferedWriter _bw = null;
		private StringBuilder _sb = null;
		boolean _entriesWritten = false;

//		private boolean _matrixMarket = false;
		
		private boolean _rc = true;
		private String _errMsg = null;
		
		public WriteMMTask(Path path, JobConf job, MatrixBlock src, long rowStart, long rlen, long clen, long nnz)
		{
			_sparse = src.isInSparseFormat();
			_path = path;
			_job = job;
			_src = src;
			_rowStart = rowStart;
			_rlen = rlen;
			_clen = clen;
			_nnz = nnz;
//			_matrixMarket = matrixMarket;
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
			_entriesWritten = false;
			FileSystem fs = FileSystem.get(_job);


	    	int rows = _src.getNumRows();
			int cols = _src.getNumColumns();

			try
			{
				//for obj reuse and preventing repeated buffer re-allocations
				_sb = new StringBuilder();
		        _bw = new BufferedWriter(new OutputStreamWriter(fs.create(_path,true)));
				
		        if (_rowStart == 0) {
					// First output MM header
					_sb.append ("%%MatrixMarket matrix coordinate real general\n");
				
					// output number of rows, number of columns and number of nnz
					_sb.append (_rlen + " " + _clen + " " + _nnz + "\n");
		            _bw.write( _sb.toString());
		            _sb.setLength(0);
		            
		        }
		        
				if( _sparse ) //SPARSE
				{			   
					SparseRowsIterator iter = new SparseRowsIterator((int)_rowStart, _src.getSparseRows());

					while( iter.hasNext() )
					{
						IJV cell = iter.next();

						_sb.append(cell.i+1);
						_sb.append(' ');
						_sb.append(cell.j+1);
						_sb.append(' ');
						_sb.append(cell.v);
						_sb.append('\n');
						_bw.write( _sb.toString() );
						_sb.setLength(0); 
						_entriesWritten = true;
					}
				}
				else //DENSE
				{
					for( int i=(int)_rowStart; i<(_rowStart+_rlen); i++ )
					{
						String rowIndex = Integer.toString(i+1);					
						for( int j=0; j<cols; j++ )
						{
							double lvalue = _src.getValueDenseUnsafe(i, j);
							if( lvalue != 0 ) //for nnz
							{
								_sb.append(rowIndex);
								_sb.append(' ');
								_sb.append( j+1 );
								_sb.append(' ');
								_sb.append( lvalue );
								_sb.append('\n');
								_bw.write( _sb.toString() );
								_sb.setLength(0); 
								_entriesWritten = true;
							}
							
						}
					}
				}
				
				//handle empty result
				if ( !_entriesWritten ) {
			        _bw = new BufferedWriter(new OutputStreamWriter(fs.create(_path,true)));
					_bw.write("1 1 0\n");
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
				IOUtilFunctions.closeSilently(_bw);
			}
			
			return null;
		}
	}
	
	
	
	/**
	 * 
	 * @param srcFileName
	 * @param fileName
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @throws IOException
	 */
	public void mergeTextcellToMatrixMarket( String srcFileName, String fileName, long rlen, long clen, long nnz )
		throws IOException
	{
		  Configuration conf = new Configuration();
		
		  Path src = new Path (srcFileName);
	      Path merge = new Path (fileName);
	      FileSystem hdfs = FileSystem.get(conf);
	    
	      if (hdfs.exists (merge)) {
	    	hdfs.delete(merge, true);
	      }
        
	      OutputStream out = hdfs.create(merge, true);

	      // write out the header first 
	      StringBuilder  sb = new StringBuilder();
	      sb.append ("%%MatrixMarket matrix coordinate real general\n");
	    
	      // output number of rows, number of columns and number of nnz
	 	  sb.append (rlen + " " + clen + " " + nnz + "\n");
	      out.write (sb.toString().getBytes());

	      // if the source is a directory
	      if (hdfs.getFileStatus(src).isDirectory()) {
	        try {
	          FileStatus[] contents = hdfs.listStatus(src);
	          for (int i = 0; i < contents.length; i++) {
	            if (!contents[i].isDirectory()) {
	               InputStream in = hdfs.open (contents[i].getPath());
	               try {
	                 IOUtils.copyBytes (in, out, conf, false);
	               }  finally {
	                  IOUtilFunctions.closeSilently(in);
	               }
	             }
	           }
	         } finally {
	        	 IOUtilFunctions.closeSilently(out);
	         }
	      } else if (hdfs.isFile(src))  {
	        InputStream in = null;
	        try {
   	          in = hdfs.open (src);
	          IOUtils.copyBytes (in, out, conf, true);
	        } 
	        finally {
	        	IOUtilFunctions.closeSilently(in);
	        	IOUtilFunctions.closeSilently(out);
	        }
	      } else {
	        throw new IOException(src.toString() + ": No such file or directory");
	      }
	}
}

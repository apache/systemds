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
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.FileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

public class WriterTextCSVParallel extends WriterTextCSV
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public WriterTextCSVParallel( CSVFileFormatProperties props )
	{
		super( props );
		_props = props;
	}
	
	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int brlen, int bclen, long nnz) 
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//validity check block dimensions
		if( src.getNumRows() != rlen || src.getNumColumns() != clen ) {
			throw new IOException("Matrix block dimensions mismatch with metadata: "+src.getNumRows()+"x"+src.getNumColumns()+" vs "+rlen+"x"+clen+".");
		}
		
		//file format property check
		if( _props!=null &&  !(_props instanceof CSVFileFormatProperties) ) {
			throw new IOException("Wrong type of file format properties for CSV writer.");
		}
		
		
		//prepare file access
		JobConf job = new JobConf();
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );
			
		//core write
		writeCSVMatrixToHDFS(path, job, src, rlen, clen, nnz, _props);
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
	protected void writeCSVMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, long nnz, FileFormatProperties formatProperties )
		throws IOException
	{
		//estimate output size and number of output blocks (min 1)
		int numPartFiles = (int)(OptimizerUtils.estimateSizeTextOutput(src.getNumRows(), src.getNumColumns(), src.getNonZeros(), 
				              OutputInfo.CSVOutputInfo)  / InfrastructureAnalyzer.getHDFSBlockSize());
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
			ArrayList<WriteCSVTask> tasks = new ArrayList<WriteCSVTask>();
			long offset = rlen/_numThreads;
			long rowStart = 0;
			WriteCSVTask t = null;

			for( int i=0; i < _numThreads; i++ ){
				if (i == (_numThreads-1)) {
					offset = rlen;
				}
				if (_numThreads > 1) {
					Path newPath = new Path(path, String.format("0-m-%05d",i));
					t = new WriteCSVTask(newPath, job, src, rowStart, offset, formatProperties);
				}
				else {
					t = new WriteCSVTask(path, job, src, rowStart, offset, formatProperties);
				}
				
				tasks.add(t);
				rowStart = rowStart + offset;
				rlen = rlen - offset;
			}
			
			//wait until all tasks have been executed
			pool.invokeAll(tasks);	
			pool.shutdown();
			
			//early error notify in case not all tasks successful
			for(WriteCSVTask rt : tasks) {
				if( !rt.getReturnCode() ) {
					throw new IOException("Parallel write task failed: " + rt.getErrMsg());
				}
			}
		} 
		catch (Exception e) {
			throw new IOException("Parallel write of csv output failed.", e);
		}
	}

	
	/**
	 * 
	 * 
	 */
	private static class WriteCSVTask implements Callable<Object> 
	{
		private JobConf _job = null;
		private MatrixBlock _src = null;
		private Path _path =null;
		private long _rowStart = -1;
		private long _rowNum = -1;
		private FileFormatProperties _formatProperties = null;
		
		private boolean _rc = true;
		private String _errMsg = null;
		
		public WriteCSVTask(Path path, JobConf job, MatrixBlock src, long rowStart, long rowNum, FileFormatProperties formatProperties)
		{
			_path = path;
			_job = job;
			_src = src;
			_rowStart = rowStart;
			_rowNum = rowNum;
			_formatProperties = formatProperties;
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
			FileSystem _fs = FileSystem.get(_job);
	        BufferedWriter bw = null;
	        
			boolean sparse = _src.isInSparseFormat();
			int cols = _src.getNumColumns();

			try
			{
				//for obj reuse and preventing repeated buffer re-allocations
				StringBuilder sb = new StringBuilder();
				bw = new BufferedWriter(new OutputStreamWriter(_fs.create(_path,true)));
				
				CSVFileFormatProperties csvProperties = (CSVFileFormatProperties)_formatProperties;
				csvProperties = (csvProperties==null)? new CSVFileFormatProperties() : csvProperties;
				String delim = csvProperties.getDelim(); //Pattern.quote(csvProperties.getDelim());
				boolean csvsparse = csvProperties.isSparse();
				
				// Write header line, if needed
				if( csvProperties.hasHeader() && _rowStart == 0 ) 
				{
					//write row chunk-wise to prevent OOM on large number of columns
					for( int bj=0; bj<cols; bj+=WriterTextCSV.BLOCKSIZE_J )
					{
						for( int j=bj; j < Math.min(cols,bj+WriterTextCSV.BLOCKSIZE_J); j++) 
						{
							sb.append("C"+ (j+1));
							if ( j < cols-1 )
								sb.append(delim);
						}
						bw.write( sb.toString() );
			            sb.setLength(0);	
					}
					sb.append('\n');
					bw.write( sb.toString() );
		            sb.setLength(0);
				}
				
				// Write data lines
				if( sparse ) //SPARSE
				{	
					SparseRow[] sparseRows = _src.getSparseRows();
					for( int i=(int)_rowStart; i<(_rowStart+_rowNum); i++ )
					{
						//write row chunk-wise to prevent OOM on large number of columns
						int prev_jix = -1;
						if(    sparseRows!=null && i<sparseRows.length 
							&& sparseRows[i]!=null && !sparseRows[i].isEmpty() )
						{
							SparseRow arow = sparseRows[i];
							int alen = arow.size();
							int[] aix = arow.getIndexContainer();
							double[] avals = arow.getValueContainer();
							
							for(int j=0; j < alen; j++) 
							{
								int jix = aix[j];
								
								// output empty fields, if needed
								for( int j2=prev_jix; j2<jix-1; j2++ ) {
									if( !csvsparse )
										sb.append('0');
									sb.append(delim);
								
									//flush buffered string
						            if( j2%WriterTextCSV.BLOCKSIZE_J==0 ){
										bw.write( sb.toString() );
							            sb.setLength(0);
						            }
								}
								
								// output the value (non-zero)
								sb.append( avals[j] );
								if( jix < cols-1)
									sb.append(delim);
								bw.write( sb.toString() );
					            sb.setLength(0);
					            
					            //flush buffered string
					            if( jix%WriterTextCSV.BLOCKSIZE_J==0 ){
									bw.write( sb.toString() );
						            sb.setLength(0);
					            }
					            
								prev_jix = jix;
							}
						}
						
						// Output empty fields at the end of the row.
						// In case of an empty row, output (clen-1) empty fields
						for( int bj=prev_jix+1; bj<cols; bj+=WriterTextCSV.BLOCKSIZE_J )
						{
							for( int j = bj; j < Math.min(cols,bj+WriterTextCSV.BLOCKSIZE_J); j++) {
								if( !csvsparse )
									sb.append('0');
								if( j < cols-1 )
									sb.append(delim);
							}
							bw.write( sb.toString() );
				            sb.setLength(0);	
						}

						sb.append('\n');
						bw.write( sb.toString() ); 
						sb.setLength(0); 
					}
				}
				else //DENSE
				{
					for( int i=(int)_rowStart; i<(_rowStart+_rowNum); i++ )
					{
						//write row chunk-wise to prevent OOM on large number of columns
						for( int bj=0; bj<cols; bj+=WriterTextCSV.BLOCKSIZE_J )
						{
							for( int j=bj; j<Math.min(cols,bj+WriterTextCSV.BLOCKSIZE_J); j++ )
							{
								double lvalue = _src.getValueDenseUnsafe(i, j);
								if( lvalue != 0 ) //for nnz
									sb.append(lvalue);
								else if( !csvsparse ) 
									sb.append('0');
								
								if( j != cols-1 )
									sb.append(delim);
							}
							bw.write( sb.toString() );
				            sb.setLength(0);
						}
						
						sb.append('\n');
						bw.write( sb.toString() ); //same as append
						sb.setLength(0); 
					}
				}
			}
			finally
			{
				IOUtilFunctions.closeSilently(bw);
			}			
			return null;
		}
	}
}

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

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

/**
 * 
 */
public class WriterTextCSVParallel extends WriterTextCSV
{
	public WriterTextCSVParallel( CSVFileFormatProperties props ) {
		super( props );
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
	protected void writeCSVMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, long nnz, CSVFileFormatProperties props )
		throws IOException
	{
		//estimate output size and number of output blocks (min 1)
		int numPartFiles = (int)(OptimizerUtils.estimateSizeTextOutput(src.getNumRows(), src.getNumColumns(), src.getNonZeros(), 
				              OutputInfo.CSVOutputInfo)  / InfrastructureAnalyzer.getHDFSBlockSize());
		numPartFiles = Math.max(numPartFiles, 1);
		
		//determine degree of parallelism
		int numThreads = OptimizerUtils.getParallelTextWriteParallelism();
		numThreads = Math.min(numThreads, numPartFiles);
		
		//create directory for concurrent tasks
		MapReduceTool.createDirIfNotExistOnHDFS(path.toString(), DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		
		//create and execute tasks
		try 
		{
			ExecutorService pool = Executors.newFixedThreadPool(numThreads);
			ArrayList<WriteCSVTask> tasks = new ArrayList<WriteCSVTask>();
			int blklen = (int)Math.ceil((double)rlen / numThreads);
			for(int i=0; i<numThreads & i*blklen<rlen; i++) {
				Path newPath = new Path(path, String.format("0-m-%05d",i));
				tasks.add(new WriteCSVTask(newPath, job, src, i*blklen, (int)Math.min((i+1)*blklen, rlen), props));
			}

			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);	
			pool.shutdown();
			
			//check for exceptions 
			for( Future<Object> task : rt )
				task.get();
		} 
		catch (Exception e) {
			throw new IOException("Failed parallel write of csv output.", e);
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
		private int _rl = -1;
		private int _ru = -1;
		private CSVFileFormatProperties _props = null;
		
		public WriteCSVTask(Path path, JobConf job, MatrixBlock src, int rl, int ru, CSVFileFormatProperties props)
		{
			_path = path;
			_job = job;
			_src = src;
			_rl = rl;
			_ru = ru;
			_props = props;
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
				
				_props = (_props==null)? new CSVFileFormatProperties() : _props;
				String delim = _props.getDelim(); //Pattern.quote(csvProperties.getDelim());
				boolean csvsparse = _props.isSparse();
				
				// Write header line, if needed
				if( _props.hasHeader() && _rl == 0 ) 
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
					for( int i=_rl; i<_ru; i++ )
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
					for( int i=_rl; i<_ru; i++ )
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

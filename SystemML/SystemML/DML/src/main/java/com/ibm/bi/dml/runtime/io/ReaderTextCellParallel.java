/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.FastStringTokenizer;

/**
 * THIS IS AN EXPERIMENTAL IMPLEMENTATION AND NOT USED BY DEFAULT YET.
 * 
 * TODO error handling (message passing between worker threads and master)
 * TODO support for matrix market file format
 * TODO thorough experimental evaluation for dense/sparse, different data sizes
 * TODO clarify unsynchronized double parsing (unsafe vs copy of jdk8 sources)
 * 
 * Notes on differences to sequential textcell reader
 *   * The parallel textcell reader does not support MM files as well and hence will throw 
 *     exceptions if MM file headers are present in the given dataset.
 * 
 */
public class ReaderTextCellParallel extends MatrixReader
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private int _numThreads = 1;
	
	public ReaderTextCellParallel(InputInfo info)
	{
		_numThreads = InfrastructureAnalyzer.getLocalParallelism();
	}
	
	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, estnnz, true);
		
		//prepare file access
		JobConf job = new JobConf();	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname );
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		readTextCellMatrixFromHDFS(path, job, ret, rlen, clen, brlen, bclen);
		
		//finally check if change of sparse/dense block representation required
		if( !ret.isInSparseFormat() )
			ret.recomputeNonZeros();
		if( ret.isInSparseFormat() )
			ret.sortSparseRows();
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
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private void readTextCellMatrixFromHDFS( Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		boolean sparse = dest.isInSparseFormat();
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		
		ExecutorService pool = Executors.newFixedThreadPool(_numThreads);
		InputSplit[] splits = informat.getSplits(job, _numThreads);
		
		try 
		{
			//create read tasks for all splits
			ArrayList<ReadTask> tasks = new ArrayList<ReadTask>();
			for( InputSplit split : splits ){
				ReadTask t = new ReadTask(split, sparse, informat, job, dest, rlen, clen);
				tasks.add(t);
			}
			
			//wait until all tasks have been executed
			pool.invokeAll(tasks);	
			pool.shutdown();
		} 
		catch (InterruptedException e) {
			throw new IOException(e);
		}
	}
	
	/**
	 * 
	 * 
	 */
	public static class ReadTask implements Callable<Object> 
	{
		private InputSplit _split = null;
		private boolean _sparse = false;
		private TextInputFormat _informat = null;
		private JobConf _job = null;
		private MatrixBlock _dest = null;
		private long _rlen = -1;
		private long _clen = -1;
		
		public ReadTask( InputSplit split, boolean sparse, TextInputFormat informat, JobConf job, MatrixBlock dest, long rlen, long clen )
		{
			_split = split;
			_sparse = sparse;
			_informat = informat;
			_job = job;
			_dest = dest;
			_rlen = rlen;
			_clen = clen;
		}

		@Override
		public Object call() throws Exception 
		{
			LongWritable key = new LongWritable();
			Text value = new Text();
			int row = -1;
			int col = -1;
			
			try
			{			
				FastStringTokenizer st = new FastStringTokenizer(' ');
				RecordReader<LongWritable,Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
			
				try
				{
					if( _sparse ) //SPARSE<-value
					{
						CellBuffer buff = new CellBuffer();
						
						while( reader.next(key, value) )
						{
							st.reset( value.toString() ); //reinit tokenizer
							row = st.nextInt() - 1;
							col = st.nextInt() - 1;
							double lvalue = st.nextDoubleForParallel(); //prevent contention
							
							buff.addCell(row, col, lvalue);
							if( buff.size()>=CellBuffer.CAPACITY )
								synchronized( _dest ){ //sparse requires lock
									buff.flushCellBufferToMatrixBlock(_dest);
								}							
						}
					} 
					else //DENSE<-value
					{
						while( reader.next(key, value) )
						{
							st.reset( value.toString() ); //reinit tokenizer
							row = st.nextInt()-1;
							col = st.nextInt()-1;
							double lvalue = st.nextDoubleForParallel(); //prevent contention
							_dest.setValueDenseUnsafe( row, col, lvalue );
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
				//post-mortem error handling and bounds checking
				if( row < 0 || row + 1 > _rlen || col < 0 || col + 1 > _clen )
				{
					throw new RuntimeException("Matrix cell ["+(row+1)+","+(col+1)+"] " +
										  "out of overall matrix range [1:"+_rlen+",1:"+_clen+"].");
				}
				else
				{
					throw new RuntimeException( "Unable to read matrix in text cell format.", ex );
				}
			}
			
			return null;
		}
	}
	
	/**
	 * 
	 * 
	 */
	public static class CellBuffer
	{
		public static final int CAPACITY = 102400; //100K elements 
		
		private int[] _rlen;
		private int[] _clen;
		private double[] _vals;
		private int _pos;
		
		public CellBuffer( )
		{
			_rlen = new int[CAPACITY];
			_clen = new int[CAPACITY];
			_vals = new double[CAPACITY];
			_pos = -1;
		}
		
		public void addCell(int rlen, int clen, double val)
		{
			_pos++;
			_rlen[_pos] = rlen;
			_clen[_pos] = clen;
			_vals[_pos] = val;
		}
		
		public void flushCellBufferToMatrixBlock( MatrixBlock dest )
		{
			for( int i=0; i<=_pos; i++ )
				dest.appendValue(_rlen[i], _clen[i], _vals[i]);
			
			reset();
		}
		
		public int size()
		{
			return _pos+1;
		}
		
		public void reset()
		{
			_pos = -1;
		}
	}
}

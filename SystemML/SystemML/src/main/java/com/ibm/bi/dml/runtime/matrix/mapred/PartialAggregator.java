/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class PartialAggregator extends MMCJMRCache
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long rlen=0;
	private long clen=0;
	private int brlen=0;
	private int bclen=0;
	private long numBlocksInRow=0;
	private long numBlocksInColumn=0;
	private AggregateBinaryOperator operation;

	//local file management
	private boolean memOnly = false; 
	private boolean rowMajor = true;
	
	
	/**
	 * 
	 * @param conf
	 * @param memSize
	 * @param resultRlen
	 * @param resultClen
	 * @param blockRlen
	 * @param blockClen
	 * @param filePrefix
	 * @param inRowMajor
	 * @param op
	 * @param vCls
	 * @throws InstantiationException
	 * @throws IllegalAccessException
	 * @throws IOException
	 */
	public PartialAggregator(JobConf conf, long memSize, long resultRlen, long resultClen, 
			int blockRlen, int blockClen, boolean inRowMajor, 
			AggregateBinaryOperator op, Class<? extends MatrixValue> vCls) 
		throws InstantiationException, IllegalAccessException, IOException
	{
		rlen = resultRlen;
		clen = resultClen;
		brlen = blockRlen;
		bclen = blockClen;
		numBlocksInRow = (long)Math.ceil((double)clen/(double)bclen);
		numBlocksInColumn = (long)Math.ceil((double)rlen/(double)brlen);
		operation = op;
		
		//allocate space for buffer
		//if the buffer space is already larger than the result size, don't need extra space
		long elementSize = 77+8*Math.min(rlen,brlen)*Math.min(clen,bclen)+20+12+12+4;//matrix block, matrix index, pair, integer in the linked list
		int buffCapacity = (int)Math.max(Math.min((memSize/elementSize), (numBlocksInRow*numBlocksInColumn)), 1);
		super.allocateBuffer(buffCapacity, vCls, true);
		
		//local file management (if necessary)
		int n = (int)Math.ceil((double)(numBlocksInRow*numBlocksInColumn)/(double)_bufferCapacity);
		memOnly = (n==1);
		if( !memOnly )
		{
			_job = conf;
			_fs = FileSystem.getLocal(_job);
			rowMajor = inRowMajor;
			_fileN = n;
			super.constructLocalFilePrefix("_partial_aggregator_");
			super.deleteAllWorkingFiles();
		}
	}
	
	/**
	 * 
	 * @param indexes
	 * @param value
	 * @param leftcached
	 * @throws IOException
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public void aggregateToBuffer(MatrixIndexes indexes, MatrixValue value, boolean leftcached) 
		throws IOException, DMLUnsupportedOperationException, DMLRuntimeException
	{
		if( !memOnly )
		{
			int newFileCursor=getFileCursor(indexes);
			if( newFileCursor>=_fileN )
				throw new IOException("indexes: "+indexes+" needs to be put in file #"+newFileCursor+" which exceeds the limit: "+_fileN);
			
			if(_fileCursor!=newFileCursor)
			{	
				super.writeBuffer();
				_fileCursor=newFileCursor;
				super.loadBuffer();
			}
		}
		
		aggregateToBufferHelp(indexes, value, leftcached);
	}
	
	/**
	 * 
	 * @param outputs
	 * @param j
	 * @param reporter
	 * @return
	 * @throws IOException
	 */
	public long outputToHadoop(CollectMultipleConvertedOutputs outputs, int j, Reporter reporter) 
		throws IOException
	{
		long nonZeros=0;
		
		//write the currentBufferSize if something is in memory
		for( Integer ix : _bufferMap.values() )
		{
			outputs.collectOutput(_buffer[ix].getKey(), _buffer[ix].getValue(), j, reporter);
			nonZeros+=_buffer[ix].getValue().getNonZeros();
		}
		
		if( !memOnly ){
			Path path = getFilePath(_fileCursor);
			MapReduceTool.deleteFileIfExistOnHDFS(path, _job);
		}
	
		
		//flush local fs buffer pages to hdfs
		if( !memOnly )
			for(int i=0; i<_fileN; i++)
				if( i != _fileCursor ){ //current cursor already flushed
					Path path = getFilePath(i);
					nonZeros+=copyFileContentAndDelete(path, outputs, j, reporter);
				}
		
		return nonZeros;
	}
	
	/**
	 * 
	 * @throws IOException
	 */
	public void close() 
		throws IOException
	{
		if( !memOnly )
			super.deleteAllWorkingFiles();
	}
	
	/**
	 * 
	 * @param indexes
	 * @param value
	 * @param leftcached
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private void aggregateToBufferHelp(MatrixIndexes indexes, MatrixValue value, boolean leftcached) 
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		Integer ix = _bufferMap.get( indexes );
		if( ix != null ) //agg into existing block 
		{
			_buffer[ix].getValue().binaryOperationsInPlace(operation.aggOp.increOp, value);
		}
		else //add as new block
		{
			addToBuffer(indexes, value);
		}
	}
	
	/**
	 * 
	 * @param indexes
	 * @param value
	 */
	private void addToBuffer(MatrixIndexes indexes, MatrixValue value) 
	{
		if(_bufferSize>=_buffer.length)
			throw new RuntimeException("indexes: "+indexes+" needed to be put in postition: "+_bufferSize+" which exceeds the buffer size: "+_buffer.length);
		
		//add to the end
		_buffer[_bufferSize].getKey().setIndexes(indexes);
		_buffer[_bufferSize].getValue().copy(value);
		_bufferMap.put(_buffer[_bufferSize].getKey(), _bufferSize);
		_bufferSize++;		
	}
	

	/**
	 * 
	 * @param indexes
	 * @return
	 */
	private int getFileCursor(MatrixIndexes indexes) 
	{
		if(rowMajor)
			return (int)(((indexes.getRowIndex()-1)*numBlocksInRow+indexes.getColumnIndex()-1)/_bufferCapacity);
		else
			return (int)(((indexes.getColumnIndex()-1)*numBlocksInColumn+indexes.getRowIndex()-1)/_bufferCapacity);
	}
	

	/**
	 * 
	 * @param path
	 * @param outputs
	 * @param j
	 * @param reporter
	 * @return
	 * @throws IOException
	 */
	private long copyFileContentAndDelete(Path path, CollectMultipleConvertedOutputs outputs, int j, Reporter reporter) 
		throws IOException 
	{
		long nonZeros=0;
		if(_fs.exists(path))
		{
			_bufferSize = LocalFileUtils.readBlockSequenceFromLocal(path.toString(), _buffer, _bufferMap);
			for( int i=0; i<_bufferSize; i++ )
			{
				outputs.collectOutput(_buffer[i].getKey(), _buffer[i].getValue(), j, reporter);
				nonZeros+=_buffer[i].getValue().getNonZeros();	
			}
			MapReduceTool.deleteFileIfExistOnHDFS(path, _job);
		}
		return nonZeros;
	}
}

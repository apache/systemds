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


package org.apache.sysml.runtime.matrix.mapred;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.sysml.runtime.util.MapReduceTool;


public class PartialAggregator extends MMCJMRCache
{
	
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
	 * @throws DMLRuntimeException
	 */
	public void aggregateToBuffer(MatrixIndexes indexes, MatrixValue value, boolean leftcached) 
		throws IOException, DMLRuntimeException
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
	 * @throws DMLRuntimeException
	 */
	private void aggregateToBufferHelp(MatrixIndexes indexes, MatrixValue value, boolean leftcached) 
		throws DMLRuntimeException 
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

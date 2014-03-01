/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class PartialAggregator 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int bufferCapacity=0;
	private int currentBufferSize=0;
	private Pair<MatrixIndexes,MatrixValue>[] buffer = null;
	private HashMap<MatrixIndexes,Integer> bufferMap = null;
	private long rlen=0;
	private long clen=0;
	private int brlen=0;
	private int bclen=0;
	private long numBlocksInRow=0;
	private long numBlocksInColumn=0;
	private AggregateBinaryOperator operation;
	private Class<? extends MatrixValue> valueClass;

	//local file management
	private boolean memOnly = false; 
	private boolean rowMajor = true;
	private JobConf job = null;
	private FileSystem fs = null;
	private int fileCursor = -1;
	private Path[] files = null;
	
	
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
	@SuppressWarnings("unchecked")
	public PartialAggregator(JobConf conf, long memSize, long resultRlen, long resultClen, 
			int blockRlen, int blockClen, String filePrefix, boolean inRowMajor, 
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
		valueClass = vCls;
		
		//allocate space for buffer
		//if the buffer space is already larger than the result size, don't need extra space
		long elementSize = 77+8*Math.min(rlen,brlen)*Math.min(clen,bclen)+20+12+12+4;//matrix block, matrix index, pair, integer in the linked list
		bufferCapacity = (int)Math.max(Math.min((memSize/elementSize), (numBlocksInRow*numBlocksInColumn)), 1);
		buffer = new Pair[bufferCapacity];
		for(int i=0; i<bufferCapacity; i++)
			buffer[i] = new Pair<MatrixIndexes, MatrixValue>(new MatrixIndexes(), valueClass.newInstance());
		bufferMap = new HashMap<MatrixIndexes, Integer>();
		
		//local file management (if necessary)
		int n = (int)Math.ceil((double)(numBlocksInRow*numBlocksInColumn)/(double)bufferCapacity);
		memOnly = (n==1);
		if( !memOnly )
		{
			job = conf;
			fs = FileSystem.getLocal(job);
			rowMajor = inRowMajor;
			files = new Path[n];
			String hadoopLocalDir=job.get("mapred.local.dir").split(",")[0];
			for(int i=0; i<n; i++)
			{
				files[i]=new Path(hadoopLocalDir, filePrefix+"_partial_aggregator_"+i);
				MapReduceTool.deleteFileIfExistOnLFS(files[i], job);
			}
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
			if(newFileCursor>=files.length)
			{
				throw new IOException("indexes: "+indexes+" needs to be put in file #"+newFileCursor+" which exceeds the limit: "+files.length);
			}
			if(fileCursor!=newFileCursor)
			{	
				writeBuffer();
				fileCursor=newFileCursor;
				loadBuffer();
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
		for( Integer ix : bufferMap.values() )
		{
			outputs.collectOutput(buffer[ix].getKey(), buffer[ix].getValue(), j, reporter);
			nonZeros+=buffer[ix].getValue().getNonZeros();
		}
		if( !memOnly )
			MapReduceTool.deleteFileIfExistOnHDFS(files[fileCursor], job);
	
		
		//flush local fs buffer pages to hdfs
		if( !memOnly )
			for(int i=0; i<files.length; i++)
				if( i != fileCursor ) //current cursor already flushed
					nonZeros+=copyFileContentAndDelete(files[i], outputs, j, reporter);
		
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
			for(Path file : files)
				MapReduceTool.deleteFileIfExistOnLFS(file, job);
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
		Integer ix = bufferMap.get( indexes );
		if( ix != null ) //agg into existing block 
		{
			buffer[ix].getValue().binaryOperationsInPlace(operation.aggOp.increOp, value);
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
		if(currentBufferSize>=buffer.length)
			throw new RuntimeException("indexes: "+indexes+" needed to be put in postition: "+currentBufferSize+" which exceeds the buffer size: "+buffer.length);
		
		//add to the end
		buffer[currentBufferSize].getKey().setIndexes(indexes);
		buffer[currentBufferSize].getValue().copy(value);
		bufferMap.put(buffer[currentBufferSize].getKey(), currentBufferSize);
		currentBufferSize++;		
	}
	
	/**
	 * 
	 * @return
	 */
	public HashMap<MatrixIndexes,Integer> getBufferMap()
	{
		return bufferMap;
	}
	

	/**
	 * 
	 * @param indexes
	 * @return
	 */
	private int getFileCursor(MatrixIndexes indexes) 
	{
		if(rowMajor)
			return (int)(((indexes.getRowIndex()-1)*numBlocksInRow+indexes.getColumnIndex()-1)/bufferCapacity);
		else
			return (int)(((indexes.getColumnIndex()-1)*numBlocksInColumn+indexes.getRowIndex()-1)/bufferCapacity);
	}
	
	/**
	 * 
	 * @throws IOException
	 */
	private void loadBuffer() 
		throws IOException
	{
		currentBufferSize=0;
		bufferMap.clear();
		
		if(fs.exists(files[fileCursor]))
		{
			currentBufferSize = LocalFileUtils.readBlockSequenceFromLocal(files[fileCursor].toString(), buffer, bufferMap);
			
			/*
			SequenceFile.Reader reader=new SequenceFile.Reader(fs, files[fileCursor], job);
			try
			{
				while(currentBufferSize<bufferCapacity && reader.next(buffer[currentBufferSize].getKey(), buffer[currentBufferSize].getValue()))
				{
					bufferMap.put( buffer[currentBufferSize].getKey(), 
							       currentBufferSize );
					currentBufferSize++;
				}
			}
			finally
			{
				if( reader!=null )
					reader.close();
			}
			*/
		}
	}
	
	/**
	 * 
	 * @throws IOException
	 */
	private void writeBuffer() 
		throws IOException 
	{
		if(fileCursor<0 || currentBufferSize<=0)
			return;
		
		//the old file will be overwritten
		LocalFileUtils.writeBlockSequenceToLocal(files[fileCursor].toString(), buffer, currentBufferSize);
		
		/*
		SequenceFile.Writer writer=new SequenceFile.Writer(fs, job, files[fileCursor], MatrixIndexes.class, valueClass);
		try
		{
			for( Integer ix : bufferMap.values() )
			{
				writer.append( buffer[ix].getKey(), 
						       buffer[ix].getValue() );
			}
		}
		finally
		{
			if( writer != null )
				writer.close();
		}
		*/
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
		if(fs.exists(path))
		{
			
			currentBufferSize = LocalFileUtils.readBlockSequenceFromLocal(path.toString(), buffer, bufferMap);
			for( int i=0; i<currentBufferSize; i++ )
			{
				outputs.collectOutput(buffer[i].getKey(), buffer[i].getValue(), j, reporter);
				nonZeros+=buffer[i].getValue().getNonZeros();	
			}
			MapReduceTool.deleteFileIfExistOnHDFS(path, job);
			
			/*
			SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
			try
			{
				while(reader.next(buffer[0].getKey(), buffer[0].getValue()))
				{
				//	System.out.println(buffer[0].getKey()+" -- "+buffer[0].getValue());
					outputs.collectOutput(buffer[0].getKey(), buffer[0].getValue(), j, reporter);
					nonZeros+=buffer[0].getValue().getNonZeros();
				}
			}
			finally
			{
				if( reader!=null )
					reader.close();
				MapReduceTool.deleteFileIfExistOnHDFS(path, job);
			}
			*/
		}
		return nonZeros;
	}
}

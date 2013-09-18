package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Iterator;
import java.util.Vector;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class MMCJMRReducerWithAggregator extends MMCJMRCombinerReducerBase 
implements Reducer<TaggedFirstSecondIndexes, MatrixValue, Writable, Writable>{

	private class RemainIndexValue
	{
		public long remainIndex;
		public MatrixValue value;
		
	//	private Class<? extends MatrixValue> valueClass;
		public RemainIndexValue(Class<? extends MatrixValue> cls) throws Exception
		{
			remainIndex=-1;
			value=cls.newInstance();
		}
		public RemainIndexValue(long ind, MatrixValue b) throws Exception
		{
			remainIndex=ind;
			Class<? extends MatrixValue> cls=b.getClass();
			value=cls.newInstance();
			value.copy(b);
		}
		public void set(long ind, MatrixValue b)
		{
			remainIndex=ind;
			value.copy(b);
		}
	}
	
	//in memory cache to hold the records from one input matrix for the cross product
	private Vector<RemainIndexValue> cache=new Vector<RemainIndexValue>(100);
	private int cacheSize=0;
	
	private PartialAggregator aggregator;
	
	//variables to keep track of the flow
	private double prevFirstIndex=-1;
	private int prevTag=-1;
	
	//temporary variable
	private MatrixIndexes indexesbuffer=new MatrixIndexes();
	private RemainIndexValue remainingbuffer=null;
	private MatrixValue valueBuffer=null;
	
	private boolean outputDummyRecords = false;
	
	@Override
	public void reduce(TaggedFirstSecondIndexes indexes, Iterator<MatrixValue> values,
			OutputCollector<Writable, Writable> out,
			Reporter report) throws IOException 
	{
		long start=System.currentTimeMillis();
		
		commonSetup(report);
		
		//perform aggregate		
		MatrixValue aggregateValue=performAggregateInstructions(indexes, values);		
		if(aggregateValue==null)
			return;
		
		int tag=indexes.getTag();
		long firstIndex=indexes.getFirstIndex();
		long secondIndex=indexes.getSecondIndex();
		
		//for a different k
		if( prevFirstIndex!=firstIndex ) 
		{
			resetCache();
			prevFirstIndex=firstIndex;
		}
		else if(prevTag>tag)
			throw new RuntimeException("tag is not ordered correctly: "+prevTag+" > "+tag);
		
		remainingbuffer.set(secondIndex, aggregateValue);
		prevTag=tag;
		
		//perform cross-product binagg
		processJoin(tag, remainingbuffer);
		
		report.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}

	private void processJoin(int tag, RemainIndexValue rValue) 
		throws IOException
	{
		try
		{
			//for the cached matrix
			if(tag==0)
			{
				if(cacheSize<cache.size())
					cache.get(cacheSize).set(rValue.remainIndex, rValue.value);
				else
					cache.add(new RemainIndexValue(rValue.remainIndex, rValue.value));
				cacheSize++;
			}
			else //for the probing matrix
			{
				for(int i=0; i<cacheSize; i++)
				{
					RemainIndexValue left, right;
					if(tagForLeft==0)
					{
						left=cache.get(i);
						right=rValue;
					}else
					{
						right=cache.get(i);
						left=rValue;
					}
					
					indexesbuffer.setIndexes(left.remainIndex, right.remainIndex);
					OperationsOnMatrixValues.performAggregateBinaryIgnoreIndexes(left.value, right.value, valueBuffer, 
							                               (AggregateBinaryOperator)aggBinInstruction.getOperator());
					
					//if(valueBuffer.getNonZeros()>0)
						aggregator.aggregateToBuffer(indexesbuffer, valueBuffer, tagForLeft==0);
				}
			}
		}
		catch(Exception ex)
		{
			throw new IOException(ex);
		}
	}
	
	private void resetCache() {
		cacheSize=0;
	}

	//output the records in the outCache.
	public void close() throws IOException
	{
		if(cachedReporter!=null)
		{
			long start=System.currentTimeMillis();
			byte resultTag=resultIndexes[0];
			resultsNonZeros[0]+=aggregator.outputToHadoop(collectFinalMultipleOutputs, 0, cachedReporter);
			cachedReporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
		}
	    //aggregator.close();
		
		//handle empty block output (on first reduce task only)
		if( outputDummyRecords ) //required for rejecting empty blocks in mappers
		{
			long rlen = dim1.numRows;
			long clen = dim2.numColumns;
			int brlen = dim1.numRowsPerBlock;
			int bclen = dim2.numColumnsPerBlock;
			MatrixIndexes tmpIx = new MatrixIndexes();
			MatrixBlock tmpVal = new MatrixBlock();
			for(long i=0, r=1; i<rlen; i+=brlen, r++)
				for(long j=0, c=1; j<clen; j+=bclen, c++)
				{
					int realBrlen=(int)Math.min((long)brlen, rlen-(r-1)*brlen);
					int realBclen=(int)Math.min((long)bclen, clen-(c-1)*bclen);
					
					tmpIx.setIndexes(r, c);
					tmpVal.reset(realBrlen,realBclen);
					collectFinalMultipleOutputs.collectOutput(tmpIx, tmpVal, 0, cachedReporter);
				}
		}
		
		super.close();
	}
	
	public void configure(JobConf job)
	{
	//	LOG.info("starts configure: ");
	//	LOG.info("Memory stats: "+Runtime.getRuntime().totalMemory()+", "+Runtime.getRuntime().freeMemory());
		
		super.configure(job);
		if(resultIndexes.length>1)
			throw new RuntimeException("MMCJMR only outputs one result");
		
		outputDummyRecords = MapReduceTool.getUniqueKeyPerTask(job, false).equals("0");
		
		try {
			//valueBuffer=valueClass.newInstance();
			valueBuffer=buffer;
			remainingbuffer=new RemainIndexValue(valueClass);
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
		
		int outbufferSize=MRJobConfiguration.getPartialAggCacheSize(job);
		try {
			aggregator=new PartialAggregator(job, (long)outbufferSize, dim1.numRows, dim2.numColumns, 
					dim1.numRowsPerBlock, dim2.numColumnsPerBlock, MapReduceTool.getGloballyUniqueName(job), (tagForLeft!=0), 
					(AggregateBinaryOperator) aggBinInstruction.getOperator(), valueClass);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		//LOG.info("Memory stats: "+Runtime.getRuntime().totalMemory()+", "+Runtime.getRuntime().freeMemory());
	}
}

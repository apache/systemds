package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;
import java.util.Map.Entry;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class MMCJMRReducer extends MMCJMRCombinerReducerBase 
implements Reducer<TaggedFirstSecondIndexes, MatrixValue, Writable, Writable>{

	private class RemainIndexValue
	{
		public long remainIndex;
		public MatrixValue value;
		private Class<? extends MatrixValue> valueClass;
		public RemainIndexValue(Class<? extends MatrixValue> cls) throws Exception
		{
			remainIndex=-1;
			valueClass=cls;
			value=valueClass.newInstance();
		}
		public RemainIndexValue(long ind, MatrixValue b) throws Exception
		{
			remainIndex=ind;
			valueClass=b.getClass();
			value=valueClass.newInstance();
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
	
	//to cache output, so that we can do some partial aggregation here
	private int OUT_CACHE_SIZE;
	private HashMap<MatrixIndexes, MatrixValue> outCache;
	
	//variables to keep track of the flow
	private double prevFirstIndex=-1;
	private int prevTag=-1;
	
	//temporary variable
	private MatrixIndexes indexesbuffer=new MatrixIndexes();
	private RemainIndexValue remainingbuffer=null;
	private MatrixValue valueBuffer=null;
	
	private long count=0;
	
	@Override
	public void reduce(TaggedFirstSecondIndexes indexes, Iterator<MatrixValue> values,
			OutputCollector<Writable, Writable> out,
			Reporter report) throws IOException {
		long start=System.currentTimeMillis();
//		LOG.info("---------- key: "+indexes);
		
		commonSetup(report);
		
		//perform aggregate
		MatrixValue aggregateValue=performAggregateInstructions(indexes, values);
		
		if(aggregateValue==null)
			return;
		
		int tag=indexes.getTag();
		long firstIndex=indexes.getFirstIndex();
		long secondIndex=indexes.getSecondIndex();
		
		//for a differe k
		if(prevFirstIndex!=firstIndex)
		{
			resetCache();
			prevFirstIndex=firstIndex;
		}else if(prevTag>tag)
			throw new RuntimeException("tag is not ordered correctly: "+prevTag+" > "+tag);
		
		remainingbuffer.set(secondIndex, aggregateValue);
		try {
			processJoin(tag, remainingbuffer);
		} catch (Exception e) {
			throw new IOException(e);
		}
		prevTag=tag;
		
		report.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}

	private void processJoin(int tag, RemainIndexValue rValue) 
	throws Exception
	{

		//for the cached matrix
		if(tag==0)
		{
			addToCache(rValue, tag);
	//		LOG.info("put in the buffer for left matrix");
	//		LOG.info(rblock.block.toString());
		}
		else//for the probing matrix
		{
			//LOG.info("process join with block size: "+rValue.value.getNumRows()+" X "+rValue.value.getNumColumns()+" nonZeros: "+rValue.value.getNonZeros());
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
				try {
					OperationsOnMatrixValues.performAggregateBinaryIgnoreIndexes(left.value, 
							right.value, valueBuffer, (AggregateBinaryOperator)aggBinInstruction.getOperator());
				} catch (DMLUnsupportedOperationException e) {
					throw new IOException(e);
				}
				
		/*		if(count%1000==0)
				{
					LOG.info("left block: sparse format="+left.value.isInSparseFormat()+
							", dimension="+left.value.getNumRows()+"x"+left.value.getNumColumns()
							+", nonZeros="+left.value.getNonZeros());
					
					LOG.info("right block: sparse format="+right.value.isInSparseFormat()+
							", dimension="+right.value.getNumRows()+"x"+right.value.getNumColumns()
							+", nonZeros="+right.value.getNonZeros());
					
					LOG.info("result block: sparse format="+valueBuffer.isInSparseFormat()+
							", dimension="+valueBuffer.getNumRows()+"x"+valueBuffer.getNumColumns()
							+", nonZeros="+valueBuffer.getNonZeros());
				}
				count++;*/
				
/*				LOG.info("left block");
				LOG.info(cache.get(i).block.toString());
				LOG.info("right block");
				LOG.info(rblock.block.toString());
				LOG.info("output block");
				LOG.info(buffer.toString());
*/
				//if(valueBuffer.getNonZeros()>0)
					collectOutput(indexesbuffer, valueBuffer);
			}
		}
	}
	
	private void collectOutput(MatrixIndexes indexes,
			MatrixValue value_out) 
	throws Exception 
	{
		MatrixValue value=outCache.get(indexes);
		
		try {
			if(value!=null)
			{
				//LOG.info("********** oops, should not run this code1 ***********");
/*				LOG.info("the output is in the cache");
				LOG.info("old block");
				LOG.info(block.toString());
*/			
				value.binaryOperationsInPlace(((AggregateBinaryOperator)aggBinInstruction.getOperator()).aggOp.increOp, 
						value_out);
				
/*				LOG.info("add block");
				LOG.info(block_out.toString());
				LOG.info("result block");
				LOG.info(block.toString());
*/				
			}
			else if(outCache.size()<OUT_CACHE_SIZE)
			{
				//LOG.info("********** oops, should not run this code2 ***********");
				value=valueClass.newInstance();
				value.reset(value_out.getNumRows(), value_out.getNumColumns(), value.isInSparseFormat());
				value.binaryOperationsInPlace(((AggregateBinaryOperator)aggBinInstruction.getOperator()).aggOp.increOp, 
						value_out);
				outCache.put(new MatrixIndexes(indexes), value);
				
/*				LOG.info("the output is not in the cache");
				LOG.info("result block");
				LOG.info(block.toString());
*/
			}else
			{
				realWriteToCollector(indexes, value_out);
			}
		} catch (DMLUnsupportedOperationException e) {
			throw new IOException(e);
		}
	}

	private void resetCache() {
		cacheSize=0;
	}

	private void addToCache(RemainIndexValue rValue, int tag) throws Exception {
	
		//LOG.info("add to cache with block size: "+rValue.value.getNumRows()+" X "+rValue.value.getNumColumns()+" nonZeros: "+rValue.value.getNonZeros());
		if(cacheSize<cache.size())
			cache.get(cacheSize).set(rValue.remainIndex, rValue.value);
		else
			cache.add(new RemainIndexValue(rValue.remainIndex, rValue.value));
		cacheSize++;
	}
	
	//output the records in the outCache.
	public void close() throws IOException
	{
		long start=System.currentTimeMillis();
		Iterator<Entry<MatrixIndexes, MatrixValue>> it=outCache.entrySet().iterator();
		while(it.hasNext())
		{
			Entry<MatrixIndexes, MatrixValue> entry=it.next();
			realWriteToCollector(entry.getKey(), entry.getValue());
		}
		if(cachedReporter!=null)
			cachedReporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
		super.close();
	}
	
	public void realWriteToCollector(MatrixIndexes indexes, MatrixValue value) throws IOException
	{
		collectOutput_N_Increase_Counter(indexes, value, 0, cachedReporter);
//		LOG.info("--------- output: "+indexes+" <--> "+block);
		
	/*	if(count%1000==0)
		{
			LOG.info("result block: sparse format="+value.isInSparseFormat()+
					", dimension="+value.getNumRows()+"x"+value.getNumColumns()
					+", nonZeros="+value.getNonZeros());
		}
		count++;*/
		
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
		if(resultIndexes.length>1)
			throw new RuntimeException("MMCJMR only outputs one result");
		
		try {
			//valueBuffer=valueClass.newInstance();
			valueBuffer=buffer;
			remainingbuffer=new RemainIndexValue(valueClass);
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
		
		int blockRlen=dim1.numRowsPerBlock;
		int blockClen=dim2.numColumnsPerBlock;
		int elementSize=(int)Math.ceil((double)(77+8*blockRlen*blockClen+20+12)/0.75);
		OUT_CACHE_SIZE=MRJobConfiguration.getPartialAggCacheSize(job)/elementSize;
		outCache=new HashMap<MatrixIndexes, MatrixValue>(OUT_CACHE_SIZE);
	}
}

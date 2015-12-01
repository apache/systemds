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


package org.apache.sysml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.data.TaggedFirstSecondIndexes;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.util.MapReduceTool;


public class MMCJMRReducer extends MMCJMRCombinerReducerBase 
implements Reducer<TaggedFirstSecondIndexes, MatrixValue, Writable, Writable>
{
	
	private static class RemainIndexValue
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
	private ArrayList<RemainIndexValue> cache=new ArrayList<RemainIndexValue>(100);
	private int cacheSize=0;
	
	//to cache output, so that we can do some partial aggregation here
	private long OUT_CACHE_SIZE;
	private HashMap<MatrixIndexes, MatrixValue> outCache;
	
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
		
		//for a different k
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
		
		//handle empty block output (on first reduce task only)
		if( outputDummyRecords ) //required for rejecting empty blocks in mappers
		{
			long rlen = dim1.getRows();
			long clen = dim2.getCols();
			int brlen = dim1.getRowsPerBlock();
			int bclen = dim2.getColsPerBlock();
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
		
		outputDummyRecords = MapReduceTool.getUniqueKeyPerTask(job, false).equals("0");
		
		try {
			//valueBuffer=valueClass.newInstance();
			valueBuffer=buffer;
			remainingbuffer=new RemainIndexValue(valueClass);
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
		
		int blockRlen=dim1.getRowsPerBlock();
		int blockClen=dim2.getColsPerBlock();
		int elementSize=(int)Math.ceil((double)(77+8*blockRlen*blockClen+20+12)/0.75);
		OUT_CACHE_SIZE=((long)OptimizerUtils.getLocalMemBudget() //current jvm max mem
				       -MRJobConfiguration.getMMCJCacheSize(job))/elementSize;
		outCache=new HashMap<MatrixIndexes, MatrixValue>(1024);
	}
}

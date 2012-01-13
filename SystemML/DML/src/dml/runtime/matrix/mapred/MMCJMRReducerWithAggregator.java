package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Iterator;
import java.util.Vector;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import dml.runtime.matrix.operators.AggregateBinaryOperator;
import dml.runtime.util.MapReduceTool;
import dml.utils.DMLUnsupportedOperationException;

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
		//System.out.println("read "+indexes+ " value: "+aggregateValue);
		
		if(aggregateValue==null)
			return;
		
		int tag=indexes.getTag();
		long firstIndex=indexes.getFirstIndex();
		long secondIndex=indexes.getSecondIndex();
		
		//LOG.info("****** now k="+firstIndex);
		//for a differe k
		if(prevFirstIndex!=firstIndex)
		{
			resetCache();
			prevFirstIndex=firstIndex;
			aggregator.startOver();
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
			//	if(valueBuffer.getNonZeros()>0)
					aggregator.aggregateToBuffer(indexesbuffer, valueBuffer, tagForLeft==0);
			}
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
		{
		//	LOG.info("before add to cache: ");
		//	LOG.info("Memory stats: "+Runtime.getRuntime().totalMemory()+", "+Runtime.getRuntime().freeMemory());
			cache.add(new RemainIndexValue(rValue.remainIndex, rValue.value));
		//	LOG.info("after add to cache: ");
		//	LOG.info("Memory stats: "+Runtime.getRuntime().totalMemory()+", "+Runtime.getRuntime().freeMemory());
		}
		cacheSize++;
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
	//	aggregator.close();
		super.close();
	}
	
	public void configure(JobConf job)
	{
	//	LOG.info("starts configure: ");
	//	LOG.info("Memory stats: "+Runtime.getRuntime().totalMemory()+", "+Runtime.getRuntime().freeMemory());
		
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

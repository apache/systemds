package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.io.TaggedMatrixValue;
import dml.runtime.matrix.io.TripleIndexes;
import dml.runtime.matrix.operators.AggregateBinaryOperator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;


public class MMRJMRReducer extends ReduceBase
implements Reducer<TripleIndexes, TaggedMatrixValue, MatrixIndexes, MatrixValue>{

	private Reporter cachedReporter=null;
	private MatrixValue resultblock=null;
	private MatrixIndexes aggIndexes=new MatrixIndexes();
	private TripleIndexes prevIndexes=new TripleIndexes(-1, -1, -1);
	private boolean firsttime=true;
	//aggregate binary instruction for the mmrj
	protected AggregateBinaryInstruction aggBinInstruction=null;
//	private MatrixIndexes indexBuf=new MatrixIndexes();
	
	@Override
	public void reduce(TripleIndexes triple, Iterator<TaggedMatrixValue> values,
			OutputCollector<MatrixIndexes, MatrixValue> out, Reporter report)
			throws IOException {
		long start=System.currentTimeMillis();
	//	System.out.println("~~~~~ group: "+triple);
		commonSetup(report);
		
		//output previous results if needed
		if(prevIndexes.getFirstIndex()!=triple.getFirstIndex() 
				|| prevIndexes.getSecondIndex()!=triple.getSecondIndex())
		{
		//	System.out.println("cacheValues before processReducerInstructions: \n"+cachedValues);
			//perform mixed operations
			processReducerInstructions();
			
	//		System.out.println("cacheValues before output: \n"+cachedValues);
			//output results
			outputResultsFromCachedValues(report);
			cachedValues.reset();
		}else
		{
			//clear the buffer
	//		System.out.println("cacheValues before remore: \n"+cachedValues);
			cachedValues.remove(aggBinInstruction.input1);
	//		System.out.println("cacheValues after remore: "+aggBinInstruction.input1+"\n"+cachedValues);
			cachedValues.remove(aggBinInstruction.input2);
	//		System.out.println("cacheValues after remore: "+aggBinInstruction.input2+"\n"+cachedValues);
		}
		
		//perform aggregation first
		aggIndexes.setIndexes(triple.getFirstIndex(), triple.getSecondIndex());
		processAggregateInstructions(aggIndexes, values);
		
	//	System.out.println("cacheValues after aggregation: \n"+cachedValues);
		
		//perform aggbinary for this group
		processAggBinaryPerGroup(aggIndexes);
		
	//	System.out.println("cacheValues after aggbinary: \n"+cachedValues);
				
		prevIndexes.setIndexes(triple);
		
		report.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}
	
	//perform pairwise aggregate binary, and added to the aggregates
	private void processAggBinaryPerGroup(MatrixIndexes indexes) throws IOException
	{
		IndexedMatrixValue left = cachedValues.get(aggBinInstruction.input1);
		IndexedMatrixValue right= cachedValues.get(aggBinInstruction.input2);
	//	System.out.println("left: \n"+left.getValue());
	//	System.out.println("right: \n"+right.getValue());
		if(left!=null && right!=null)
		{
			try {
				resultblock=left.getValue().aggregateBinaryOperations(left.getValue(), right.getValue(), 
						resultblock, (AggregateBinaryOperator) aggBinInstruction.getOperator());
		//		System.out.println("resultblock: \n"+resultblock);
				IndexedMatrixValue out=cachedValues.get(aggBinInstruction.output);
				if(out==null)
				{
					out=cachedValues.holdPlace(aggBinInstruction.output, valueClass);
					out.getIndexes().setIndexes(indexes);
					OperationsOnMatrixValues.startAggregation(out.getValue(), null, ((AggregateBinaryOperator) aggBinInstruction.getOperator()).aggOp, 
							resultblock.getNumRows(), resultblock.getNumColumns(), resultblock.isInSparseFormat(), false);
				}
				OperationsOnMatrixValues.incrementalAggregation(out.getValue(), null, resultblock, 
						((AggregateBinaryOperator) aggBinInstruction.getOperator()).aggOp, false);

		//		System.out.println("agg: \n"+out.getValue());
			} catch (Exception e) {
				throw new IOException(e);
			}
		}
	}
	
	public void close() throws IOException
	{
		long start=System.currentTimeMillis();
		
//		System.out.println("cacheValues before processReducerInstructions: \n"+cachedValues);
		//perform mixed operations
		processReducerInstructions();
		
//		System.out.println("cacheValues before output: \n"+cachedValues);
		
		//output results
		outputResultsFromCachedValues(cachedReporter);
		
		if(cachedReporter!=null)
			cachedReporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
		super.close();
	}
	

	public void configure(JobConf job)
	{
		super.configure(job);
		AggregateBinaryInstruction[] ins;
		try {
			ins = MRJobConfiguration.getAggregateBinaryInstructions(job);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		if(ins.length!=1)
			throw new RuntimeException("MMRJ only perform one aggregate binary instruction");
		aggBinInstruction=ins[0];
	}

}

package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixPackedCell;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.TaggedMatrixValue;

public class GMRReducer extends ReduceBase
implements Reducer<MatrixIndexes, TaggedMatrixValue, MatrixIndexes, MatrixValue>{
	
	MatrixValue realOutValue;
	public void reduce(MatrixIndexes indexes,
			Iterator<TaggedMatrixValue> values,
			OutputCollector<MatrixIndexes, MatrixValue> out,
			Reporter reporter) throws IOException {
		
		long start=System.currentTimeMillis();
		commonSetup(reporter);
		
		cachedValues.reset();
	//	LOG.info("before aggregation: \n"+cachedValues);
		//perform aggregate operations first
		processAggregateInstructions(indexes, values);
		
	//	LOG.info("after aggregation: \n"+cachedValues);
		
		//perform mixed operations
		processReducerInstructions();
		
	//	LOG.info("after mixed operations: \n"+cachedValues);

		//output the final result matrices
		outputResultsFromCachedValuesForGMR(reporter);

		reporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}
	
	protected void outputResultsFromCachedValuesForGMR(Reporter reporter) throws IOException
	{
		for(int i=0; i<resultIndexes.length; i++)
		{
			byte output=resultIndexes[i];
			IndexedMatrixValue outValue=cachedValues.get(output);
			if(outValue==null)
				continue;
			if(valueClass.equals(MatrixPackedCell.class))
			{
				realOutValue.copy(outValue.getValue());
				collectOutput_N_Increase_Counter(outValue.getIndexes(), 
						realOutValue, i, reporter);
			}
			else
				collectOutput_N_Increase_Counter(outValue.getIndexes(), 
					outValue.getValue(), i, reporter);
	//		LOG.info("output: "+outValue.getIndexes()+" -- "+outValue.getValue()+" ~~ tag: "+output);
		//	System.out.println("Reducer output: "+outValue.getIndexes()+" -- "+outValue.getValue()+" ~~ tag: "+output);
		}
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
		try {
			realOutValue=valueClass.newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		//this is to make sure that aggregation works for GMR
		if(valueClass.equals(MatrixCell.class))
			valueClass=MatrixPackedCell.class;
	}
}

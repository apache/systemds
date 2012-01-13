package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Iterator;
import java.util.Vector;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.TaggedFirstSecondIndexes;

public class MMCJMRCombiner extends MMCJMRCombinerReducerBase 
implements Reducer<TaggedFirstSecondIndexes, MatrixValue, TaggedFirstSecondIndexes, MatrixValue>{

	public void reduce(TaggedFirstSecondIndexes indexes, Iterator<MatrixValue> values,
			OutputCollector<TaggedFirstSecondIndexes, MatrixValue> out,
			Reporter report) throws IOException {
	
		//perform aggregate
		MatrixValue aggregateValue=performAggregateInstructions(indexes, values);
		
		if(aggregateValue!=null)
			out.collect(indexes, aggregateValue);
		
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
	}
}

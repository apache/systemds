package dml.runtime.test.numericalStability;

import java.io.IOException;
import java.util.Vector;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Tagged;
import dml.runtime.matrix.io.WeightedCell;
import dml.runtime.matrix.io.WeightedPair;

public class GroupedAggBigDecimalMRMapper extends MapReduceBase
implements Mapper<MatrixIndexes, WeightedPair, IntWritable, DoubleWritable>{
	
	//block instructions that need to be performed in part by mapper
	private DoubleWritable outKeyValue=new DoubleWritable();
	private IntWritable outKey=new IntWritable();
	private DoubleWritable outValue=new DoubleWritable();
	
	@Override
	public void map(MatrixIndexes index, WeightedPair wpair,
			OutputCollector<IntWritable, DoubleWritable> out,
			Reporter reporter) throws IOException {
			outKey.set((int)wpair.getOtherValue());
			outValue.set(wpair.getValue());
			out.collect(outKey, outValue);
		//	System.out.println("map output: "+outKey+" -- "+outValue);
	}
}

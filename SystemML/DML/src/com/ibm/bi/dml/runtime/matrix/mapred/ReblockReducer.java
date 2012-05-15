package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.TaggedPartialBlock;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;



public class ReblockReducer extends ReduceBase 
implements Reducer<MatrixIndexes, TaggedPartialBlock, MatrixIndexes, MatrixBlock>{
	private HashMap<Byte, MatrixCharacteristics> dimensions=new HashMap<Byte, MatrixCharacteristics>();
	
	public void reduce(MatrixIndexes indexes, Iterator<TaggedPartialBlock> values,
			OutputCollector<MatrixIndexes, MatrixBlock> out, Reporter reporter)
			throws IOException {
		
		long start=System.currentTimeMillis();
		
		commonSetup(reporter);
		
		cachedValues.reset();
		
		//process the reducer part of the reblock operation
		processReblockInReducer(indexes, values, dimensions);
		
		//perform mixed operations
		processReducerInstructions();
		
		//output results
		outputResultsFromCachedValues(reporter);
		
		reporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}
	
	public void configure(JobConf job) {
		MRJobConfiguration.setMatrixValueClass(job, true);
		super.configure(job);
		//parse the reblock instructions 
		ReblockInstruction[] reblockInstructions;
		try {
			reblockInstructions = MRJobConfiguration.getReblockInstructions(job);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		for(ReblockInstruction ins: reblockInstructions)
			dimensions.put(ins.output, MRJobConfiguration.getMatrixCharactristicsForReblock(job, ins.output));
	}
}

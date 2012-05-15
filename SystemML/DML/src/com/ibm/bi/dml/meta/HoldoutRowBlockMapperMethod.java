package com.ibm.bi.dml.meta;

import java.io.IOException;

import org.apache.commons.math.random.Well1024a;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.Pair;


public class HoldoutRowBlockMapperMethod extends BlockMapperMethod {
	MatrixBlock block ;
	
	public HoldoutRowBlockMapperMethod(PartitionParams pp,
			MultipleOutputs multipleOutputs) {
		super(pp, multipleOutputs);
	}

	@Override
	void execute(Well1024a currRandom, Pair<MatrixIndexes, MatrixBlock> pair,
			Reporter reporter, OutputCollector out) throws IOException {
		IntWritable obj = new IntWritable() ;		
		int numtimes = (pp.toReplicate == true) ? pp.numIterations : 1;
		for(int i = 0; i < numtimes; i++) {
			double value = currRandom.nextDouble();
			block = pair.getValue() ;
			if(value < pp.frac) {	//send to test; ignore for el
				if(pp.isEL == true)
					continue;
				obj.set(2*i);
			}
			else	//train set
				obj.set((pp.isEL == true) ? i : (2*i + 1));
			out.collect(obj, block) ;
		}
	}
}

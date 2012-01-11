package dml.meta;
import java.io.IOException;
import org.apache.hadoop.io.LongWritable ;

import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import umontreal.iro.lecuyer.rng.WELL1024;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;

public class KFoldReducerMethod extends ReducerMethod {
	public KFoldReducerMethod(PartitionParams pp, MultipleOutputs multipleOutputs) {
		super(pp, multipleOutputs);
	}

	//@Override
	void execute(WELL1024 currRandom, LongWritable pair, MatrixBlock block, Reporter reporter) throws IOException {
		
		if (pp.toReplicate == false) {
			int partId = currRandom.nextInt(0,pp.numFolds-1) ;
			mi.setIndexes(reporter.getCounter("counter", ""+partId).getCounter(), 1) ;	//systemml matrixblks start from (1,1)
			multipleOutputs.getCollector(""+partId, reporter).collect(mi, block) ;
			reporter.incrCounter("counter", ""+partId, 1) ; 
		}
		else {
			int partId = currRandom.nextInt(0,pp.numFolds-1) ;
			mi.setIndexes(reporter.getCounter("counter", ""+2*partId).getCounter(), 1) ;
			multipleOutputs.getCollector(""+2*partId, reporter).collect(mi, block) ;
			reporter.incrCounter("counter", ""+2*partId, 1) ; 
			
			for(int i = 0 ; i < pp.numFolds; i++) {
				if(i != partId) {
					int val = 2*i+1 ;
					mi.setIndexes(reporter.getCounter("counter", ""+val).getCounter(), 1) ;
					multipleOutputs.getCollector(""+val,reporter).collect(mi, block) ;
					reporter.incrCounter("counter", ""+val, 1) ;
				}
			}
		}
	}
}

package dml.meta;
//<Arun>
import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import umontreal.iro.lecuyer.rng.WELL1024;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;

public class HoldoutKFoldBlockJoinMapperMethodIDTable extends BlockJoinMapperMethodIDTable {
	
	public HoldoutKFoldBlockJoinMapperMethodIDTable(PartitionParams pp,
			MultipleOutputs multipleOutputs) {
		super(pp, multipleOutputs);
	}

	@Override
	void execute(LongWritable key, WritableLongArray value, Reporter reporter, OutputCollector out)	throws IOException {
		//now, we read in the future row id tuple and output the key/value pairs!
		//System.out.println("$$$$$$$$ Holdout/kfold blkjoinmapper read in idtable tuple key:" + key.get() + 
		//			", value: " + value.toString()+" $$$$$$$$$$$$");
		for(int i=0; i<value.length; i++) {
			BlockJoinMapOutputValue outval = new BlockJoinMapOutputValue();
			outval.val1 = value.array[i];
			outval.val2 = -1*i - 1;		//fold num indicator, -ve means holdout/kfold and -1 is done to avoid 0
			out.collect(key, outval);	//so mk keyval pairs are snet out, rather than just m before!
		}		
		//BlockJoinMapOutputValue outval = new BlockJoinMapOutputValue();
		//outval.futrowids = value;
		//out.collect(key, outval);	//the reducer reconstructs the row and writes outs to folds using this
	}
}
//</Arun>
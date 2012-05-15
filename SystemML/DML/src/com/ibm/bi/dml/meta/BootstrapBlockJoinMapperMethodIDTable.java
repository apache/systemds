package com.ibm.bi.dml.meta;
//<Arun>
import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

public class BootstrapBlockJoinMapperMethodIDTable extends BlockJoinMapperMethodIDTable {
	
	public BootstrapBlockJoinMapperMethodIDTable(PartitionParams pp,
			MultipleOutputs multipleOutputs) {
		super(pp, multipleOutputs);
	}

	@Override
	void execute(LongWritable key, WritableLongArray value, Reporter reporter, OutputCollector out)	throws IOException {
		//now, we read in the sample rowids and swap with id, and send out output key/val pairs!
		for(int i=0; i<value.length; i++) {
			long samplerowid = value.array[i];
			LongWritable outkey = new LongWritable(samplerowid);
			BlockJoinMapOutputValue outval  = new BlockJoinMapOutputValue();
			outval.val1 = key.get();
			outval.val2 = i + 1;	//foldnum for btstrp is +ve and +1 to avoid 0
			//outval.foldnum = i;
			//outval.futrowid = key.get();
			out.collect(outkey, outval);	//the reducer reconstructs the row and writes outs to folds using these
		}
	}
}
//</Arun>
/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;
//<Arun>
import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

public class BootstrapBlockJoinMapperMethodIDTable extends BlockJoinMapperMethodIDTable 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
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
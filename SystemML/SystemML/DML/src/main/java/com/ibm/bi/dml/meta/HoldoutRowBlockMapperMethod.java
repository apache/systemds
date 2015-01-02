/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;

import java.io.IOException;

import org.apache.commons.math3.random.Well1024a;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.Pair;


public class HoldoutRowBlockMapperMethod extends BlockMapperMethod 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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

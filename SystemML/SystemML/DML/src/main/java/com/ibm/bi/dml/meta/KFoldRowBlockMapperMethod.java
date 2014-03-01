/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

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


public class KFoldRowBlockMapperMethod extends BlockMapperMethod 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	MatrixBlock block ;
	IntWritable obj = new IntWritable() ;
	public KFoldRowBlockMapperMethod(PartitionParams pp,
			MultipleOutputs multipleOutputs) {
		super(pp, multipleOutputs);
	}
	
	@Override
	void execute(Well1024a currRandom, Pair<MatrixIndexes, MatrixBlock> pair,
			Reporter reporter, OutputCollector out) throws IOException {
		
		if (pp.toReplicate == false){
			block = pair.getValue() ;
			int partId = currRandom.nextInt(pp.numFolds) ;
			obj.set(partId) ;
			out.collect(obj, block) ;
		}
		
		else {
			block = pair.getValue() ;
			int partId = currRandom.nextInt(pp.numFolds) ;
			obj.set(2*partId) ;
			out.collect(obj, block) ;
			
			for(int i = 0 ; i < pp.numFolds; i++) {
				if(i != partId) {
					obj.set(2*i + 1) ;
					out.collect(obj, block) ;
				}
			}
		}
	}
}

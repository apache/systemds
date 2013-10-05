/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixPackedCell;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixValue;


public class GMRCombiner extends ReduceBase
implements Reducer<MatrixIndexes, TaggedMatrixValue, MatrixIndexes, TaggedMatrixValue>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//temporary variable
	private TaggedMatrixValue taggedbuffer=null;
	
	public void reduce(MatrixIndexes indexes, Iterator<TaggedMatrixValue> values,
			OutputCollector<MatrixIndexes, TaggedMatrixValue> out, Reporter report) 
	throws IOException 
	{
		long start=System.currentTimeMillis();
		
		cachedValues.reset();
		
		processAggregateInstructions(indexes, values, true);
		
		//output the matrices needed by the reducer
		outputInCombinerFromCachedValues(indexes, taggedbuffer, out);
		
		report.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}

	public void configure(JobConf job)
	{
		super.configure(job);
		if(valueClass.equals(MatrixCell.class))
			valueClass=MatrixPackedCell.class;
		taggedbuffer=TaggedMatrixValue.createObject(valueClass);//new TaggedMatrixValue(valueClass);
	}
}


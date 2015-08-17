/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;

public class WeightedCellToSortInputConverter implements 
Converter<MatrixIndexes, WeightedCell, DoubleWritable, IntWritable>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private DoubleWritable outKey=new DoubleWritable();
	private IntWritable outValue=new IntWritable();
	private Pair<DoubleWritable, IntWritable> pair=new Pair<DoubleWritable, IntWritable>(outKey, outValue);
	private boolean hasValue=false;
	@Override
	public void convert(MatrixIndexes k1, WeightedCell v1) {
		outKey.set(v1.getValue());
		outValue.set((int)v1.getWeight());
		hasValue=true;
	}

	@Override
	public boolean hasNext() {
		return hasValue;
	}

	@Override
	public Pair<DoubleWritable, IntWritable> next() {
		if(!hasValue)
			return null;
		
		hasValue=false;
		return pair;
	}

	@Override
	public void setBlockSize(int rl, int cl) {
		//DO nothing
	}

}

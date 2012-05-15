package com.ibm.bi.dml.runtime.matrix.io;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;

public class WeightedCellToSortInputConverter implements 
Converter<MatrixIndexes, WeightedCell, DoubleWritable, IntWritable>{

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

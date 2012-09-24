package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;

/**
 * Custom writable for a pair of matrix indexes and matrix block
 * as required for binaryblock in remote data partitioning.
 * 
 */
public class PairWritableBlock implements Writable
{
	public MatrixIndexes indexes;
	public MatrixBlock block;
	
	@Override
	public void readFields(DataInput in) throws IOException 
	{
		indexes = new MatrixIndexes();
		indexes.readFields(in);
		
		block = new MatrixBlock();
		block.readFields(in);
	}

	@Override
	public void write(DataOutput out) throws IOException 
	{
		indexes.write(out);
		block.write(out);
	}
}

package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;

/**
 * Custom writable for a pair of matrix indexes and matrix cell
 * as required for binarycell in remote data partitioning.
 * 
 */
public class PairWritableCell implements Writable
{
	public MatrixIndexes indexes;
	public MatrixCell cell;
	
	@Override
	public void readFields(DataInput in) throws IOException 
	{
		indexes = new MatrixIndexes();
		indexes.readFields(in);
		
		cell = new MatrixCell();
		cell.readFields(in);
	}

	@Override
	public void write(DataOutput out) throws IOException 
	{
		indexes.write(out);
		cell.write(out);
	}
}

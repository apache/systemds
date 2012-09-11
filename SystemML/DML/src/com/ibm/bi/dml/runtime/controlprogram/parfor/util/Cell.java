package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

/**
 * Helper class for representing text cell and binary cell records in order to
 * allow for buffering and buffered read/write.
 * 
 * NOTE: could be replaced by IJV.class but used in order to ensure independence.
 */
public class Cell 
{
	public long row;
	public long col;
	public double value;
	
	public Cell( long r, long c, double v )
	{
		row = r;
		col = c;
		value = v;
	}
}

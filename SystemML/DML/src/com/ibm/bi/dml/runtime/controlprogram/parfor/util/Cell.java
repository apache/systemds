package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

/**
 * Helper class for representing text cell and binary cell records in order to
 * allow for buffering and buffered read/write.
 * 
 * NOTE: could be replaced by IJV.class but used in order to ensure independence.
 */
public class Cell 
{
	private long _row;
	private long _col;
	private double _value;
	
	public Cell( long r, long c, double v )
	{
		_row = r;
		_col = c;
		_value = v;
	}
	
	public long getRow()
	{
		return _row;
	}
	
	public long getCol()
	{
		return _col;
	}
	
	public double getValue()
	{
		return _value;
	}
	
	public void setRow( long row )
	{
		_row = row;
	}
	
	public void setCol( long col )
	{
		_col = col;
	}
	
	public void setValue( double value )
	{
		_value = value;
	}
}

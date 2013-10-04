/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

/**
 * Helper class for representing text cell and binary cell records in order to
 * allow for buffering and buffered read/write.
 * 
 * NOTE: could be replaced by IJV.class but used in order to ensure independence.
 */
public class Cell 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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

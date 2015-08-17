/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.sort;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.WritableComparable;

@SuppressWarnings("rawtypes")
public class IndexSortComparable implements WritableComparable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected DoubleWritable _dval = null;
	protected LongWritable _lval = null; 
	
	public IndexSortComparable()
	{
		_dval = new DoubleWritable();
		_lval = new LongWritable();
	}
	
	public void set(double dval, long lval)
	{
		_dval.set(dval);
		_lval.set(lval);
	}

	@Override
	public void readFields(DataInput arg0)
		throws IOException 
	{
		_dval.readFields(arg0);
		_lval.readFields(arg0);
	}

	@Override
	public void write(DataOutput arg0) 
		throws IOException 
	{
		_dval.write(arg0);
		_lval.write(arg0);
	}

	@Override
	public int compareTo(Object o) 
	{
		//compare only double value (e.g., for partitioner)
		if( o instanceof DoubleWritable ) {
			return _dval.compareTo((DoubleWritable) o);
		}
		//compare double value and index (e.g., for stable sort)
		else if( o instanceof IndexSortComparable) {
			IndexSortComparable that = (IndexSortComparable)o;
			int tmp = _dval.compareTo(that._dval);
			if( tmp==0 ) //secondary sort
				tmp = _lval.compareTo(that._lval);
			return tmp;
		}	
		else {
			throw new RuntimeException("Unsupported comparison involving class: "+o.getClass().getName());
		}
	}
}

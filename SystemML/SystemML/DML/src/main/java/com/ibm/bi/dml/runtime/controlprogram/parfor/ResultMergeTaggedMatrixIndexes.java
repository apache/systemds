/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

/**
 * This class serves as composite key for the remote result merge job
 * (for any data format) in order to sort on both matrix indexes and tag
 * but group all blocks according to matrix indexes only. This prevents
 * us from doing an 2pass out-of-core algorithm at the reducer since we
 * can guarantee that the compare block (tag 0) will be the first element
 * in the iterator.
 * 
 */
public class ResultMergeTaggedMatrixIndexes implements WritableComparable<ResultMergeTaggedMatrixIndexes>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixIndexes _ix;
	private byte _tag = -1;
	
	public ResultMergeTaggedMatrixIndexes()
	{
		_ix = new MatrixIndexes();
	}
	
	public ResultMergeTaggedMatrixIndexes(long r, long c, byte tag)
	{
		_ix = new MatrixIndexes(r, c);
		_tag = tag;
	}
	
	public MatrixIndexes getIndexes()
	{
		return _ix;
	}
	
	
	public byte getTag()
	{
		return _tag;
	}
	
	public void setTag(byte tag)
	{
		_tag = tag;
	}

	@Override
	public void readFields(DataInput in) 
		throws IOException 
	{
		if( _ix == null )
			_ix = new MatrixIndexes();
		_ix.readFields(in);
		_tag = in.readByte();
	}

	@Override
	public void write(DataOutput out) 
		throws IOException 
	{
		_ix.write(out);
		out.writeByte(_tag);
	}

	@Override
	public int compareTo(ResultMergeTaggedMatrixIndexes that) 
	{
		int ret = _ix.compareTo(that._ix);
		
		if( ret == 0 )
		{
			ret = ((_tag == that._tag) ? 0 : 
				   (_tag < that._tag)? -1 : 1);
		}
		
		return ret;
	}
}

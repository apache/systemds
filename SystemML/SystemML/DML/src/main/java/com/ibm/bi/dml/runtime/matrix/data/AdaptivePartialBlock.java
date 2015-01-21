/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

/**
 * Writable used in ReblockMR for intermediate results. Essentially, it's a wrapper around
 * binarycell and binaryblock in order to adaptive choose the intermediate format based
 * on data characteristics.
 * 
 */
public class AdaptivePartialBlock implements WritableComparable<AdaptivePartialBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private boolean _blocked = false; 
	private PartialBlock _cell = null;
	private MatrixBlock _block = null;
	
	//constructors
	public AdaptivePartialBlock()
	{
		
	}
	
	public AdaptivePartialBlock(MatrixBlock block)
	{
		set(block);
	}
	
	public AdaptivePartialBlock(PartialBlock cell)
	{
		set(cell);
	}
	
	//getter and setter
	public void set(MatrixBlock mb)
	{
		_blocked = true;
		_block = mb;
		//_cell = null;
	}
	
	public void set(PartialBlock pb)
	{
		_blocked = false;
		//_block = null;
		_cell = pb;
	}

	public boolean isBlocked()
	{
		return _blocked;
	}
	
	public MatrixBlock getMatrixBlock()
	{
		return _block;
	}
	
	public PartialBlock getPartialBlock()
	{
		return _cell;
	}
	
	@Override
	public void readFields(DataInput in) 
		throws IOException 
	{
		_blocked = in.readBoolean();
		if( _blocked )
		{
			if( _block==null )
				_block = new MatrixBlock();
			_block.readFields(in);
		}
		else
		{
			if( _cell == null )
				_cell = new PartialBlock();
			_cell.readFields(in);
		}
	}

	@Override
	public void write(DataOutput out) 
		throws IOException 
	{
		out.writeBoolean(_blocked);
		if( _blocked )
			_block.write(out);
		else
			_cell.write(out);
	}

	@Override
	public int compareTo(AdaptivePartialBlock that) 
	{
		if(_blocked != that._blocked)
			return -1;
		
		else if( _blocked )
			return _block.compareTo(that._block);
		else 
			return _cell.compareTo(that._cell);
	}
	
	@Override 
	public boolean equals(Object o) {
		if( !(o instanceof AdaptivePartialBlock) )
			return false;
		
		AdaptivePartialBlock that = (AdaptivePartialBlock)o;
		return _blocked==that._blocked && (_blocked ? _block.equals(that._block) : _cell.equals(that._cell));
	}
	
	@Override
	public int hashCode() {
		throw new RuntimeException("hashCode() should never be called on instances of this class.");
	}

	@Override
	public String toString()
	{
		String ret = "null";
		
		if( _blocked )
		{
			if( _block!=null )
				ret = _block.toString();
		}
		else
		{
			if( _cell!=null )
				ret = _cell.toString();
		}
		
		return ret;
	}
}

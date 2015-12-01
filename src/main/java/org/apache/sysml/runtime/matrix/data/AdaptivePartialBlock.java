/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
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

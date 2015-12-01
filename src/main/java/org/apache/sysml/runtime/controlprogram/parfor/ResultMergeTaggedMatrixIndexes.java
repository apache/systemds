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
	
	@Override
	public boolean equals(Object other) 
	{
		if( !(other instanceof ResultMergeTaggedMatrixIndexes) )
			return false;
		
		ResultMergeTaggedMatrixIndexes that = (ResultMergeTaggedMatrixIndexes)other;
		return (_ix.equals(that._ix) && _tag == that._tag);
	}
	
	@Override
	public int hashCode() {
		throw new RuntimeException("hashCode() should never be called on instances of this class.");
	}
}

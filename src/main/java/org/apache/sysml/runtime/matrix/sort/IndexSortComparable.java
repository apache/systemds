/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.matrix.sort;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.WritableComparable;

@SuppressWarnings("rawtypes")
public class IndexSortComparable implements WritableComparable
{
	
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

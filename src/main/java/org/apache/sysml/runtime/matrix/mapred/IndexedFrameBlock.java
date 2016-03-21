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


package org.apache.sysml.runtime.matrix.mapred;

import java.io.Serializable;

import org.apache.sysml.runtime.matrix.data.FrameIndexes;
import org.apache.sysml.runtime.matrix.data.FrameBlock;

public class IndexedFrameBlock implements Serializable
{
	private static final long serialVersionUID = 4785786078340105455L;
	private FrameIndexes _indexes = null;
	private FrameBlock   _value = null;
	
	public IndexedFrameBlock()
	{
		_indexes = new FrameIndexes();
	}
	
	public IndexedFrameBlock(Class<? extends FrameBlock> cls)
	{
		this();
		
		//create new value object for given class
		try {
			_value=cls.newInstance();
		} 
		catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	public IndexedFrameBlock(FrameIndexes ind, FrameBlock b)
	{
		this();
		
		_indexes.setIndexes(ind);
		_value = b;
	}

	public IndexedFrameBlock(IndexedFrameBlock that)
	{
		this(that._indexes, that._value); 
	}

	
	public FrameIndexes getIndexes()
	{
		return _indexes;
	}
	
	public FrameBlock getValue()
	{
		return _value;
	}
	
	public void set(FrameIndexes indexes2, FrameBlock block2) {
		_indexes.setIndexes(indexes2);
		_value = block2;
	}
	
	public String toString()
	{
		return "("+_indexes.getRowIndex()+", "+_indexes.getColumnIndex()+"): \n"+_value;
	}
}

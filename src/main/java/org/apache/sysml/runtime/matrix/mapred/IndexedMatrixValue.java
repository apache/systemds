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


package org.apache.sysml.runtime.matrix.mapred;

import java.io.Serializable;

import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;

public class IndexedMatrixValue implements Serializable
{

	private static final long serialVersionUID = 6723389820806752110L;

	private MatrixIndexes _indexes = null;
	private MatrixValue   _value = null;
	
	public IndexedMatrixValue()
	{
		_indexes = new MatrixIndexes();
	}
	
	public IndexedMatrixValue(Class<? extends MatrixValue> cls)
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
	
	public IndexedMatrixValue(MatrixIndexes ind, MatrixValue b)
	{
		this();
		
		_indexes.setIndexes(ind);
		_value = b;
	}

	public IndexedMatrixValue(IndexedMatrixValue that)
	{
		this(that._indexes, that._value); 
	}

	
	public MatrixIndexes getIndexes()
	{
		return _indexes;
	}
	
	public MatrixValue getValue()
	{
		return _value;
	}
	
	public void set(MatrixIndexes indexes2, MatrixValue block2) {
		_indexes.setIndexes(indexes2);
		_value = block2;
	}
	
	public String toString()
	{
		return "("+_indexes.getRowIndex()+", "+_indexes.getColumnIndex()+"): \n"+_value;
	}
}

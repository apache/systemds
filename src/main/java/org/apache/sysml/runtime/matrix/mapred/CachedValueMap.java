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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;


public class CachedValueMap //extends CachedMap<IndexedMatrixValue>
{

	private HashMap<Byte, ArrayList<IndexedMatrixValue>> map = null;
	
	public CachedValueMap()
	{
		map = new HashMap<Byte, ArrayList<IndexedMatrixValue>>();
	}
	
	public IndexedMatrixValue getFirst(byte tag) 
	{
		ArrayList<IndexedMatrixValue> tmp = map.get(tag);
		if( tmp != null )
			return tmp.get(0);
		else
			return null;
	}

	public IndexedMatrixValue holdPlace(byte tag, Class<? extends MatrixValue> cls) 
	{
		IndexedMatrixValue newVal = new IndexedMatrixValue(cls);				
		add(tag, newVal);
		
		return newVal;
	}

	public void set(byte tag, MatrixIndexes indexes, MatrixValue value) 
	{
		IndexedMatrixValue newVal = new IndexedMatrixValue(indexes, value);
		add(tag, newVal);
	}
	
	public void set(byte tag, MatrixIndexes indexes, MatrixValue value, boolean copy) 
	{
		if( copy )
		{
			//create value copy
			MatrixValue tmp = null;
			try {
				tmp=value.getClass().newInstance();
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			tmp.copy(value);
			
			set(tag, indexes, tmp);
		}
		else
			set(tag, indexes, value);
	}
	
	public void reset() 
	{
		map.clear();
	}

	public ArrayList<IndexedMatrixValue> get(byte tag) 
	{
		return map.get(tag);
	}

	public Set<Byte> getIndexesOfAll() 
	{
		return map.keySet();
	}

	public void remove(byte tag) 
	{
		map.remove(tag);
	}

	public void add(byte tag, IndexedMatrixValue value) 
	{	
		ArrayList<IndexedMatrixValue> list = map.get(tag);
		
		if( list == null ){
			list = new ArrayList<IndexedMatrixValue>();
			map.put(tag, list);	
		}
		
		list.add( value );
	}

	
	/*
	public IndexedMatrixValue set(byte thisMatrix, MatrixIndexes indexes, MatrixValue value) {
		if(numValid<cache.size())	
			cache.get(numValid).set(indexes, value);
		else
			cache.add(new IndexedMatrixValue(indexes, value));
		
		ArrayList<Integer> list=map.get(thisMatrix);
		if(list==null)
		{
			list=new ArrayList<Integer>(4);
			map.put(thisMatrix, list);
		}
		list.add(numValid);
		numValid++;
		return cache.get(numValid-1);
		
	}

	public IndexedMatrixValue holdPlace(byte thisMatrix, Class<? extends MatrixValue> cls)
	{
		
		if(numValid>=cache.size())	
			cache.add(new IndexedMatrixValue(cls));
		
		ArrayList<Integer> list=map.get(thisMatrix);
		if(list==null)
		{
			list=new ArrayList<Integer>(4);
			map.put(thisMatrix, list);
		}
		list.add(numValid);
		numValid++;
		return cache.get(numValid-1);
	}
	*/
}

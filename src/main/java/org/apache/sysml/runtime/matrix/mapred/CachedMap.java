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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;


public class CachedMap<T extends CachedMapElement> 
{

	
	protected HashMap<Byte, ArrayList<Integer>> map=new HashMap<Byte, ArrayList<Integer>>();
	protected ArrayList<T> cache=new ArrayList<T>();
	protected int numValid=0;
	protected ArrayList<T> returnListCache=new ArrayList<T>(4);
	
	public CachedMap()
	{}
	
	@SuppressWarnings("unchecked")
	public T add(Byte index, T value)
	{
		if(numValid<cache.size())	
			cache.get(numValid).set(value);
		else
			cache.add((T) value.duplicate());
		ArrayList<Integer> list=map.get(index);
		if(list==null)
		{
			list=new ArrayList<Integer>(4);
			map.put(index, list);
		}
		list.add(numValid);
		numValid++;
		return cache.get(numValid-1);
	}
	
	public void reset()
	{
		numValid=0;
		map.clear();
	}
	
	public void remove(byte index)
	{
		ArrayList<Integer> list=map.remove(index);
		if(list==null)
			return;
		
		for(Integer cacheIndex: list)
		{
			if(cacheIndex==numValid-1)
			{
				numValid--;
				return;
			}
			//swap the last element and the element to remove
			T lastElem=cache.get(numValid-1);
			cache.set(numValid-1, cache.get(cacheIndex));
			cache.set(cacheIndex, lastElem);
			//remap the indexes
			for(ArrayList<Integer> lst: map.values())
				for(int i=0; i<lst.size(); i++)
				{
					if(lst.get(i)==numValid-1)
					{
						lst.set(i, cacheIndex);
						break;
					}
				}
			numValid--;
		}
	}
	
	public ArrayList<T> get(byte index)
	{
		ArrayList<Integer> list=map.get(index);
		if(list==null)
			return null;
		
		returnListCache.clear();
		for(Integer i: list)
			returnListCache.add(cache.get(i));
		return returnListCache;
	}
	
	public T getFirst(byte index)
	{
		ArrayList<Integer> list=map.get(index);
		if(list!=null && !list.isEmpty())
			return cache.get(list.get(0));
		else
			return null;
	}
	
	public Set<Byte> getIndexesOfAll()
	{
		return map.keySet();
	}
	
/*	public String toString()
	{
		String str="";
		for(Entry<Byte,Integer> e: map.entrySet())
			str+=e.getKey()+" <--> "+cache.get(e.getValue())+"\n";
		return str;
	}*/
	
	public String toString()
	{
		String str="numValid: "+numValid+"\n"+map.toString()+"\n"+cache.toString();
		return str;
	}
	
}

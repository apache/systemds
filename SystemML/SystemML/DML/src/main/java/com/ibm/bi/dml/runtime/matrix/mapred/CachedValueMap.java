/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;


public class CachedValueMap //extends CachedMap<IndexedMatrixValue>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

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

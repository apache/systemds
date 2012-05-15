package com.ibm.bi.dml.runtime.matrix.io;

public class Pair <K, V> {
	private K key;
	private V value;
	
	public Pair()
	{
		key=null;
		value=null;
	}
	
	public Pair(K k, V v)
	{
		set(k, v);
	}
	
	public void setKey(K k)
	{
		key=k;
	}
	
	public void setValue(V v)
	{
		value=v;
	}
	
	public void set(K k, V v)
	{
		key=k;
		value=v;
	}
	public K getKey()
	{
		return key;
	}
	public V getValue()
	{
		return value;
	}
}

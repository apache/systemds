/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

public class Pair <K, V> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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

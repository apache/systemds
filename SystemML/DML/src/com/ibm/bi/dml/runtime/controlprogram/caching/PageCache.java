/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.caching;

import java.lang.ref.SoftReference;
import java.util.HashMap;
import java.util.LinkedList;

/**
 * 
 * 
 */
public class PageCache 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final int CLEANUP_THRESHOLD = 128;
	private static HashMap<Integer, LinkedList<SoftReference<byte[]>>> _pool;
	
	/**
	 * 
	 */
	public static void init()
	{
		_pool = new HashMap<Integer, LinkedList<SoftReference<byte[]>>>();
	}
	
	/**
	 * 
	 */
	public static void clear()
	{
		_pool = null;
	}
	
	/**
	 *
	 * @param data
	 */
	public static void putPage( byte[] data )
	{
		//cleanup if too many different size lists
		if( _pool.size()>CLEANUP_THRESHOLD )
			_pool.clear();
		
		LinkedList<SoftReference<byte[]>> list = _pool.get( data.length );
		if( list==null ){
			list = new LinkedList<SoftReference<byte[]>>();
			_pool.put(data.length, list);
		}
		list.addLast(new SoftReference<byte[]>(data));	
	}
	
	/**
	 * 
	 * @param size
	 * @return
	 */
	public static byte[] getPage( int size )
	{
		LinkedList<SoftReference<byte[]>> list = _pool.get( size );
		if( list!=null ) {
			while( list.size()>0 ){
				SoftReference<byte[]> ref = list.removeFirst();
				byte[] tmp = ref.get();
				if( tmp!=null )
					return tmp;
			}
		}
		return null;
	}
	
	/*
	// NOTE: Below is an alternative implementation that was however slightly slower.
	 
	private static HashMap<Integer, SoftReference<LinkedList<byte[]>>> _pool;
	
	static{
		_pool = new HashMap<Integer, SoftReference<LinkedList<byte[]>>>();
	}

	public static void putPage( byte[] data )
	{
		SoftReference<LinkedList<byte[]>> list = _pool.get( data.length );
		LinkedList<byte[]> llist = null;
		if( list!=null  )
			llist = list.get();
		if( llist==null ){
			llist = new LinkedList<byte[]>();
			list = new SoftReference<LinkedList<byte[]>>(llist);
			_pool.put(data.length, list);
		}
		llist.addLast(data);	
	}
	
	public static byte[] getPage( int size )
	{
		SoftReference<LinkedList<byte[]>> list = _pool.get( size );
		if( list!=null ) {
			LinkedList<byte[]> llist = list.get();
			if( llist!=null && llist.size()>0 )
				return llist.removeFirst();
		}
		return null;
	}
	
	*/
}

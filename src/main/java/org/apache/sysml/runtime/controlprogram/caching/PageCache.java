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

package org.apache.sysml.runtime.controlprogram.caching;

import java.lang.ref.SoftReference;
import java.util.HashMap;
import java.util.LinkedList;

/**
 * 
 * 
 */
public class PageCache 
{
	
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
			while( !list.isEmpty() ){
				SoftReference<byte[]> ref = list.removeFirst();
				byte[] tmp = ref.get();
				if( tmp!=null )
					return tmp;
			}
		}
		return null;
	}
}

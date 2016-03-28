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

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map.Entry;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.util.LocalFileUtils;

/**
 * 
 * 
 */
public class LazyWriteBuffer 
{
	public enum RPolicy {
		FIFO, //first-in, first-out eviction
		LRU   //least recently used eviction
	}
	
	//global size limit in bytes
	private static long _limit; 
	
	//current size in bytes
	private static long _size;  
	
	//eviction queue of <filename,buffer> pairs (implemented via linked hash map 
	//for (1) queue semantics and (2) constant time get/insert/delete operations)
	private static EvictionQueue _mQueue;
	
	static {
		//obtain the logical buffer size in bytes
		long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
		_limit = (long)(CacheableData.CACHING_BUFFER_SIZE * maxMem);
	}
	
	/**
	 * 
	 * @param fname
	 * @param mb
	 * @throws IOException
	 */
	public static void writeBlock( String fname, CacheBlock cb ) 
		throws IOException
	{	
		//obtain basic meta data of cache block
		long lSize = cb.getExactSerializedSize();  
		boolean requiresWrite = (   lSize > _limit  //global buffer limit
			|| !ByteBuffer.isValidCapacity(lSize, cb) ); //local buffer limit
	
		//handle caching/eviction if it fits in writebuffer
		if( !requiresWrite ) 
		{			
			ByteBuffer bbuff = null;
			
			//modify buffer pool
			synchronized( _mQueue )
			{
				//evict matrices to make room (by default FIFO)
				while( _size+lSize >= _limit )
				{
					//remove first entry from eviction queue
					Entry<String, ByteBuffer> entry = _mQueue.removeFirst();
					String ftmp = entry.getKey();
					ByteBuffer tmp = entry.getValue();
					
					if( tmp != null ) 
					{
						//wait for pending serialization
						tmp.checkSerialized();
						
						//evict matrix
						tmp.evictBuffer(ftmp);
						tmp.freeMemory();
						_size-=tmp.getSize();
						
						if( DMLScript.STATISTICS )
							CacheStatistics.incrementFSWrites();
					}
				}
				
				//create buffer (reserve mem), and lock
				bbuff = new ByteBuffer( lSize );
				
				//put placeholder into buffer pool 
				_mQueue.addLast(fname, bbuff);
				_size += lSize;	
			}
			
			//serialize matrix (outside synchronized critical path)
			bbuff.serializeBlock(cb); 
			
			if( DMLScript.STATISTICS )
				CacheStatistics.incrementFSBuffWrites();
		}	
		else
		{
			//write directly to local FS (bypass buffer if too large)
			LocalFileUtils.writeCacheBlockToLocal(fname, cb);
			if( DMLScript.STATISTICS )
				CacheStatistics.incrementFSWrites();
		}	
	}
	
	/**
	 * 
	 * @param fname
	 */
	public static void deleteBlock( String fname )
	{
		boolean requiresDelete = true;
		
		synchronized( _mQueue )
		{
			//remove queue entry 
			ByteBuffer ldata = _mQueue.remove(fname);
			if( ldata != null ) {
				_size -= ldata.getSize(); 
				requiresDelete = false;
				ldata.freeMemory(); //cleanup
			}
		}
		
		//delete from FS if required
		if( requiresDelete )
			LocalFileUtils.deleteFileIfExists(fname, true);
	}
	
	/**
	 * 
	 * @param fname
	 * @return
	 * @throws IOException
	 */
	public static CacheBlock readBlock( String fname, boolean matrix ) 
		throws IOException
	{
		CacheBlock cb = null;
		ByteBuffer ldata = null;
		
		//probe write buffer
		synchronized( _mQueue )
		{
			ldata = _mQueue.get(fname);
			
			//modify eviction order (accordingly to access)
			if(    CacheableData.CACHING_BUFFER_POLICY == RPolicy.LRU 
				&& ldata != null )
			{
				//reinsert entry at end of eviction queue
				_mQueue.remove( fname );
				_mQueue.addLast( fname, ldata );
			}
		}
		
		//deserialize or read from FS if required
		if( ldata != null )
		{
			cb = ldata.deserializeBlock();
			if( DMLScript.STATISTICS )
				CacheStatistics.incrementFSBuffHits();
		}
		else
		{
			cb = LocalFileUtils.readCacheBlockFromLocal(fname, matrix); 
			if( DMLScript.STATISTICS )
				CacheStatistics.incrementFSHits();
		}
		
		return cb;
	}
		
	/**
	 * 
	 */
	public static void init() {
		_mQueue = new EvictionQueue();		
		_size = 0;
		if( CacheableData.CACHING_BUFFER_PAGECACHE )
			PageCache.init();
	}
	
	/**
	 * 
	 */
	public static void cleanup() {
		if( _mQueue!=null )
			_mQueue.clear();
		if( CacheableData.CACHING_BUFFER_PAGECACHE )
			PageCache.clear();
	}
	
	/**
	 * 
	 * @return
	 */
	public static long getWriteBufferSize() {
		long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
		return (long)(CacheableData.CACHING_BUFFER_SIZE * maxMem);
	}
	
	/**
	 * 
	 */
	public static void printStatus( String position )
	{
		System.out.println("WRITE BUFFER STATUS ("+position+") --");
		
		//print buffer meta data
		System.out.println("\tWB: Buffer Meta Data: " +
				     "limit="+_limit+", " +
				     "size[bytes]="+_size+", " +
				     "size[elements]="+_mQueue.size()+"/"+_mQueue.size());
		
		//print current buffer entries
		int count = _mQueue.size();
		for( Entry<String, ByteBuffer> entry : _mQueue.entrySet() )
		{
			String fname = entry.getKey();
			ByteBuffer bbuff = entry.getValue();
			
			System.out.println("\tWB: buffer element ("+count+"): "+fname+", "+bbuff.getSize()+", "+bbuff.isShallow());
			count--;
		}
	}
	
	/**
	 * Extended LinkedHashMap with convenience methods for adding and removing 
	 * last/first entries.
	 * 
	 */
	private static class EvictionQueue extends LinkedHashMap<String, ByteBuffer>
	{
		private static final long serialVersionUID = -5208333402581364859L;
		
		public void addLast( String fname, ByteBuffer bbuff ) {
			//put entry into eviction queue w/ 'addLast' semantics
			put(fname, bbuff);
		}
		
		public Entry<String, ByteBuffer> removeFirst() 
		{
			//move iterator to first entry
			Iterator<Entry<String, ByteBuffer>> iter = entrySet().iterator();
			Entry<String, ByteBuffer> entry = iter.next();
			
			//remove current iterator entry
			iter.remove();
			
			return entry;
		}
	}
}

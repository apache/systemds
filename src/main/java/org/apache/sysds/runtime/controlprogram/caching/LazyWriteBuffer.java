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

package org.apache.sysds.runtime.controlprogram.caching;

import java.io.IOException;
import java.util.Map.Entry;
import java.util.concurrent.ExecutorService;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.util.LocalFileUtils;

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
	private static CacheEvictionQueue _mQueue;
	
	//maintenance service for synchronous or asynchronous delete of evicted files
	private static CacheMaintenanceService _fClean;
	
	public static int writeBlock(String fname, CacheBlock<?> cb)
		throws IOException
	{
		//obtain basic meta data of cache block
		long lSize = getCacheBlockSize(cb);
		boolean requiresWrite = (lSize > _limit        //global buffer limit
			|| !ByteBuffer.isValidCapacity(lSize, cb)); //local buffer limit
		int numEvicted = 0;
		
		//handle caching/eviction if it fits in writebuffer
		if( !requiresWrite ) 
		{
			//create byte buffer handle (no block allocation yet)
			ByteBuffer bbuff = new ByteBuffer( lSize );
			
			//modify buffer pool
			synchronized( _mQueue )
			{
				//evict matrices to make room (by default FIFO)
				while( _size+lSize > _limit && !_mQueue.isEmpty() )
				{
					//remove first entry from eviction queue
					Entry<String, ByteBuffer> entry = _mQueue.removeFirst();
					String ftmp = entry.getKey();
					ByteBuffer tmp = entry.getValue();
					
					if( tmp != null ) {
						//wait for pending serialization
						tmp.checkSerialized();
						
						//evict matrix
						tmp.evictBuffer(ftmp);
						tmp.freeMemory();
						_size -= tmp.getSize();
						numEvicted++;
					}
				}
				
				//put placeholder into buffer pool (reserve mem)
				_mQueue.addLast(fname, bbuff);
				_size += lSize;
			}
			
			//serialize matrix (outside synchronized critical path)
			_fClean.serializeData(bbuff, cb);
			
			if( DMLScript.STATISTICS ) {
				CacheStatistics.incrementBPoolWrites();
				CacheStatistics.incrementFSWrites(numEvicted);
			}
		}
		else
		{
			//write directly to local FS (bypass buffer if too large)
			LocalFileUtils.writeCacheBlockToLocal(fname, cb);
			if( DMLScript.STATISTICS ) {
				CacheStatistics.incrementFSWrites();
			}
			numEvicted++;
		}
		
		return numEvicted;
	}

	public static void deleteBlock(String fname)
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
			_fClean.deleteFile(fname);
	}
	
	public static CacheBlock<?> readBlock(String fname, boolean matrix)
		throws IOException
	{
		CacheBlock<?> cb = null;
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

	public static void init() {
		_mQueue = new CacheEvictionQueue();
		_fClean = new CacheMaintenanceService();
		_limit = OptimizerUtils.getBufferPoolLimit();
		_size = 0;
		if( CacheableData.CACHING_BUFFER_PAGECACHE )
			PageCache.init();
	}

	public static void cleanup() {
		if( _mQueue != null )
			_mQueue.clear();
		if( _fClean != null )
			_fClean.close();
		if( CacheableData.CACHING_BUFFER_PAGECACHE )
			PageCache.clear();
	}

	public static long getWriteBufferLimit() {
		//return constant limit because InfrastructureAnalyzer.getLocalMaxMemory() is
		//dynamically adjusted in a parfor context, which wouldn't reflect the actual size
		return _limit;
	}

	public static void setWriteBufferLimit(long limit) {
		_limit = limit;
	}
	
	public static long getWriteBufferSize() {
		synchronized( _mQueue ) {
			return _size; }
	}
	
	public static long getWriteBufferFree() {
		synchronized( _mQueue ) {
			return _limit - _size; }
	}
	
	public static int getQueueSize() {
		return _mQueue.size();
	}
	
	public static long getCacheBlockSize(CacheBlock<?> cb) {
		return cb.isShallowSerialize() ?
			cb.getInMemorySize() : cb.getExactSerializedSize();
	}
	
	/**
	 * Print current status of buffer pool, including all entries.
	 * NOTE: use only for debugging or testing.
	 * 
	 * @param position the position
	 */
	public static void printStatus(String position)
	{
		System.out.println("WRITE BUFFER STATUS ("+position+") --");
		
		synchronized( _mQueue ) {
			//print buffer meta data
			System.out.println("\tWB: Buffer Meta Data: " +
				"limit="+_limit+", " +
				"size[bytes]="+_size+", " +
				"size[elements]="+_mQueue.size()+"/"+_mQueue.size());
			
			//print current buffer entries
			int count = _mQueue.size();
			for( Entry<String, ByteBuffer> entry : _mQueue.entrySet() ) {
				String fname = entry.getKey();
				ByteBuffer bbuff = entry.getValue();
				System.out.println("\tWB: buffer element ("+count+"): "
					+fname+", "+(bbuff.isShallow()?bbuff._cdata.getClass().getSimpleName():"?")
					+", "+bbuff.getSize()+", "+bbuff.isShallow());
				count--;
			}
		}
	}
	
	/**
	 * Evicts all buffer pool entries.
	 * NOTE: use only for debugging or testing.
	 * 
	 * @throws IOException if IOException occurs
	 */
	public static void forceEviction()
		throws IOException 
	{
		//evict all matrices and frames
		while( !_mQueue.isEmpty() )
		{
			//remove first entry from eviction queue
			Entry<String, ByteBuffer> entry = _mQueue.removeFirst();
			ByteBuffer tmp = entry.getValue();
			
			if( tmp != null ) {
				//wait for pending serialization
				tmp.checkSerialized();
				
				//evict matrix
				tmp.evictBuffer(entry.getKey());
				tmp.freeMemory();
			}
		}
	}
	
	public static ExecutorService getUtilThreadPool() {
		return _fClean != null ? _fClean._pool : null;
	}
}

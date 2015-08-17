/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.caching;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;

/**
 * 
 * 
 */
public class LazyWriteBuffer 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum RPolicy{
		FIFO,
		LRU
	}
	
	//global size limit in bytes
	private static long _limit; 
	
	//current size in bytes
	private static long _size;  
	
	//eviction queue of <filename,buffer> pairs (implemented via linked hash map 
	//for (1) queue semantics and (2) constant time get/insert/delete operations)
	private static EvictionQueue _mQueue;
	
	static 
	{
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
	public static void writeMatrix( String fname, MatrixBlock mb ) 
		throws IOException
	{	
		long lSize = mb.getExactSizeOnDisk(); 
		boolean requiresWrite = (   lSize > _limit  //global buffer limit
				                 || !ByteBuffer.isValidCapacity(lSize, mb) ); //local buffer limit
	
		if( !requiresWrite ) //if it fits in writebuffer
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
			bbuff.serializeMatrix(mb);
			
			if( DMLScript.STATISTICS )
				CacheStatistics.incrementFSBuffWrites();
		}	
		else
		{
			//write directly to local FS (bypass buffer if too large)
			LocalFileUtils.writeMatrixBlockToLocal(fname, mb);
			if( DMLScript.STATISTICS )
				CacheStatistics.incrementFSWrites();
		}	
	}
	
	/**
	 * 
	 * @param fname
	 */
	public static void deleteMatrix( String fname )
	{
		boolean requiresDelete = true;
		
		synchronized( _mQueue )
		{
			//remove queue entry 
			ByteBuffer ldata = _mQueue.remove(fname);
			if( ldata != null )
			{
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
	public static MatrixBlock readMatrix( String fname ) 
		throws IOException
	{
		MatrixBlock mb = null;
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
			mb = ldata.deserializeMatrix();
			if( DMLScript.STATISTICS )
				CacheStatistics.incrementFSBuffHits();
		}
		else
		{
			mb = LocalFileUtils.readMatrixBlockFromLocal(fname); //read from FS
			if( DMLScript.STATISTICS )
				CacheStatistics.incrementFSHits();
		}
		
		return mb;
	}
		
	/**
	 * 
	 */
	public static void init()
	{
		_mQueue = new EvictionQueue();		
		_size = 0;
		if( CacheableData.CACHING_BUFFER_PAGECACHE )
			PageCache.init();
	}
	
	/**
	 * 
	 */
	public static void cleanup()
	{
		if( _mQueue!=null )
			_mQueue.clear();
		if( CacheableData.CACHING_BUFFER_PAGECACHE )
			PageCache.clear();
	}
	
	/**
	 * 
	 * @return
	 */
	public static long getWriteBufferSize()
	{
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
			
			System.out.println("\tWB: buffer element ("+count+"): "+fname+", "+bbuff.getSize()+", "+bbuff.isInSparseFormat());
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
		
		public void addLast( String fname, ByteBuffer bbuff )
		{
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

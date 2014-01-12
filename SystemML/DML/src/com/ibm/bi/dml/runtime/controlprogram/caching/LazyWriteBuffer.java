/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.caching;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;

/**
 * 
 * 
 */
public class LazyWriteBuffer 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum RPolicy{
		FIFO,
		LRU
	}
	
	private static long _limit; //global limit
	private static long _size;  //current size
	private static HashMap<String, ByteBuffer> _mData;
	private static LinkedList<String> _mQueue;
	
	static 
	{
		//obtain the buffer size in bytes
		long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
		_limit = (long)(CacheableData.CACHING_BUFFER_SIZE * maxMem);
	}
	
	/**
	 * 
	 * @param fname
	 * @param mb
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static void writeMatrix( String fname, MatrixBlock mb ) 
		throws FileNotFoundException, IOException
	{
		long lSize = mb.getExactSizeOnDisk(); 
		boolean requiresWrite = ( lSize >= _limit || lSize >= Integer.MAX_VALUE );
	
		if( !requiresWrite ) //if it fits in writebuffer
		{			
			ByteBuffer bbuff = null;
			byte[] buff = null; 
			
			//modify buffer
			synchronized( _mData )
			{
				//evict matrices to make room (by default FIFO)
				while( _size+lSize >= _limit )
				{
					String ftmp = _mQueue.removeFirst();
					ByteBuffer tmp = _mData.remove(ftmp);
					if( tmp != null )
					{
						//wait for pending serialization
						tmp.checkSerialized();
						
						//evict matrix
						LocalFileUtils.writeByteArrayToLocal(ftmp, tmp.data);
						if( DMLScript.STATISTICS )
							CacheStatistics.incrementFSWrites();
						_size-=tmp.data.length;
						
						//keep page for reuse
						//if( lSize <= tmp.data.length && lSize*1.5d >= tmp.data.length && (buff==null||tmp.data.length<buff.length) ) //TODO 
						//	buff = tmp.data;
					}
				}
				
				//allocate mem (if necessary) and lock
				if( CacheableData.CACHING_BUFFER_PAGECACHE )
					buff = PageCache.getPage((int)lSize);
				if( buff==null )
					buff = new byte[(int)lSize];
				bbuff = new ByteBuffer(buff);
		
				//put placeholder into buffer
				_mData.put(fname, bbuff);
				_mQueue.addLast(fname);
				_size+=buff.length;	
			}
			
			//serialize matrix
			DataOutput dout = new CacheDataOutput(buff);
			mb.write(dout);
			bbuff.markSerialized(); //for serialization outside global lock
			
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
		
		synchronized( _mData )
		{
			//remove serialized matrix
			ByteBuffer ldata = _mData.remove(fname);
			if( ldata != null )
			{
				_size -= ldata.data.length; 
				requiresDelete = false;
				if( CacheableData.CACHING_BUFFER_PAGECACHE )
					PageCache.putPage(ldata.data);
			}
			
			//remove queue entry
			_mQueue.remove(fname);	
		}
		
		//delete from FS if required
		if( requiresDelete )
			LocalFileUtils.deleteFileIfExists(fname);
	}
	
	/**
	 * 
	 * @param fname
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static MatrixBlock readMatrix( String fname ) 
		throws FileNotFoundException, IOException
	{
		MatrixBlock mb = null;
		ByteBuffer ldata = null;
		
		//probe write buffer
		synchronized( _mData )
		{
			ldata = _mData.get(fname);
			
			//modify eviction order (accordingly to access)
			if(    CacheableData.CACHING_BUFFER_POLICY == RPolicy.LRU 
				&& ldata != null )
			{
				_mQueue.remove( fname ); //equals
				_mQueue.addLast( fname );
			}
		}
		
		//deserialize or read from FS if required
		if( ldata != null )
		{
			ByteArrayInputStream bis = new ByteArrayInputStream(ldata.data);
			DataInputStream din = new DataInputStream(bis); 
			mb = new MatrixBlock();
			mb.readFields(din);
			if( DMLScript.STATISTICS )
				CacheStatistics.incrementFSBuffWrites();
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
		_mData = new HashMap<String, ByteBuffer>();
		_mQueue = new LinkedList<String>();		
		_size = 0;
		if( CacheableData.CACHING_BUFFER_PAGECACHE )
			PageCache.init();
	}
	
	/**
	 * 
	 */
	public static void cleanup()
	{
		_mData = null;
		_mQueue = null;
		if( CacheableData.CACHING_BUFFER_PAGECACHE )
			PageCache.clear();
	}
	
}

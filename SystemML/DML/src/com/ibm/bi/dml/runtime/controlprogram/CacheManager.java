package com.ibm.bi.dml.runtime.controlprogram;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.utils.CacheOutOfMemoryException;


/**
 * A global class that manages the cache.  Refer to {@link CacheableData} for
 * more information.
 * 
 * @author Alexandre Evfimievski
 */
public class CacheManager
{
	private Set <CacheableData> inMemory;
	private Set <CacheableData> evictable;
	private final long maxCacheMemory;
	private volatile long usedCacheMemory;
	private volatile long evictableCacheMemory;
	
	public CacheManager (long maxCacheMemory_)
	{
		maxCacheMemory = maxCacheMemory_;
		inMemory  = Collections.synchronizedSet (new HashSet <CacheableData> ());
		evictable = Collections.synchronizedSet (new HashSet <CacheableData> ());
		usedCacheMemory = 0;
		evictableCacheMemory = 0;
	}
	
	/**
	 * Attempts to release enough cache memory for a data blob of given size.
	 * This may, or may not, involve evicting other data blobs.
	 * 
	 * @param numBytes : the memory size required for the data blob
	 * @throws CacheOutOfMemoryException if fails to make enough memory available
	 */
	public synchronized void releaseCacheMemory (long numBytes)
		throws CacheOutOfMemoryException
	{
		if (DMLScript.DEBUG) 
		{
			System.out.println ("CACHE: Releasing " + numBytes + " of cache memory...");
		}

		if (numBytes > getFreeMemory ())
		{
			if (numBytes > getAvailableMemory ())
				throw new CacheOutOfMemoryException ();
			CacheableData[] evictionQueue = getEvictionQueue ();
			boolean is_successful = false;
			for (int i = 0; ((! is_successful) && i < evictionQueue.length); i++)
			{
				evictionQueue [i].attemptEviction ();
				is_successful = (numBytes <= getFreeMemory ());
			}
			if (! is_successful)
				throw new CacheOutOfMemoryException ();
		}
		
		if (DMLScript.DEBUG) 
		{
			System.out.println ("CACHE: Releasing cache memory - COMPLETED.");
		}
	}
	
	/**
	 * Registers a memory size change or/and status change for a cacheable data blob, namely
	 * whether or not it resides in main memory and whether it is available for eviction.
	 * The methods {@link CacheableData#isInMemory()} and {@link CacheableData#isEvictable()}
	 * should return their correct output values.
	 * This is a low-level operation called only by this class or by {@link CacheableData}.
	 * 
	 * @param cdata : the envelope containing the data blob
	 * @param oldMemorySize : the old memory size of the data blob
	 * @param newMemorySize : the new memory size of the data blob
	 * @throws CacheOutOfMemoryException 
	 */
	public synchronized void updateMemorySize (CacheableData cdata, long oldMemorySize, long newMemorySize)
		throws CacheOutOfMemoryException 
	{
		if (cdata == null)
		{
			if (DMLScript.DEBUG) 
			{
				System.out.println ("CACHE: Updating size/status - NOTHING TO DO: cacheable object == null.");
			}
			return;
		}
		
		if (DMLScript.DEBUG) 
		{
			System.out.println ("CACHE: Updating size/status of cacheable data object...");
			System.out.println ("CACHE: Object ID: " + cdata.getUniqueCacheID () + 
					";  Status: " + cdata.getStatusAsString () + 
					";  Size: old = " + oldMemorySize + ", new = " + newMemorySize);
		}

		boolean wasInMemory = inMemory.contains (cdata);
		boolean wasEvictable = evictable.contains (cdata);
		
		//  If the blob is not in memory, register it as "not in memory"
		
		if (wasInMemory && ! cdata.isInMemory ())
		{
			inMemory.remove (cdata);
			wasInMemory = false;
			usedCacheMemory -= oldMemorySize;
		}
		
		//  Do we need to release more memory by evicting some blobs?
		
		final long memoryNeeded = 
			(wasInMemory ? newMemorySize - oldMemorySize : newMemorySize);
		final boolean isMemoryNeeded = 
			(cdata.isInMemory () && memoryNeeded > getFreeMemory ());
		
		//  Before releasing memory, make sure that
		//  the current blob is taken out of eviction queue
		
		if (wasEvictable && (! cdata.isEvictable () || ! wasInMemory || isMemoryNeeded))
		{
			evictable.remove (cdata);
			wasEvictable = false;
			evictableCacheMemory -= oldMemorySize;
		}
		
		if (isMemoryNeeded)
			releaseCacheMemory (memoryNeeded);
		
		//  Add the current blob where it belongs and update memory counts
			
		if (cdata.isInMemory ())
		{
			if (wasInMemory)
				usedCacheMemory += newMemorySize - oldMemorySize;
			else
			{
				inMemory.add (cdata);
				usedCacheMemory += newMemorySize;
			}
		}
		if (cdata.isEvictable ())
		{
			if (wasEvictable)
				evictableCacheMemory += newMemorySize - oldMemorySize;
			else
			{
				evictable.add (cdata);
				evictableCacheMemory += newMemorySize;
			}
		}
		
		if (DMLScript.DEBUG) 
		{
			System.out.println ("CACHE: Updating size/status - COMPLETED.");
		}
	}
	
	/**
	 * Registers the status of a cacheable data blob, namely: whether or not it
	 * resides in main memory, and whether it is available for eviction.
	 * The methods {@link CacheableData#isInMemory()} and {@link CacheableData#isEvictable()}
	 * should return their correct output values.
	 * This is a low-level operation called only by this class or by {@link CacheableData}.
	 * 
	 * @param cdata : the envelope containing the data blob
	 * @throws CacheOutOfMemoryException
	 */
	public synchronized void refresh (CacheableData cdata)
		throws CacheOutOfMemoryException
	{
		final long memorySize = cdata.getRegisteredBlobSize ();
		updateMemorySize (cdata, memorySize, memorySize);
	}
	
	private synchronized CacheableData[] getEvictionQueue ()
	{
		CacheableData[] evictionQueue = new CacheableData [evictable.size ()];
		int index = 0;
		for (CacheableData cdata : evictable)
			evictionQueue [index ++] = cdata;
		Arrays.sort (evictionQueue, CacheableData.getComparator ());
		return evictionQueue;
	}
	
	private synchronized long getFreeMemory ()
	{
		final long freeMemory = maxCacheMemory - usedCacheMemory;
		return (freeMemory < 0 ? 0 : freeMemory);
	}
	
	private synchronized long getAvailableMemory ()
	{
		final long availableMemory = maxCacheMemory - (usedCacheMemory - evictableCacheMemory);
		return (availableMemory < 0 ? 0 : availableMemory);
	}
}

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.utils.CacheOutOfMemoryException;


/**
 * A global class that manages the cache.  Refer to {@link CacheableData} for
 * more information.
 * 
 */
public class CacheManager
{
	private Set <CacheableData> inMemory;
	private Set <CacheableData> evictable;
	private final long maxCacheMemory;
	private long usedCacheMemory;
	private long evictableCacheMemory;
	
	public CacheManager (long maxCacheMemory_)
	{
		maxCacheMemory = maxCacheMemory_;
		inMemory  = new HashSet<CacheableData>();
		evictable = new HashSet<CacheableData>();
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
	public synchronized void releaseCacheMemory (long numBytes) //TODO should be called acquire mem
		throws CacheOutOfMemoryException
	{
		internalReleaseCacheMemory(numBytes);
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
		internalUpdateMemorySize (cdata, oldMemorySize, newMemorySize);
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
		long memorySize = cdata.getRegisteredBlobSize ();
		internalUpdateMemorySize (cdata, memorySize, memorySize); 
	}
	
	/**
	 * 
	 */
	public synchronized void printCacheStatus() 
	{
		internalPrintCacheStatus();
	}
	
	
	
	///////
	//internal (unsynchronized methods)

	private void internalReleaseCacheMemory (long numBytes)
		throws CacheOutOfMemoryException
	{
		Timing time = null;
		
		if (DMLScript.DEBUG)  
		{
			//internalPrintCacheStatus();
			System.out.println ("    CACHE: Releasing " + numBytes + " of cache memory...");
			
			time = new Timing();
			time.start();
		}
	
		if (numBytes > getFreeMemory())
		{			
			if (numBytes > getAvailableMemory())
				throw new CacheOutOfMemoryException ();
			CacheableData[] evictionQueue = getEvictionQueue();
			boolean is_successful = false;
			for (int i = 0; ((! is_successful) && i < evictionQueue.length); i++)
			{
				evictionQueue [i].attemptEviction();
				is_successful = (numBytes <= getFreeMemory());
			}
			if (! is_successful)
				throw new CacheOutOfMemoryException ("requested " + numBytes + " <= " + getFreeMemory() + " available!");
		}
		
		//if (DMLScript.DEBUG) 
		{
			//printCacheStatus(); 
			System.out.println ("    CACHE: Releasing cache memory - COMPLETED.");
			//System.out.println ("    CACHE: Releasing cache memory - COMPLETED in "+time.stop()+"ms.");
		}
	}
	
	private void internalUpdateMemorySize (CacheableData cdata, long oldMemorySize, long newMemorySize)
		throws CacheOutOfMemoryException 
	{
		if (cdata == null)
		{
			if (DMLScript.DEBUG) 
			{
				System.out.println ("    CACHE: Updating envelope size/status - NOTHING TO DO: envelope == null.");
			}
			return;
		}
		
		if (DMLScript.DEBUG) 
		{
			System.out.println ("    CACHE: Updating envelope size/status of " + cdata.getDebugName() +
					String.format (", ID:%4d;  ", cdata.getUniqueCacheID ()) + 
					"Status: " + cdata.getStatusAsString () + ";  Size: " +
					(oldMemorySize == newMemorySize ? newMemorySize : 
						"old = " + oldMemorySize + ", new = " + newMemorySize));
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
		//
		//  Make sure that one blob eviction does not trigger more evictions,
		//  because that may create an infinite recursion sequence.
		
		final long memoryNeeded = 
			(wasInMemory ? newMemorySize - oldMemorySize : newMemorySize);
		final boolean isMemoryNeeded = 
			(cdata.isInMemory() && memoryNeeded > getFreeMemory() );
		
		//  Before releasing memory, make sure that
		//  the current blob is taken out of eviction queue
		
		if (wasEvictable && (! cdata.isEvictable () || ! wasInMemory || isMemoryNeeded))
		{
			evictable.remove (cdata);
			wasEvictable = false;
			evictableCacheMemory -= oldMemorySize;
		}
		
		if (isMemoryNeeded)
			internalReleaseCacheMemory(memoryNeeded);
		
		//  Add the current blob where it belongs and update memory counts
			
		if (cdata.isInMemory())
		{
			if (wasInMemory)
				usedCacheMemory += newMemorySize - oldMemorySize;
			else
			{
				inMemory.add (cdata);
				usedCacheMemory += newMemorySize;
			}
		}
		if (cdata.isEvictable())
		{
			if (wasEvictable)
				evictableCacheMemory += newMemorySize - oldMemorySize;
			else
			{
				evictable.add (cdata);
				evictableCacheMemory += newMemorySize;
			}
		}
		
		//if (DMLScript.DEBUG) 
		//{
		//	System.out.println ("CACHE: Updating envelope size/status - COMPLETED.");
		//}
	}
	
	private CacheableData[] getEvictionQueue ()
	{
		CacheableData[] evictionQueue = evictable.toArray( new CacheableData[evictable.size()] );
		Arrays.sort( evictionQueue, CacheableData.getComparator() );

		return evictionQueue;
	}
	
	private long getFreeMemory ()
	{
		final long freeMemory = maxCacheMemory - usedCacheMemory;
		return (freeMemory < 0 ? 0 : freeMemory);
	}
	
	private long getAvailableMemory ()
	{
		final long availableMemory = maxCacheMemory - (usedCacheMemory - evictableCacheMemory);
		return (availableMemory < 0 ? 0 : availableMemory);
	}

	private void internalPrintCacheStatus() 
	{
		System.out.println("\t-----------CACHE STATUS-----------" );
		System.out.println("\t      max: " + maxCacheMemory);
		System.out.println("\t     used: " + usedCacheMemory);
		System.out.println("\tevictable: " + evictableCacheMemory);
		System.out.println("\t--inMemory--");
		for(CacheableData cdata : inMemory ) {
			System.out.println("\t             " + cdata.getDebugName() + ", status: " + cdata.getStatusAsString() + ", size: " + cdata.getBlobSize());
		}
/*		System.out.println("\t--EVICTABLE--" );
		for(CacheableData cdata : evictable ) {
			System.out.println("\t             " + cdata.getDebugName());
		}
*/		System.out.println("\t---------------------------------\n\n" );
	}
}

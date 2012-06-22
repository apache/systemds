package com.ibm.bi.dml.runtime.controlprogram;

import java.util.Calendar;
import java.util.Comparator;
import java.util.Date;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.utils.CacheAssignmentException;
import com.ibm.bi.dml.utils.CacheException;
import com.ibm.bi.dml.utils.CacheIOException;
import com.ibm.bi.dml.utils.CacheOutOfMemoryException;
import com.ibm.bi.dml.utils.CacheStatusException;


/**
 * Each object of this class is a cache envelope for some large piece of data
 * called "data blob". (I prefer "blob" to "block" to avoid ambiguity.)  For
 * example, the body of a matrix can be the data blob.  The term "data blob"
 * refers strictly to the cacheable portion of the data object, often excluding
 * metadata and auxiliary parameters, as defined in the subclasses.
 * Under the protection of the envelope, the data blob may be evicted to
 * the file system; then the subclass must set its reference to <code>null</code>
 * to allow Java garbage collection.  If other parts of the system continue
 * keep references to the data blob, its eviction will not release any memory.
 * To make the eviction meaningful, the rest of the system
 * must dispose of all references prior to giving the permission for eviction. 
 * 
 * @author Alexandre Evfimievski
 * 
 */
public abstract class CacheableData extends Data
{
	// --------- ABSTRACT LOW-LEVEL CACHE I/O OPERATIONS ----------

	/**
	 * Returns the size of the data blob that can be evicted, in bytes.
	 * This value is important only if the data blob is in memory, that is,
	 * {@link #isInMemory()} = <code>true</code>.
	 * If the data blob has been evicted, this function will not be called.
	 * 
	 * @return size in bytes
	 */
	protected abstract long getBlobSize ();
	
	/**
	 * Checks if the data blob reference points to some in-memory object.
	 * This method is called when releasing the (last) lock, i.e. changing
	 * {@link #cacheStatus} from <code>READ</code> or <code>MODIFY</code>
	 * to <code>EVICTABLE</code> or <code>EMPTY</code>, in order to choose
	 * between <code>EVICTABLE</code> and <code>EMPTY</code>.  Its purpose
	 * is to support "lazy" import: otherwise, we could disallow the return
	 * status of <code>EMPTY</code>.  Do not call this method for a blob
	 * that has been evicted.
	 *
	 * @return <code>true</code> if the blob is in main memory and the
	 * reference points to it;
	 * <code>false</code> if the blob reference is <code>null</code>.
	 */
	protected abstract boolean isBlobPresent ();
	
	/**
	 * Low-level cache I/O method that physically evicts the data blob from
	 * main memory.  Must be defined by a subclass, never called by users.
	 * 
	 * @throws CacheIOException if the eviction fails, the data blob
	 *     remains as it was at the start.
	 */
	protected abstract void evictBlobFromMemory () throws CacheIOException;
	
	/**
	 * Low-level cache I/O method that physically restores the data blob to
	 * main memory.  Must be defined by a subclass, never called by users.
	 *
	 * @throws CacheIOException if the restore fails, the data blob
	 *     remains as it was at the start.
	 * @throws CacheAssignmentException if the restored blob cannot be assigned
	 *     to this envelope.
	 */
	protected abstract void restoreBlobIntoMemory ()
	throws CacheIOException, CacheAssignmentException;

	/**
	 * Low-level cache I/O method that deletes the file containing the
	 * evicted data blob, without reading it.
	 * Must be defined by a subclass, never called by users.
	 */
	protected abstract void freeEvictedBlob ();
	
	protected CacheableData (DataType dt, ValueType vt)
	{
		super (dt, vt);
	}
	
	
	// ------------- IMPLEMENTED CACHE LOGIC METHODS --------------

	
	/**
	 * The data structure that keeps track of all cacheable data blobs.
	 * It is unique and globally defined.
	 */
	private static final CacheManager cacheManager = new CacheManager (DMLScript.maxCacheMemory);
	
	/**
	 * The global counter for the objects of this class, used to provide
	 * a unique identifier and file name to every evicted data blob.
	 */
	private volatile static int globalCount = 0;

	private final int uniqueID = globalCount ++;
	
	protected synchronized int getUniqueCacheID ()
	{
		return uniqueID;
	}
	
	/**
	 * The size of the data blob as registered with {@link #cacheManager}.
	 * May not always be equal to the true size.  Measured in bytes.
	 * Must be zero when {@link #cacheStatus} = <code>EMPTY</code>.
	 */
	private volatile long registeredBlobSize = 0;
	private volatile long lastAccessTime = 0;
	
	/**
	 * The cache status of the data blob (whether it can be or is evicted, etc.)
	 */
	private CacheStatus cacheStatus = new CacheStatus ();
	
	public String getStatusAsString ()
	{
		return cacheStatus.type.toString();
	}
	
	
	/**
	 * Checks if the status is "empty" ({@link #cacheStatus} == <code>EMPTY</code>),
	 * which should imply that the data blob is <code>null</code> and no lock has 
	 * been acquired.  Keep in mind that the data blob reference may be equal to
	 * <code>null</code> for a non-empty blob, namely because it has been evicted.
	 * 
	 * @return <code>true</code> if the data blob does not exist,
	 *         <code>false</code> if the data blob exists.
	 */
	public synchronized boolean isEmpty ()
	{
		return cacheStatus.isEmpty ();
	}
	
	/**
	 * Checks if the data blob is available to be read, that is, not locked
	 * by an exclusive lock.
	 * 
	 * @return <code>true</code> if the data blob is available to be read,
	 *         <code>false</code> if exclusive-locked (in "MODIFY" status).
	 */
	public synchronized boolean isAvailableToRead ()
	{
		return cacheStatus.isAvailableToRead ();
	}

	/**
	 * Checks if the data blob is available to be modified, that is, not locked
	 * by any lock (shared or exclusive).
	 * 
	 * @return <code>true</code> if the data blob is available to be modified,
	 *         <code>false</code> if locked (in "READ" or "MODIFY" status).
	 */
	public synchronized boolean isAvailableToModify ()
	{
		return cacheStatus.isAvailableToModify ();
	}

	/**
	 * Checks if the data blob is being modified, that is, exclusively locked
	 * for writing by some thread.
	 * 
	 * @return
	 */
	public synchronized boolean isModify ()
	{
		return cacheStatus.isModify ();
	}
	
	/**
	 * Checks if the data blob resides in the cache main memory.
	 * 
	 * @return <code>false</code> if the data blob is evicted or deleted or
	 *         not yet created; <code>true</code> otherwise.
	 */
	public synchronized boolean isInMemory ()
	{
		return cacheStatus.isInMemory ();
	}
	
	/**
	 * Checks if the data blob is in main memory and available for eviction,
	 * that is, has the permission to be evicted.
	 * 
	 * @return
	 */
	public synchronized boolean isEvictable ()
	{
		return cacheStatus.isEvictable ();
	}
	
	/**
	 * Checks if the data blob has been evicted.
	 * 
	 * @return
	 */
	public synchronized boolean isEvicted ()
	{
		return cacheStatus.isEvicted ();
	}
	
	/**
	 * This method "acquires the lock" to ensure that the data blob is in main memory
	 * (not evicted) while it is being accessed.  When called, the method will try to
	 * restore the blob if it has been evicted.  There are two kinds of locks it may
	 * acquire: a shared "read" lock (if the argument is <code>false</code>) or the 
	 * exclusive "modify" lock (if the argument is <code>true</code>).
	 * The method can fail in three ways:
	 * (1) if there is lock status conflict;
	 * (2) if there is not enough cache memory to restore the blob;
	 * (3) if the restore method returns an error.
	 * The method locks the data blob in memory (which disables eviction) and updates
	 * its last-access timestamp.  For the shared "read" lock, acquiring a new lock
	 * increments the associated count.  The "read" count has to be decremented once
	 * the blob is no longer used, which may re-enable eviction.  This method has to
	 * be called only once per matrix operation and coupled with {@link #release()}, 
	 * because it increments the lock count and the other method decrements this count.
	 * 
	 * @param isModify : <code>true</code> for the exclusive "modify" lock,
	 *     <code>false</code> for a shared "read" lock.
	 * @throws CacheException
	 */
	protected synchronized void acquire (boolean isModify) throws CacheException
	{
		CacheStatusType oldStatus = cacheStatus.getType ();
		switch (oldStatus)
		{
		case EVICTED:
			cacheManager.releaseCacheMemory (registeredBlobSize);
			restoreBlobIntoMemory ();
		case EMPTY:
		case EVICTABLE:
			if (isModify)
				cacheStatus.setModify ();
			else
				cacheStatus.addOneRead ();
			break;
		case READ:
			if (isModify)
				throw new CacheStatusException ();
			else
				cacheStatus.addOneRead ();
			break;
		case MODIFY:
			throw new CacheStatusException ();
		}
		lastAccessTime = Calendar.getInstance().getTimeInMillis();
		CacheStatusType newStatus = cacheStatus.getType ();
		if (newStatus != oldStatus)
		{
			if (oldStatus == CacheStatusType.EMPTY)
				updateRegisteredBlobSize ();
			else
				cacheManager.refresh (this);
		}
		if(DMLScript.DEBUG) {
			System.out.println("    CACHE: acquired lock on " + this.getDebugName() + ", status: " + this.getStatusAsString() );
			cacheManager.printCacheStatus();
		}
	}
	
	public synchronized void downgrade ()
	throws CacheStatusException, CacheOutOfMemoryException
	{
		if (cacheStatus.getType () == CacheStatusType.MODIFY)
		{
			cacheStatus.addOneRead ();
			lastAccessTime = Calendar.getInstance().getTimeInMillis();
			updateRegisteredBlobSize ();
		}
		else
			throw new CacheStatusException ();
		
		if(DMLScript.DEBUG) {
			System.out.println("    CACHE: downgraded lock on " + this.getDebugName() + ", status: " + this.getStatusAsString());
			cacheManager.printCacheStatus();
		}
	}
	
	/**
	 * Call this method to permit eviction for the stored data blob, or to
	 * decrement its "read" count if it is "read"-locked by other threads.
	 * It is expected that you eliminate all external references to the blob
	 * prior to calling this method, because otherwise eviction will
	 * duplicate the blob, but not release memory.  This method has to be
	 * called only once per process and coupled with {@link #acquire(boolean)},
	 * because it decrements the lock count and the other method increments
	 * the lock count.
	 * 
	 * @throws CacheOutOfMemoryException 
	 * @throws CacheStatusException 
	 */
	public synchronized void release ()
	throws CacheOutOfMemoryException, CacheStatusException
	{
		CacheStatusType oldStatus = cacheStatus.getType ();
		boolean isModified = false;
		switch (oldStatus)
		{
		case EMPTY:
		case EVICTABLE:
		case EVICTED:
			throw new CacheStatusException ();
		case READ:
			cacheStatus.removeOneRead (isBlobPresent ());
			break;
		case MODIFY:
			isModified = true;
			if (isBlobPresent ())
				cacheStatus.setEvictable ();
			else
				cacheStatus.setEmpty ();
		    break;
		}
		CacheStatusType newStatus = cacheStatus.getType ();
		lastAccessTime = Calendar.getInstance().getTimeInMillis();
		if (isModified)
			updateRegisteredBlobSize ();
		else if (newStatus != oldStatus)
			cacheManager.refresh (this);			
		
		if(DMLScript.DEBUG) {
			System.out.println("    CACHE: released lock on " + this.getDebugName() + ", status: " + this.getStatusAsString());
			cacheManager.printCacheStatus();
		}
	}
	
	/**
	 * Registers the deletion of the data blob.  Only a subclass can change
	 * the blob itself, here we only register the change for cache purposes.
	 * If the old blob has been evicted, we issue a command to abandon it.
	 * If the blob has a lock on it, an exception will be thrown (because maybe
	 * other processes are using it).
	 * 
	 * @throws CacheOutOfMemoryException 
	 * @throws CacheStatusException 
	 */
	protected synchronized void registerBlobDeletion ()
	throws CacheOutOfMemoryException, CacheStatusException
	{
		switch (cacheStatus.getType ())
		{
		case EMPTY:
		    break;
		case READ:
		case MODIFY:
			throw new CacheStatusException ();
		case EVICTED:
			freeEvictedBlob ();
		case EVICTABLE:
			cacheStatus.setEmpty ();
			updateRegisteredBlobSize ();
		    break;
		}
	}
	
	/**
	 * Tries to evict the data blob if it is evictable.
	 * 
	 * @return <code>true</code> if successfully evicted;
	 *         <code>false</code> otherwise.
	 * @throws CacheOutOfMemoryException 
	 */
	public synchronized boolean attemptEviction ()
	throws CacheOutOfMemoryException
	{
		if (! cacheStatus.isEvictable ())
			return false;
		try
		{
			evictBlobFromMemory ();
		}
		catch (CacheIOException e)
		{
			return false;
		}
		cacheStatus.setEvicted ();
		cacheManager.refresh (this);
		return true;
	}
	
	/**
	 * This comparator is used to order the envelopes with evictable data blobs
	 * in the order of eviction preference.  When this comparator is created, it
	 * asks for the current time, and may use that time in its comparisons.  Hence,
	 * two comparators created at different times may produce different orderings.
	 * "Note: this comparator imposes orderings that are inconsistent with equals." 
	 * 
	 * @return a comparator where "less than" means "stronger eviction preference" 
	 */
	public static Comparator <CacheableData> getComparator ()
	{
		return new Comparator <CacheableData> ()
		{
			private final Date currTime = Calendar.getInstance().getTime();
			public int compare (CacheableData cdata1, CacheableData cdata2)
			{
				synchronized (cdata1) { synchronized (cdata2)
				{
					boolean b1 = (cdata1 == null || ! cdata1.isEvictable ());
					boolean b2 = (cdata2 == null || ! cdata2.isEvictable ());
					return (b1 ? (b2 ? 0 : +1) : (b2 ? -1 : cdata1.compareTo (cdata2, currTime)));
				}}
			}
		};
	}
	
	/**
	 * Get the data blob's registered memory size, i.e. the way it is currently
	 * registered with {@link #cacheManager}, which may be different from the
	 * actual blob size.
	 * 
	 * @return the registered blob size
	 */
	public long getRegisteredBlobSize ()
	{
		return registeredBlobSize;
	}

	/**
	 * Call this method to inform the cache about memory size changes in the data blob.
	 * @throws CacheOutOfMemoryException
	 */
	protected synchronized void updateRegisteredBlobSize ()
	throws CacheOutOfMemoryException 
	{
		if (! isEvicted ())
		{	
			long oldSize = registeredBlobSize;
			long newSize = (isEmpty () ? 0 : getBlobSize ());
			registeredBlobSize = newSize;
			cacheManager.updateMemorySize (this, oldSize, newSize);
		}
		else
		{
			//  When a data blob is evicted, we cannot update its registered size,
			//  but we call the refresh-method to make sure the status is in sync.
			cacheManager.refresh (this);
		}
	}
	
	public long getLastAccessTime ()
	{
		return lastAccessTime;
	}
	
	/**
	 * Compares two {@link CacheableData} objects, <code>this</code> and the first
	 * argument, applying the eviction policy specified in the environment.  The
	 * second argument sets the current time, in order to compute how much time has
	 * passed since the last access to each object.
	 * 
	 * @param other
	 * @param currentTime
	 * @return
	 *     -1 if <code>this</code> <  the argument,
	 *      0 if <code>this</code> == the argument,
	 *     +1 if <code>this</code> >  the argument.
	 * Remember: The head of the eviction queue is the least element!
	 */
	private int compareTo (CacheableData other, Date currentTime)
	{
		if (other == null)
			throw new NullPointerException ("CacheableData.compareTo is given a null pointer as the argument");
		
		int result = 0;
		long duration1, duration2, size1, size2;
		
		synchronized (this) { synchronized (other)
		{
			duration1 = currentTime.getTime () - getLastAccessTime ();
			duration2 = currentTime.getTime () - other.getLastAccessTime ();
			size1 = registeredBlobSize;
			size2 = other.registeredBlobSize;
		}}
		
		double defaultPriority1 = (duration1 > 10 ? Math.log (duration1) : 0) + 0.5 * (size1 > 10000 ? Math.log (size1) : 0);
		double defaultPriority2 = (duration2 > 10 ? Math.log (duration2) : 0) + 0.5 * (size2 > 10000 ? Math.log (size2) : 0);
			
		switch (DMLScript.cacheEvictionPolicy)
		{
		case DEFAULT:
			if (defaultPriority1 > defaultPriority2)
				result = -1;
			else if (defaultPriority2 > defaultPriority1)
				result = 1;
			break;
		}
		return result;
	}
	
	/**
	 * When the envelope is garbage collected, it has to free its evicted
	 * data object if it has any.
	 */
	@Override
	protected void finalize ()
	{
		if (cacheStatus.isEvicted ())
			freeEvictedBlob ();
	}
	
	
	
	
	//  **************************************************
	//  ***                                            ***
	//  ***  CACHE STATUS FIELD - CLASSES AND METHODS  ***
	//  ***                                            ***
	//  **************************************************
	
	
	/**
	 * Defines all possible cache status types for a data blob.
     * An object of class {@link CacheableData} can be in one of the following
     * five status types:
	 *
	 * <code>EMPTY</code>:   Either there is no data blob at all, or the data blob
	 * resides in a specified import file and has never been downloaded yet.
	 * <code>READ</code>:   The data blob is in main memory; one or more threads are
	 * referencing and reading it (shared "read-only" lock).  This status uses a
	 * counter.  Eviction is NOT allowed.
	 * <code>MODIFY</code>:   The data blob is in main memory; exactly one thread is
	 * referencing and modifying it (exclusive "write" lock).  Eviction is NOT allowed.
	 * <code>EVICTABLE</code>:   The data blob is in main memory, and nobody is using
	 * nor referencing it.  Eviction is allowed.
	 * <code>EVICTED</code>:   The data blob has been evicted.
	 */
	private enum CacheStatusType {EMPTY, READ, MODIFY, EVICTABLE, EVICTED};
	private class CacheStatus
	{
		private volatile CacheStatusType type = CacheStatusType.EMPTY;
	    private volatile int numReadThreads = 0;
	    
	    CacheStatus ()
	    {
	    	;
	    }
	    
	    public CacheStatusType getType ()
	    {
	    	return type;
	    }
	    
		public boolean isEmpty ()
		{
			final CacheStatusType theType = type;
			return (theType == CacheStatusType.EMPTY);
		}
		
		public boolean isAvailableToRead ()
		{
			final CacheStatusType theType = type;
			return (theType == CacheStatusType.EMPTY || theType == CacheStatusType.EVICTABLE 
					|| theType == CacheStatusType.EVICTED || theType == CacheStatusType.READ);
		}
		
		public boolean isAvailableToModify ()
		{
			final CacheStatusType theType = type;
			return (theType == CacheStatusType.EMPTY || theType == CacheStatusType.EVICTABLE 
					|| theType == CacheStatusType.EVICTED);
		}
		
		public boolean isModify ()
		{
			final CacheStatusType theType = type;
			return (theType == CacheStatusType.MODIFY);
		}
		
		public boolean isInMemory ()
		{
			final CacheStatusType theType = type;
			return (theType == CacheStatusType.READ || theType == CacheStatusType.MODIFY
					|| theType == CacheStatusType.EVICTABLE);
		}
		
		public boolean isEvictable ()
		{
			final CacheStatusType theType = type;
			return (theType == CacheStatusType.EVICTABLE);
		}
		
		public boolean isEvicted ()
		{
			final CacheStatusType theType = type;
			return (theType == CacheStatusType.EVICTED);
		}

		public synchronized void addOneRead ()
		{
			if (type == CacheStatusType.READ)
				numReadThreads ++;
			else
			{
				numReadThreads = 1;
				type = CacheStatusType.READ;				
			}
		}
		
		public synchronized void removeOneRead (boolean doesBlobExist)
		{
			if (type == CacheStatusType.READ)
			{
				if (numReadThreads == 1)
				{
					numReadThreads = 0;
					type = (doesBlobExist ? CacheStatusType.EVICTABLE : CacheStatusType.EMPTY);
				}
				else
					numReadThreads --;				
			}
		}
		
		public void setEmpty ()
		{
			type = CacheStatusType.EMPTY;
		}
		
		public void setModify ()
		{
			type = CacheStatusType.MODIFY;
		}
		
		public void setEvictable ()
		{
			type = CacheStatusType.EVICTABLE;
		}
		
		public void setEvicted ()
		{
			type = CacheStatusType.EVICTED;
		}
	}
}

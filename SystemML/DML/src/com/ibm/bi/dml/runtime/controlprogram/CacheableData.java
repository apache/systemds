package com.ibm.bi.dml.runtime.controlprogram;

import java.io.File;
import java.io.IOException;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.CacheAssignmentException;
import com.ibm.bi.dml.utils.CacheException;
import com.ibm.bi.dml.utils.CacheIOException;
import com.ibm.bi.dml.utils.CacheStatusException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;


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
 */
public abstract class CacheableData extends Data
{
	public static final boolean LDEBUG = false;

	//flag indicating if caching is turned on (eviction writes only happen if activeFlag is true)
	private static boolean _activeFlag = false;
		
	public enum CACHE_EVICTION_POLICY { 
		DEFAULT 
	};
	
	public enum CACHE_EVICTION_STORAGE_TYPE { 
		LOCAL, 
		HDFS 
	};
    
	public static final CACHE_EVICTION_POLICY cacheEvictionPolicy = CACHE_EVICTION_POLICY.DEFAULT;
    public static final CACHE_EVICTION_STORAGE_TYPE cacheEvictionStorageType = CACHE_EVICTION_STORAGE_TYPE.LOCAL;
	
    public static final String cacheEvictionLocalFilePath = "/tmp/systemml/cache/"+Lops.PROCESS_PREFIX+DMLScript.getUUID()+"/"; 
    public static String cacheEvictionLocalFilePrefix = "cache";
    public static final String cacheEvictionLocalFileExtension = ".dat";
    public static final String cacheEvictionHDFSFilePath = Lops.PROCESS_PREFIX+DMLScript.getUUID()+"/"; //prefix dir is set during runtime
    public static String cacheEvictionHDFSFilePrefix = "cache";
    public static final String cacheEvictionHDFSFileExtension = ".dat";
    
    private enum CacheStatusType {
    	EMPTY, 
    	READ, 
    	MODIFY, 
    	EVICTABLE, 
    	EVICTED
    };
	    
	private static IDSequence _seq = null; //TODO ensure that each JVM writes to its own cache dir, otherwise id will fail; MB: well, those ID need only be globally unique if we use HDFS cache dirs, what we dont do; separation maybe even necessary to prevent conflict on single node  
	
	static
	{
		_seq = new IDSequence();
	}
	
	/**
	 * The unique (JVM-wide) of a cacheable data object. 
	 */
	private final int _uniqueID;
	
	/**
	 * The cache status of the data blob (whether it can be or is evicted, etc.)
	 */
	private CacheStatus _cacheStatus = null;
	
	
	
	protected CacheableData (DataType dt, ValueType vt)
	{
		super (dt, vt);
		
		_uniqueID = (int)_seq.getNextID();
		_cacheStatus = new CacheStatus();
	}
	
	// --------- ABSTRACT LOW-LEVEL CACHE I/O OPERATIONS ----------

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
	 * @param mb 
	 * 
	 * @throws CacheIOException if the eviction fails, the data blob
	 *     remains as it was at the start.
	 */
	protected abstract void evictBlobFromMemory (MatrixBlock mb) 
		throws CacheIOException;
	
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
	
	/**
	 * Deletes the DML-script-specific caching working dir.
	 */
	public synchronized static void cleanupCacheDir()
	{
		//get directory name
		String dir = null;
		switch (CacheableData.cacheEvictionStorageType)
		{
			case LOCAL:
				dir = cacheEvictionLocalFilePath;
				break;
			case HDFS:
				dir = cacheEvictionHDFSFilePath;
				break;
		}
		
		//clean files with cache prefix
		try
		{
			switch (CacheableData.cacheEvictionStorageType)
			{
				case LOCAL:
					File fdir = new File(dir);
					File[] files = fdir.listFiles();
					for( File f : files )
						if( f.getName().startsWith(cacheEvictionLocalFilePrefix) )
							f.delete();
					fdir.delete();
					break;
				case HDFS:
					MapReduceTool.deleteFileIfExistOnHDFS( dir );
					//old version
					//FileSystem fs = FileSystem.get(new Configuration());
					//FileStatus[] status = fs.listStatus(new Path(dir));
					//for( FileStatus f : status )
					//	if( f.getPath().getName().startsWith(cacheEvictionHDFSFilePrefix) )
					//		fs.delete(f.getPath(), true);
					break;
			}
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
		
		_activeFlag = false;
	}
	
	
	/**
	 * Creates the DML-script-specific caching working dir.
	 */
	public synchronized static void createCacheDir()
	{
		//get directory name
		String dir = null;
		switch (CacheableData.cacheEvictionStorageType)
		{
			case LOCAL:
				dir = cacheEvictionLocalFilePath;
				break;
			case HDFS:
				try {
					dir = ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE) + 
					      Lops.FILE_SEPARATOR + cacheEvictionHDFSFilePath;
				} 
				catch (ParseException e){ 
					System.out.println("Error parsing configuration file."); 
				}
				break;
		}
		
		//create dir 
		switch (CacheableData.cacheEvictionStorageType)
		{
			case LOCAL:
				new File(dir).mkdirs();
				break;
			case HDFS:
				//do nothing (create on the fly)
				break;
		}

		_activeFlag = true; //turn on caching
	}
	
	public static synchronized boolean isCachingActive()
	{
		return _activeFlag;
	}
	
	
	// ------------- IMPLEMENTED CACHE LOGIC METHODS --------------	
	
	protected int getUniqueCacheID ()
	{
		return _uniqueID;
	}
	
	public String getStatusAsString ()
	{
		return _cacheStatus.getType().toString();
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
	public boolean isEmpty ()
	{
		return _cacheStatus.isEmpty ();
	}
	
	/**
	 * Checks if the data blob is available to be read, that is, not locked
	 * by an exclusive lock.
	 * 
	 * @return <code>true</code> if the data blob is available to be read,
	 *         <code>false</code> if exclusive-locked (in "MODIFY" status).
	 */
	public boolean isAvailableToRead ()
	{
		return _cacheStatus.isAvailableToRead ();
	}

	/**
	 * Checks if the data blob is available to be modified, that is, not locked
	 * by any lock (shared or exclusive).
	 * 
	 * @return <code>true</code> if the data blob is available to be modified,
	 *         <code>false</code> if locked (in "READ" or "MODIFY" status).
	 */
	public boolean isAvailableToModify ()
	{
		return _cacheStatus.isAvailableToModify ();
	}

	/**
	 * Checks if the data blob is being modified, that is, exclusively locked
	 * for writing by some thread.
	 * 
	 * @return
	 */
	public boolean isModify ()
	{
		return _cacheStatus.isModify ();
	}
	
	/**
	 * Checks if the data blob resides in the cache main memory.
	 * 
	 * @return <code>false</code> if the data blob is evicted or deleted or
	 *         not yet created; <code>true</code> otherwise.
	 */
	public boolean isInMemory ()
	{
		return _cacheStatus.isInMemory ();
	}
	
	/**
	 * Checks if the data blob is in main memory and available for eviction,
	 * that is, has the permission to be evicted.
	 * 
	 * @return
	 */
	public boolean isEvictable ()
	{
		return _cacheStatus.isEvictable ();
	}
	
	/**
	 * Checks if the data blob has been evicted.
	 * 
	 * @return
	 */
	public boolean isEvicted ()
	{
		return _cacheStatus.isEvicted ();
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
	protected void acquire (boolean isModify, boolean restore) 
		throws CacheException
	{
		switch ( _cacheStatus.getType() )
		{
			case EVICTED:
				if(restore)
					restoreBlobIntoMemory ();
			case EMPTY:
			case EVICTABLE:
				if (isModify)
					_cacheStatus.setModify();
				else
					_cacheStatus.addOneRead();
				break;
			case READ:
				if (isModify)
					throw new CacheStatusException ("READ-MODIFY not allowed.");
				else
					_cacheStatus.addOneRead();
				break;
			case MODIFY:
				throw new CacheStatusException ("MODIFY-MODIFY not allowed.");
		}

		if(DMLScript.DEBUG) 
			System.out.println("    CACHE: acquired lock on " + this.getDebugName() + ", status: " + this.getStatusAsString() );
		
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
	 * @throws CacheException 
	 */
	public void release()
		throws CacheException
	{
		switch ( _cacheStatus.getType() )
		{
			case EMPTY:
			case EVICTABLE:
			case EVICTED:
				throw new CacheStatusException("Redundant release.");
			case READ:
				_cacheStatus.removeOneRead( isBlobPresent() );
				break;
			case MODIFY:
				if ( isBlobPresent() )
					_cacheStatus.setEvictable();
				else
					_cacheStatus.setEmpty();
			    break;
		}
		
		if(DMLScript.DEBUG) {
			System.out.println("    CACHE: released lock on " + this.getDebugName() + ", status: " + this.getStatusAsString());
		}
	}
	
	public void setEmpty()
	{
		_cacheStatus.setEmpty();
	}
	
	/**
	 * Tries to evict the data blob if it is evictable.
	 * @param matrixBlock 
	 * 
	 * @return <code>true</code> if successfully evicted;
	 *         <code>false</code> otherwise.
	 * @throws CacheIOException 
	 */
	public synchronized boolean attemptEviction (MatrixBlock mb) 
		throws CacheException
	{
		boolean ret = false;
		
		if( isCachingActive() ) //discard eviction requests after caching already turned off
		{			
			if( isEvictable() )//proceed with eviction request
			{
				evictBlobFromMemory( mb );
				_cacheStatus.setEvicted();
				ret = true;
			}
		}
		else if ( LDEBUG )  
		{
			System.out.println("Warning: caching not active, discard eviction request.");
		}
		
		return ret;
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
	
	protected class CacheStatus
	{
		protected CacheStatusType _type = null;
		protected int _numReadThreads = 0;
	    
		protected CacheStatus ()
	    {
			_type = CacheStatusType.EMPTY;
			_numReadThreads = 0;
	    }
	    
		protected CacheStatusType getType ()
	    {
	    	return _type;
	    }
	    
		protected boolean isEmpty()
		{
			return (_type == CacheStatusType.EMPTY);
		}
		
		protected boolean isAvailableToRead ()
		{
			return (   _type == CacheStatusType.EMPTY 
					|| _type == CacheStatusType.EVICTABLE 
					|| _type == CacheStatusType.EVICTED 
					|| _type == CacheStatusType.READ);
		}
		
		protected boolean isAvailableToModify ()
		{
			return (   _type == CacheStatusType.EMPTY 
					|| _type == CacheStatusType.EVICTABLE 
					|| _type == CacheStatusType.EVICTED);
		}
		
		protected boolean isModify ()
		{
			return (_type == CacheStatusType.MODIFY);
		}
		
		protected boolean isInMemory ()
		{
			return (   _type == CacheStatusType.READ 
					|| _type == CacheStatusType.MODIFY
					|| _type == CacheStatusType.EVICTABLE);
		}
		
		protected boolean isEvictable ()
		{
			return (_type == CacheStatusType.EVICTABLE);
		}
		
		protected boolean isEvicted ()
		{
			return (_type == CacheStatusType.EVICTED);
		}

		protected void addOneRead ()
		{
			_numReadThreads ++;
			_type = CacheStatusType.READ;
		}
		
		protected void removeOneRead (boolean doesBlobExist)
		{
			_numReadThreads --;					
			if (_numReadThreads == 0)
				_type = (doesBlobExist ? CacheStatusType.EVICTABLE : CacheStatusType.EMPTY);
		}
		
		protected void setEmpty ()
		{
			_type = CacheStatusType.EMPTY;
		}
		
		protected void setModify ()
		{
			_type = CacheStatusType.MODIFY;
		}
		
		protected void setEvictable ()
		{
			_type = CacheStatusType.EVICTABLE;
		}
		
		protected void setEvicted ()
		{
			_type = CacheStatusType.EVICTED;
		}
	}

}

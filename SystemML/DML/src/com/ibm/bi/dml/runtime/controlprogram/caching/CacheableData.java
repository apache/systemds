package com.ibm.bi.dml.runtime.controlprogram.caching;

import java.io.File;
import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
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
	protected static final Log LOG = LogFactory.getLog(CacheableData.class.getName());
    
	protected static final long CACHING_THRESHOLD = 128; //obj not subject to caching if num values below threshold
	
	//flag indicating if caching is turned on (eviction writes only happen if activeFlag is true)
	private static boolean _activeFlag = false;
	
	protected enum CACHE_EVICTION_STORAGE_TYPE { 
		LOCAL, 
		HDFS 
	};
    
    public static final CACHE_EVICTION_STORAGE_TYPE cacheEvictionStorageType = CACHE_EVICTION_STORAGE_TYPE.LOCAL;
	
    public static String cacheEvictionLocalFilePath = null; //set during init
    public static String cacheEvictionLocalFilePrefix = "cache";
    public static final String cacheEvictionLocalFileExtension = ".dat";
    public static String cacheEvictionHDFSFilePath = null; //prefix dir is set during runtime
    public static String cacheEvictionHDFSFilePrefix = "cache";
    public static final String cacheEvictionHDFSFileExtension = ".dat";
    
	/**
	 * Defines all possible cache status types for a data blob.
     * An object of class {@link CacheableData} can be in one of the following
     * five status types:
	 *
	 * <code>EMPTY</code>: Either there is no data blob at all, or the data blob  
	 * resides in a specified import file and has never been downloaded yet.
	 * <code>READ</code>:   The data blob is in main memory; one or more threads are
	 * referencing and reading it (shared "read-only" lock).  This status uses a
	 * counter.  Eviction is NOT allowed.
	 * <code>MODIFY</code>:   The data blob is in main memory; exactly one thread is
	 * referencing and modifying it (exclusive "write" lock).  Eviction is NOT allowed.
	 * <code>CACHED</code>:   The data blob is in main memory, and nobody is using nor referencing it. 
	 * There is always an persistent recovery object for it
	 **/
    protected enum CacheStatus {
    	EMPTY, 
    	READ, 
    	MODIFY, 
    	CACHED
    };
	    
	private static IDSequence _seq = null;   
	
	static
	{
		_seq = new IDSequence();
	}
	
	/**
	 * The unique (JVM-wide) ID of a cacheable data object; to ensure unique IDs across JVMs, we
	 * concatenate filenames with a unique prefix (map task ID). 
	 */
	private final int _uniqueID;
	
	/**
	 * The cache status of the data blob (whether it can be or is evicted, etc.)
	 */
	private CacheStatus _cacheStatus = null;
	private int         _numReadThreads = 0;
	
	protected CacheableData (DataType dt, ValueType vt)
	{
		super (dt, vt);
		
		_uniqueID = (int)_seq.getNextID();
		
		_cacheStatus = CacheStatus.EMPTY;
		_numReadThreads = 0;
	}
	
	// --------- ABSTRACT LOW-LEVEL CACHE I/O OPERATIONS ----------

	/**
	 * Checks if the data blob reference points to some in-memory object.
	 * This method is called when releasing the (last) lock. Do not call 
	 * this method for a blob that has been evicted.
	 *
	 * @return <code>true</code> if the blob is in main memory and the
	 * reference points to it;
	 * <code>false</code> if the blob reference is <code>null</code>.
	 */
	public abstract boolean isBlobPresent();
	
	/**
	 * Low-level cache I/O method that physically evicts the data blob from
	 * main memory.  Must be defined by a subclass, never called by users.
	 * @param mb 
	 * 
	 * @throws CacheIOException if the eviction fails, the data blob
	 *     remains as it was at the start.
	 */
	protected abstract void evictBlobFromMemory(MatrixBlock mb) 
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
	protected abstract void restoreBlobIntoMemory()
		throws CacheIOException, CacheAssignmentException;

	/**
	 * Low-level cache I/O method that deletes the file containing the
	 * evicted data blob, without reading it.
	 * Must be defined by a subclass, never called by users.
	 */
	protected abstract void freeEvictedBlob();
		
	
	// ------------- IMPLEMENTED CACHE LOGIC METHODS --------------	
	
	protected int getUniqueCacheID()
	{
		return _uniqueID;
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
		switch ( _cacheStatus )
		{
			case CACHED:
				if(restore)
					restoreBlobIntoMemory();
			case EMPTY:
				if (isModify)
					setModify();
				else
					addOneRead();
				break;
			case READ:
				if (isModify)
					throw new CacheStatusException ("READ-MODIFY not allowed.");
				else
					addOneRead();
				break;
			case MODIFY:
				throw new CacheStatusException ("MODIFY-MODIFY not allowed.");
		}

		LOG.trace("Acquired lock on " + this.getDebugName() + ", status: " + this.getStatusAsString() );
		
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
	protected void release()
		throws CacheException
	{
		switch ( _cacheStatus )
		{
			case EMPTY:
			case CACHED:
				throw new CacheStatusException("Redundant release.");
			case READ:
				removeOneRead( isBlobPresent() );
				break;
			case MODIFY:
				if ( isBlobPresent() )
					setCached();
				else
					setEmpty();
			    break;
		}
		
		LOG.trace("Released lock on " + this.getDebugName() + ", status: " + this.getStatusAsString());
		
	}

	
	//  **************************************************
	//  ***                                            ***
	//  ***  CACHE STATUS FIELD - CLASSES AND METHODS  ***
	//  ***                                            ***
	//  **************************************************
	
	
	public String getStatusAsString()
	{
		return _cacheStatus.toString();
	}
    
	protected boolean isEmpty()
	{
		return (_cacheStatus == CacheStatus.EMPTY);
	}
	
	protected boolean isModify()
	{
		return (_cacheStatus == CacheStatus.MODIFY);
	}

	protected boolean isCached()
	{
		return (_cacheStatus == CacheStatus.CACHED);
	}
	
	protected void setEmpty()
	{
		_cacheStatus = CacheStatus.EMPTY;
	}
	
	protected void setModify()
	{
		_cacheStatus = CacheStatus.MODIFY;
	}
	
	protected void setCached()
	{
		_cacheStatus = CacheStatus.CACHED;
	}

	protected void addOneRead ()
	{
		_numReadThreads ++;
		_cacheStatus = CacheStatus.READ;
	}
	
	protected void removeOneRead (boolean doesBlobExist)
	{
		_numReadThreads --;					
		if (_numReadThreads == 0)
			_cacheStatus = (doesBlobExist ? CacheStatus.CACHED : CacheStatus.EMPTY);
	}
	
	protected boolean isAvailableToRead()
	{
		return (   _cacheStatus == CacheStatus.EMPTY 
				|| _cacheStatus == CacheStatus.CACHED
				|| _cacheStatus == CacheStatus.READ);
	}
	
	protected boolean isAvailableToModify()
	{
		return (   _cacheStatus == CacheStatus.EMPTY 
				|| _cacheStatus == CacheStatus.CACHED);
	}

	// --------- STATIC CACHE INIT/CLEANUP OPERATIONS ----------
	
	
	/**
	 * 
	 */
	public synchronized static void cleanupCacheDir()
	{
		cleanupCacheDir(true);
	}
	
	/**
	 * Deletes the DML-script-specific caching working dir.
	 * 
	 * @param withDir
	 */
	public synchronized static void cleanupCacheDir(boolean withDir)
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
		if( dir != null ) //if previous init cache
		{
			try
			{
				switch (CacheableData.cacheEvictionStorageType)
				{
					case LOCAL:
						File fdir = new File(dir);
						if( fdir.exists()){ //just for robustness
							//String[] fnames = fdir.list();
							//for( String fname : fnames )
							//	if( fname.startsWith(cacheEvictionLocalFilePrefix) )
							//		new File(fname).delete();
							File[] files = fdir.listFiles();
							for( File f : files )
								if( f.getName().startsWith(cacheEvictionLocalFilePrefix) )
									f.delete();
							if( withDir )
								fdir.delete(); //deletes dir only if empty
						}
						break;
					case HDFS:
						MapReduceTool.deleteFileIfExistOnHDFS( dir );
						break;
				}
			}
			catch (IOException e)
			{
				e.printStackTrace();
			}
		}
		
		_activeFlag = false;
	}
	
	/**
	 * Inits caching with the default uuid of DMLScript
	 * @throws IOException 
	 */
	public synchronized static void initCaching() 
		throws IOException
	{
		initCaching(DMLScript.getUUID());
	}
	
	/**
	 * Creates the DML-script-specific caching working dir.
	 * 
	 * Takes the UUID in order to allow for custom uuid, e.g., for remote parfor caching
	 * 
	 * @throws IOException 
	 */
	public synchronized static void initCaching( String uuid ) 
		throws IOException
	{
		//get directory name
		String dir = null;
		DMLConfig conf = ConfigurationManager.getConfig();
		
		switch (CacheableData.cacheEvictionStorageType)
		{
			case LOCAL:
			    dir = LocalFileUtils.getWorkingDir( LocalFileUtils.CATEGORY_CACHE );
				LocalFileUtils.createLocalFileIfNotExist(dir);
				cacheEvictionLocalFilePath = dir;
				
				break;
			case HDFS:
				//get directory
				dir = conf.getTextValue(DMLConfig.SCRATCH_SPACE) 
				      + Lops.FILE_SEPARATOR + Lops.PROCESS_PREFIX + uuid + Lops.FILE_SEPARATOR;
				cacheEvictionHDFSFilePath = dir;
				break;
		}
	
		_activeFlag = true; //turn on caching
	}
	
	public static synchronized boolean isCachingActive()
	{
		return _activeFlag;
	}
	
	public static synchronized void disableCaching()
	{
		_activeFlag = false;
	}

}

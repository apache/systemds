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

import java.io.File;
import java.io.IOException;
import java.lang.ref.SoftReference;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.LazyWriteBuffer.RPolicy;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.util.LocalFileUtils;


/**
 * Each object of this class is a cache envelope for some large piece of data
 * called "cache block". For example, the body of a matrix can be the cache block.  
 * The term cache block refers strictly to the cacheable portion of the data object, 
 * often excluding metadata and auxiliary parameters, as defined in the subclasses.
 * Under the protection of the envelope, the data blob may be evicted to
 * the file system; then the subclass must set its reference to <code>null</code>
 * to allow Java garbage collection. If other parts of the system continue
 * keep references to the cache block, its eviction will not release any memory.
 */
public abstract class CacheableData<T extends CacheBlock> extends Data
{
	private static final long serialVersionUID = -413810592207212835L;

	/** Global logging instance for all subclasses of CacheableData */
	protected static final Log LOG = LogFactory.getLog(CacheableData.class.getName());
    
	// global constant configuration parameters
	public static final long 	CACHING_THRESHOLD = 4*1024; //obj not s.t. caching if below threshold [in bytes]
	public static final double 	CACHING_BUFFER_SIZE = 0.15; 
	public static final RPolicy CACHING_BUFFER_POLICY = RPolicy.FIFO; 
	public static final boolean CACHING_BUFFER_PAGECACHE = false; 
	public static final boolean CACHING_WRITE_CACHE_ON_READ = false;	
	public static final String  CACHING_COUNTER_GROUP_NAME    = "SystemML Caching Counters";
	public static final String  CACHEING_EVICTION_FILEEXTENSION = ".dat";
    
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
    	CACHED,
    	CACHED_NOWRITE,
    };
	
	/** Global flag indicating if caching is enabled (controls eviction) */
	private static boolean _activeFlag = false;
	
	/** Global sequence for generating unique ids. */
	private static IDSequence _seq = null;   

	// Global eviction path and prefix (prefix used for isolation purposes)
    public static String cacheEvictionLocalFilePath = null; //set during init
    public static String cacheEvictionLocalFilePrefix = "cache";

	static {
		_seq = new IDSequence();
	}
		
	/**
	 * The unique (JVM-wide) ID of a cacheable data object; to ensure unique IDs across JVMs, we
	 * concatenate filenames with a unique prefix (map task ID). 
	 */
	private final int _uniqueID;
	
	/** The cache status of the data blob (whether it can be or is evicted, etc. */
	private CacheStatus _cacheStatus = null;
	
	/** Cache for actual data, evicted by garbage collector. */
	protected SoftReference<T> _cache = null;
	
	/** Container object that holds the actual data. */
	protected T _data = null;

	/** The name of HDFS file in which the data is backed up. */
	protected String _hdfsFileName = null; // file name and path
	
	/** 
	 * Flag that indicates whether or not hdfs file exists.It is used 
	 * for improving the performance of "rmvar" instruction. When it has 
	 * value <code>false</code>, one can skip file system existence 
	 * checks which can be expensive.
	 */
	private boolean _hdfsFileExists = false; 

	/** Information relevant to specific external file formats. */
	private FileFormatProperties _formatProps = null;
	
	/**
	 * <code>true</code> if the in-memory or evicted matrix may be different from
	 * the matrix located at {@link #_hdfsFileName}; <code>false</code> if the two
	 * matrices should be the same.
	 */
	private boolean _dirtyFlag = false;
	
	
	// additional private flags and meta data
	private int     _numReadThreads = 0;   //number of threads for read from HDFS
	private boolean _cleanupFlag = true;   //flag if obj unpinned (cleanup enabled)	
	private String  _varName = "";         //plan variable name
	private String  _cacheFileName = null; //local eviction file name
		
	/**
	 * Basic constructor for any cacheable data.
	 * 
	 * @param dt
	 * @param vt
	 */
	protected CacheableData(DataType dt, ValueType vt) {
		super (dt, vt);		
		_uniqueID = (int)_seq.getNextID();		
		_cacheStatus = CacheStatus.EMPTY;
		_numReadThreads = 0;
	}
	
	/**
	 * Copy constructor for cacheable data (of same type).
	 * 
	 * @param that
	 */
	protected CacheableData(CacheableData<T> that) {
		this( that.getDataType(), that.getValueType() );
		_cleanupFlag = that._cleanupFlag;
		_hdfsFileName = that._hdfsFileName;
		_hdfsFileExists = that._hdfsFileExists; 
		_varName = that._varName;
	}

	
	/**
	 * Enables or disables the cleanup of the associated 
	 * data object on clearData().
	 * 
	 * @param flag
	 */
	public void enableCleanup(boolean flag) {
		_cleanupFlag = flag;
	}

	/**
	 * Indicates if cleanup of the associated data object 
	 * is enabled on clearData().
	 * 
	 * @return
	 */
	public boolean isCleanupEnabled() {
		return _cleanupFlag;
	}
	
	/**
	 * 
	 * @param s
	 */
	public void setVarName(String s) {
		_varName = s;
	}
	
	/**
	 * 
	 * @return
	 */
	public String getVarName() {
		return _varName;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isHDFSFileExists() {
		return _hdfsFileExists;
	}
	
	/**
	 * 
	 * @param flag
	 */
	public void setHDFSFileExists( boolean flag )  {
		_hdfsFileExists = flag;
	}
	
	/**
	 * 
	 * @return
	 */
	public String getFileName() {
		return _hdfsFileName;
	}

	/**
	 * 
	 * @param file
	 */
	public synchronized void setFileName( String file ) {
		if( _hdfsFileName!=null && !_hdfsFileName.equals(file) )
			if( !isEmpty(true) )
				_dirtyFlag = true;
		_hdfsFileName = file;
	}
	
	/**
	 * <code>true</code> if the in-memory or evicted matrix may be different from
	 * the matrix located at {@link #_hdfsFileName}; <code>false</code> if the two
	 * matrices are supposed to be the same.
	 */
	public boolean isDirty() {
		return _dirtyFlag;
	}
	
	/**
	 * 
	 * @param flag
	 */
	public void setDirty(boolean flag) {
		_dirtyFlag = flag;
	}
	
	/**
	 * 
	 * @return
	 */
	public FileFormatProperties getFileFormatProperties() {
		return _formatProps;
	}
	
	/**
	 * 
	 * @param props
	 */
	public void setFileFormatProperties(FileFormatProperties props) {
		_formatProps = props;
	}
	
	// --------- ABSTRACT HIGH-LEVEL CACHE I/O OPERATIONS ----------

	/**
	 * 
	 * @return
	 * @throws CacheException
	 */
	public abstract T acquireRead() 
		throws CacheException;

	/**
	 * 
	 * @return
	 * @throws CacheException
	 */
	public abstract T acquireModify()
		throws CacheException;
		
	/**
	 * 
	 * @param newData
	 * @return
	 * @throws CacheException
	 */
	public abstract T acquireModify(T newData)
		throws CacheException;

	/**
	 * 
	 * @throws CacheException
	 */
	public abstract void release()
		throws CacheException;

	/**
	 * 
	 * @throws CacheException
	 */
	public abstract void clearData()
		throws CacheException;
	
	/**
	 * 
	 * @throws CacheException
	 */
	public synchronized void exportData() throws CacheException {
		exportData( -1 );
	}
	
	/**
	 * Writes, or flushes, the matrix data to HDFS.
	 * 
	 * In-Status:  EMPTY, EVICTABLE, EVICTED, READ;
	 * Out-Status: EMPTY, EVICTABLE, EVICTED, READ.
	 * 
	 * @throws CacheException 
	 */
	public synchronized void exportData( int replication ) 
		throws CacheException 
	{
		exportData(_hdfsFileName, null, replication, null);
		_hdfsFileExists = true;
	}

	/**
	 * 
	 * @param fName
	 * @param outputFormat
	 * @throws CacheException
	 */
	public synchronized void exportData(String fName, String outputFormat)
		throws CacheException
	{
		exportData(fName, outputFormat, -1, null);
	}
	
	/**
	 * 
	 * @param fName
	 * @param outputFormat
	 * @param formatProperties
	 * @throws CacheException
	 */
	public synchronized void exportData(String fName, String outputFormat, FileFormatProperties formatProperties)
		throws CacheException
	{
		exportData(fName, outputFormat, -1, formatProperties);
	}
	
	/**
	 * 
	 * @param fName
	 * @param outputFormat
	 * @param replication
	 * @param formatProperties
	 * @throws CacheException
	 */
	public abstract void exportData (String fName, String outputFormat, int replication, FileFormatProperties formatProperties)
		throws CacheException;

	
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
	protected boolean isBlobPresent() {
		return (_data != null);
	}

	/**
	 * Low-level cache I/O method that physically restores the data blob to
	 * main memory.  Must be defined by a subclass, never called by users.
	 *
	 * @throws CacheException 
	 */
	protected void restoreBlobIntoMemory() 
		throws CacheException
	{
		String cacheFilePathAndName = getCacheFilePathAndName();
		long begin = LOG.isTraceEnabled() ? System.currentTimeMillis() : 0;
		
		if( LOG.isTraceEnabled() )
			LOG.trace ("CACHE: Restoring matrix...  " + getVarName() + "  HDFS path: " + 
						(_hdfsFileName == null ? "null" : _hdfsFileName) + ", Restore from path: " + cacheFilePathAndName);
				
		if (_data != null)
			throw new CacheException (cacheFilePathAndName + " : Cannot restore on top of existing in-memory data.");

		try {
			_data = readBlobFromCache(cacheFilePathAndName);
		}
		catch (IOException e) {
			throw new CacheException (cacheFilePathAndName + " : Restore failed.", e);	
		}
		
		//check for success
	    if (_data == null)
			throw new CacheException (cacheFilePathAndName + " : Restore failed.");
	    
	    if( LOG.isTraceEnabled() )
	    	LOG.trace("Restoring matrix - COMPLETED ... " + (System.currentTimeMillis()-begin) + " msec.");
	}		
	/**
	 * 
	 * @param fname
	 * @return
	 */
	protected abstract T readBlobFromCache(String fname)
		throws IOException;
	
	/**
	 * Low-level cache I/O method that deletes the file containing the
	 * evicted data blob, without reading it.
	 * Must be defined by a subclass, never called by users.
	 */
	protected void freeEvictedBlob() {
		String cacheFilePathAndName = getCacheFilePathAndName();
		long begin = LOG.isTraceEnabled() ? System.currentTimeMillis() : 0;
		if( LOG.isTraceEnabled() )
			LOG.trace("CACHE: Freeing evicted matrix...  " + getVarName() + "  HDFS path: " + 
						(_hdfsFileName == null ? "null" : _hdfsFileName) + " Eviction path: " + cacheFilePathAndName);
		
		LazyWriteBuffer.deleteBlock(cacheFilePathAndName);
		
		if( LOG.isTraceEnabled() )
			LOG.trace("Freeing evicted matrix - COMPLETED ... " + (System.currentTimeMillis()-begin) + " msec.");		
	}
	
	/**
	 * 
	 */
	protected boolean isBelowCachingThreshold() {
		return (_data.getInMemorySize() <= CACHING_THRESHOLD);
	}
	
	/**
	 * 
	 */
	@Override //Data
	public synchronized String getDebugName() {
		int maxLength = 23;
		String debugNameEnding = (_hdfsFileName == null ? "null" : 
			(_hdfsFileName.length() < maxLength ? _hdfsFileName : "..." + 
				_hdfsFileName.substring (_hdfsFileName.length() - maxLength + 3)));
		return getVarName() + " " + debugNameEnding;
	}
	
	
	// ------------- IMPLEMENTED CACHE LOGIC METHODS --------------	
	
	protected int getUniqueCacheID() {
		return _uniqueID;
	}
	
	/**
	 * 
	 */
	protected String getCacheFilePathAndName () {
		if( _cacheFileName==null ) {
			StringBuilder sb = new StringBuilder();
			sb.append(CacheableData.cacheEvictionLocalFilePath); 
			sb.append(CacheableData.cacheEvictionLocalFilePrefix);
			sb.append(String.format ("%09d", getUniqueCacheID()));
			sb.append(CacheableData.CACHEING_EVICTION_FILEEXTENSION);			
			_cacheFileName = sb.toString();
		}
		
		return _cacheFileName;
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
			case CACHED_NOWRITE:
			case EMPTY:
				if (isModify)
					setModify();
				else
					addOneRead();
				break;
			case READ:
				if (isModify)
					throw new CacheException ("READ-MODIFY not allowed.");
				else
					addOneRead();
				break;
			case MODIFY:
				throw new CacheException ("MODIFY-MODIFY not allowed.");
		}

		if( LOG.isTraceEnabled() )
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
	protected void release(boolean cacheNoWrite)
		throws CacheException
	{
		switch ( _cacheStatus )
		{
			case EMPTY:
			case CACHED:
			case CACHED_NOWRITE:	
				throw new CacheException("Redundant release.");
			case READ:
				removeOneRead( isBlobPresent(), cacheNoWrite );
				break;
			case MODIFY:
				if ( isBlobPresent() )
					setCached();
				else
					setEmpty();
			    break;
		}
		
		if( LOG.isTraceEnabled() )
			LOG.trace("Released lock on " + this.getDebugName() + ", status: " + this.getStatusAsString());
		
	}

	
	//  **************************************************
	//  ***                                            ***
	//  ***  CACHE STATUS FIELD - CLASSES AND METHODS  ***
	//  ***                                            ***
	//  **************************************************
	
	
	public String getStatusAsString() {
		return _cacheStatus.toString();
	}
    
	public boolean isCached(boolean inclCachedNoWrite) {
		if( inclCachedNoWrite )
			return (_cacheStatus == CacheStatus.CACHED || _cacheStatus == CacheStatus.CACHED_NOWRITE);
		else
			return (_cacheStatus == CacheStatus.CACHED);
	}
	
	protected boolean isEmpty(boolean inclCachedNoWrite) {
		if( inclCachedNoWrite )
			return (_cacheStatus == CacheStatus.EMPTY || _cacheStatus == CacheStatus.CACHED_NOWRITE);
		else
			return (_cacheStatus == CacheStatus.EMPTY);
	}
	
	protected boolean isModify() {
		return (_cacheStatus == CacheStatus.MODIFY);
	}
	
	protected void setEmpty() {
		_cacheStatus = CacheStatus.EMPTY;
	}
	
	protected void setModify() {
		_cacheStatus = CacheStatus.MODIFY;
	}
	
	protected void setCached() {
		_cacheStatus = CacheStatus.CACHED;
	}

	protected void addOneRead() {
		_numReadThreads ++;
		_cacheStatus = CacheStatus.READ;
	}
	
	protected void removeOneRead(boolean doesBlobExist, boolean cacheNoWrite) {
		_numReadThreads --;					
		if (_numReadThreads == 0) {
			if( cacheNoWrite )
				_cacheStatus = (doesBlobExist ? 
						CacheStatus.CACHED_NOWRITE : CacheStatus.EMPTY);
			else
				_cacheStatus = (doesBlobExist ? 
						CacheStatus.CACHED : CacheStatus.EMPTY);
		}
	}
	
	protected boolean isAvailableToRead() {
		return (   _cacheStatus == CacheStatus.EMPTY 
				|| _cacheStatus == CacheStatus.CACHED
				|| _cacheStatus == CacheStatus.CACHED_NOWRITE
				|| _cacheStatus == CacheStatus.READ);
	}
	
	protected boolean isAvailableToModify() {
		return (   _cacheStatus == CacheStatus.EMPTY 
				|| _cacheStatus == CacheStatus.CACHED
				|| _cacheStatus == CacheStatus.CACHED_NOWRITE);
	}

	// *******************************************
	// ***                                     ***
	// ***      LOW-LEVEL PRIVATE METHODS      ***
	// ***       FOR SOFTREFERENCE CACHE       ***
	// ***                                     ***
	// *******************************************
	
	/**
	 * Creates a new cache soft reference to the currently
	 * referenced cache block.  
	 */
	protected void createCache( ) {
		_cache = new SoftReference<T>( _data );	
	}

	/**
	 * Tries to get the cache block from the cache soft reference
	 * and subsequently clears the cache soft reference if existing.
	 */
	protected void getCache() {
		if( _cache !=null ) {
			_data = _cache.get();
			clearCache();
		}
	}
	
	/** Clears the cache soft reference if existing. */
	protected void clearCache() {
		if( _cache != null ) {
			_cache.clear();
			_cache = null;
		}
	}
	
	
	// --------- STATIC CACHE INIT/CLEANUP OPERATIONS ----------
	
	/**
	 * 
	 */
	public synchronized static void cleanupCacheDir() {
		//cleanup remaining cached writes
		LazyWriteBuffer.cleanup();
		
		//delete cache dir and files
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
		String dir = cacheEvictionLocalFilePath;
		
		//clean files with cache prefix
		if( dir != null ) //if previous init cache
		{
			File fdir = new File(dir);
			if( fdir.exists()){ //just for robustness
				File[] files = fdir.listFiles();
				for( File f : files )
					if( f.getName().startsWith(cacheEvictionLocalFilePrefix) )
						f.delete();
				if( withDir )
					fdir.delete(); //deletes dir only if empty
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
		try
		{
			String dir = LocalFileUtils.getWorkingDir( LocalFileUtils.CATEGORY_CACHE );
			LocalFileUtils.createLocalFileIfNotExist(dir);
			cacheEvictionLocalFilePath = dir;
		}
		catch(DMLRuntimeException e)
		{
			throw new IOException(e);
		}
	
		//init write-ahead buffer
		LazyWriteBuffer.init();
		
		_activeFlag = true; //turn on caching
	}
	
	public static synchronized boolean isCachingActive() {
		return _activeFlag;
	}
	
	public static synchronized void disableCaching() {
		_activeFlag = false;
	}
	
	public static synchronized void enableCaching() {
		_activeFlag = true;
	}
}

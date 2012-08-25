package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.ref.SoftReference;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.CacheableData;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.MetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.CacheAssignmentException;
import com.ibm.bi.dml.utils.CacheException;
import com.ibm.bi.dml.utils.CacheIOException;
import com.ibm.bi.dml.utils.CacheStatusException;
import com.ibm.bi.dml.utils.DMLRuntimeException;


/**
 * Represents a matrix in control program. This class contains method to read
 * matrices from HDFS and convert them to a specific format/representation. It
 * is also able to write several formats/representation of matrices to HDFS.
 * 
 * TODO tuning possibility: -XX:SoftRefLRUPolicyMSPerMB=<value>  time in ms to hold an object per MB of free mem
 * 
 * IMPORTANT: Preserve one-to-one correspondence between {@link MatrixObjectNew}
 * and {@link MatrixBlock} objects, for cache purposes.  Do not change a
 * {@link MatrixBlock} object without informing its {@link MatrixObjectNew} object.
 * 
 */
public class MatrixObjectNew extends CacheableData
{
	private CacheReference _cache = null;
	private boolean _cleanupFlag = true; //indicates if obj unpinned
	private boolean _partitioned = false; //indicates if obj partitioned
	
	
	/**
	 * Container object that holds the actual data.
	 */
	private MatrixBlock _data = null;

	/**
	 * The name of HDFS file by which the data is backed up.
	 */
	private String _hdfsFileName = null; // file name and path
	
	/**
	 * <code>true</code> if the in-memory or evicted matrix may be different from
	 * the matrix located at {@link #_hdfsFileName}; <code>false</code> if the two
	 * matrices should be the same.
	 */
	private boolean _dirtyFlag = false;
	
	private String _varName = "";
	private String _cacheFileName = null;
	
	/**
	 * Object that holds the metadata associated with the matrix, which
	 * includes: 1) Matrix dimensions, if available 2) Number of non-zeros, if
	 * available 3) Block dimensions, if applicable 4) InputInfo -- subsequent
	 * operations that use this Matrix expect it to be in this format.
	 * 
	 * When the matrix is written to HDFS (local file system, as well?), one
	 * must get the OutputInfo that matches with InputInfo stored inside _mtd.
	 */
	private MetaData _metaData = null;

	/**
	 * Constructor that takes only the HDFS filename.
	 */
	public MatrixObjectNew (ValueType vt, String file)
	{
		this (vt, file, null); //HDFS file path
	}
	
	/**
	 * Constructor that takes both HDFS filename and associated metadata.
	 */
	public MatrixObjectNew (ValueType vt, String file, MetaData mtd)
	{
		super (DataType.MATRIX, vt);
		_metaData = mtd; // Metadata
		_data = null;
		_hdfsFileName = file;
		_dirtyFlag = false;
		_cleanupFlag = true;
	}

	public void setVarName(String s) 
	{
		_varName = s;
	}
	
	public String getVarName() 
	{
		return _varName;
	}
	
	@Override
	public void setMetaData (MetaData md)
	{
		_metaData = md;
	}
	
	@Override
	public MetaData getMetaData ()
	{
		return _metaData;
	}

	@Override
	public void removeMetaData ()
	{
		_metaData = null;
	}
	
	@Override
	public void updateMatrixCharacteristics (MatrixCharacteristics mc)
	{
		((MatrixDimensionsMetaData) _metaData).setMatrixCharacteristics (mc);
	}

	/**
	 * Make the matrix metadata consistent with the in-memory matrix data
	 * @throws CacheException 
	 */
	public void refreshMetaData () 
		throws CacheException
	{
		if ( _data == null || _metaData ==null ) //refresh only for existing data
			throw new CacheException("Cannot refresh meta data because there is no data or meta data. "); 
		    //we need to throw an exception, otherwise input/output format cannot be inferred
		
		MatrixCharacteristics mc = ((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics();
		mc.setDimension( _data.getNumRows(),
						 _data.getNumColumns() );
		mc.setNonZeros( _data.getNonZeros() );		
	}

	public String getFileName ()
	{
		return _hdfsFileName;
	}

	public void setFileName (String file)
	{
		if (!_hdfsFileName.equals (file))
		{
			_hdfsFileName = file;
			if (! isEmpty ())
				_dirtyFlag = true;
		}
	}

	public long getNumRows () 
		throws DMLRuntimeException
	{
		if(_metaData == null)
			throw new DMLRuntimeException("No metadata available.");
		MatrixCharacteristics mc = ((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics ();
		return mc.get_rows ();
	}

	public long getNumColumns() 
		throws DMLRuntimeException
	{
		if(_metaData == null)
			throw new DMLRuntimeException("No metadata available.");
		MatrixCharacteristics mc = ((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics ();
		return mc.get_cols ();
	}
	
	/**
	 * <code>true</code> if the in-memory or evicted matrix may be different from
	 * the matrix located at {@link #_hdfsFileName}; <code>false</code> if the two
	 * matrices are supposed to be the same.
	 */
	public boolean isDirty ()
	{
		return _dirtyFlag;
	}
	
	public String toString()
	{ 
		StringBuilder str = new StringBuilder();
		str.append("MatrixObjectNew: ");
		str.append(_hdfsFileName + ", ");
		System.out.println(_hdfsFileName);
		if ( _metaData instanceof NumItemsByEachReducerMetaData ) {
			str.append("NumItemsByEachReducerMetaData");
		} 
		else {
			MatrixFormatMetaData md = (MatrixFormatMetaData)_metaData;
			if ( md != null ) {
				MatrixCharacteristics mc = ((MatrixDimensionsMetaData)_metaData).getMatrixCharacteristics();
				str.append(mc.toString());
				
				InputInfo ii = md.getInputInfo();
				if ( ii == null )
					str.append("null");
				else {
					if ( InputInfo.inputInfoToString(ii) == null ) {
						try {
							throw new DMLRuntimeException("Unexpected input format");
						} catch (DMLRuntimeException e) {
							e.printStackTrace();
						}
					}
					str.append(InputInfo.inputInfoToString(ii));
				}
			}
			else {
				str.append("null, null");
			}
		}
		
		return str.toString();
	}
	
	
	// *********************************************
	// ***                                       ***
	// ***    HIGH-LEVEL METHODS THAT SPECIFY    ***
	// ***   THE LOCKING AND CACHING INTERFACE   ***
	// ***                                       ***
	// *********************************************
	
	
	/**
	 * Acquires a shared "read-only" lock, produces the reference to the matrix data,
	 * restores the matrix to main memory, reads from HDFS if needed.
	 * 
	 * Synchronized because there might be parallel threads (parfor local) that
	 * access the same MatrixObjectNew object (in case it was created before the loop).
	 * 
	 * In-Status:  EMPTY, EVICTABLE, EVICTED, READ;
	 * Out-Status: READ(+1).
	 * 
	 * @return the matrix data reference
	 * @throws CacheException 
	 */
	public synchronized MatrixBlock acquireRead ()
		throws CacheException
	{
		if( LDEBUG )
			System.out.println("acquire read "+_varName);
		
		if (! isAvailableToRead ())
			throw new CacheStatusException ("MatrixObject not available to read.");
		
		//get object from cache
		getCache();
		
		//read data from HDFS if required
		if( isEmpty() ) 
		{
			//check filename
			String fName = _hdfsFileName;
			if( fName == null )
				throw new CacheException("Cannot read matrix for empty filename.");
			
			try
			{
				MatrixBlock newData = readMatrixFromHDFS( fName );
				if (newData != null)
				{
					newData.setEnvelope (this);
					_data = newData;
					_dirtyFlag = false;
				}
			}
			catch (IOException e)
			{
				throw new CacheIOException (fName + " : Reading ("+_varName+") failed.", e);
			}
		}
		acquire( false, true );

		return _data;
	}
	
	/**
	 * Acquires the exclusive "write" lock for a thread that wants to change matrix
	 * cell values.  Produces the reference to the matrix data, restores the matrix
	 * to main memory, reads from HDFS if needed.
	 * 
	 * In-Status:  EMPTY, EVICTABLE, EVICTED;
	 * Out-Status: MODIFY.
	 * 
	 * @return the matrix data reference
	 * @throws CacheException 
	 */
	public synchronized MatrixBlock acquireModify() 
		throws CacheException
	{
		if( LDEBUG )
			System.out.println("acquire modify "+_varName);

		if (! isAvailableToModify ())
			throw new CacheStatusException("MatrixObject not available to modify.");
		
		//get object from cache
		getCache();
		
		//read data from HDFS if required
		if( isEmpty() )
		{
			//check filename
			String fName = _hdfsFileName;
			if( fName == null )
				throw new CacheException("Cannot read matrix for empty filename.");
			
			//load data
			try
			{
				MatrixBlock newData = readMatrixFromHDFS (fName);
				
				//set reference to loaded data
				if (newData != null)
				{
					newData.setEnvelope (this);
					_data = newData;
				}
			}
			catch (IOException e)
			{
				throw new CacheIOException (fName + " : Reading failed.", e);
			}
		}

		//cache status maintenance
		acquire( true, true );
		_dirtyFlag = true;
		
		return _data;
	}
	
	/**
	 * Acquires the exclusive "write" lock for a thread that wants to throw away the
	 * old matrix data and link up with new matrix data.  Abandons the old matrix data
	 * without reading it.  Sets the new matrix data reference.

	 * In-Status:  EMPTY, EVICTABLE, EVICTED;
	 * Out-Status: MODIFY.
	 * 
	 * @param newData : the new matrix data reference
	 * @return the matrix data reference, which is the same as the argument
	 * @throws CacheException 
	 */
	public synchronized MatrixBlock acquireModify (MatrixBlock newData)
		throws CacheException
	{
		if( LDEBUG )
			System.out.println("acquire modify newdata "+_varName);
		
		if (! isAvailableToModify ())
			throw new CacheStatusException ("MatrixObject not available to modify.");
		
		//clear old data 
		clearData(); 
		
		//cache status maintenance
		acquire (true, false); //no need to load evicted matrix
		_dirtyFlag = true;
		
		//set references to new data
		if (newData != null)
		{
			newData.setEnvelope (this);
			_data = newData; 
		}
		else
			throw new CacheException("acquireModify with empty matrix block.");
		
		return _data;
	}

	/**
	 * Releases the shared ("read-only") or exclusive ("write") lock.  Updates
	 * the matrix size, last-access time, metadata, etc.
	 * 
	 * Synchronized because there might be parallel threads (parfor local) that
	 * access the same MatrixObjectNew object (in case it was created before the loop).
	 * 
	 * In-Status:  READ, MODIFY;
	 * Out-Status: READ(-1), EVICTABLE, EMPTY.
	 * 
	 * @throws CacheStatusException
	 */
	@Override
	public synchronized void release () 
		throws CacheException
	{
		if( LDEBUG )
			System.out.println("release "+_varName);
		
		if (isModify ())
		{
			_dirtyFlag = true;
			refreshMetaData ();
		}
		
		super.release ();
		
		if( isEvictable() )
		{
			//get object from cache
			if( _data != null )
			{
				createCache();
				_data = null;
			}
		}
	}

	/**
	 * Sets the matrix data reference to <code>null</code>, abandons the old matrix.
	 * Makes the "envelope" empty.  Run it to finalize the matrix (otherwise the
	 * evicted matrix file may remain undeleted).
	 * 
	 * In-Status:  EMPTY, EVICTABLE, EVICTED;
	 * Out-Status: EMPTY.
	 * 
	 * @throws CacheStatusException
	 */
	public void clearData ( )  
		throws CacheStatusException
	{
		clearData(false);
	}
	
	public synchronized void clearData ( boolean delFileOnHDFS ) //TODO usage in variable cp instruction
		throws CacheStatusException
	{
		if( LDEBUG )
			System.out.println("clear data "+_varName);
		
		if( !_cleanupFlag ) //if cleanup not enabled, do nothing
			return;
		
		if (! isAvailableToModify ())
			throw new CacheStatusException ("MatrixObject (" + this.getDebugName() + ") not available to modify. Status = " + this.getStatusAsString() + ".");
		
		if (_data != null) //e.g., in case of evicted matrix
		{
			_data.clearEnvelope ();
		}
			
		if( _cache!=null )
		{
			MatrixBlock tmp = _cache.get();
			if( tmp!=null )
				tmp.clearEnvelope();
			clearCache();
		}
		
		if (! isEmpty())
		{
			_data = null;
			_dirtyFlag = false;
			setEmpty();
		}
		
		//TODO delete from file; currently not necessary as we cleanup everything at the end of script execution 
	}

	/**
	 * NOTE: never used
	 * 
	 * Same as {@link #clearData()}, but in addition, sets the HDFS source file name
	 * for the matrix.  The next {@link #acquireRead()} will read it from HDFS.  So,
	 * this is a "lazy" import.
	 * 
	 * In-Status:  EMPTY, EVICTABLE, EVICTED;
	 * Out-Status: EMPTY.
	 * 
	 * @param filePath : the new HDFS path and file name
	 * @throws CacheStatusException
	 */
	/*public void importData (String filePath) 
		throws CacheStatusException
	{
		if (! isAvailableToModify ())
			throw new CacheStatusException ("MatrixObject not available to modify.");
		
		_hdfsFileName = filePath;
		clearData();
	}*/

	/**
	 * Writes, or flushes, the matrix data to HDFS.
	 * 
	 * In-Status:  EMPTY, EVICTABLE, EVICTED, READ;
	 * Out-Status: EMPTY, EVICTABLE, EVICTED, READ.
	 * 
	 * @throws CacheException 
	 */
	public void exportData ()
		throws CacheException
	{
		exportData (_hdfsFileName, null);
	}
	
	/**
	 * Synchronized because there might be parallel threads (parfor local) that
	 * access the same MatrixObjectNew object (in case it was created before the loop).
	 * If all threads export the same data object concurrently it results in errors
	 * because they all write to the same file. Efficiency for loops and parallel threads
	 * is achieved by checking if the in-memory matrix block is dirty.
	 * 
	 * 
	 * @param fName
	 * @param outputFormat
	 * @throws CacheException
	 */
	public synchronized void exportData (String fName, String outputFormat)
		throws CacheException
	{
		if( LDEBUG )
			System.out.println("export data "+_varName+" "+fName);
		
		//prevent concurrent modifications
		if (! isAvailableToRead ())
			throw new CacheStatusException ("MatrixObject not available to read.");
				
		//check status, 
		boolean pWrite = !fName.equals(_hdfsFileName); //persistent write flag

		if( isDirty() || pWrite ) //use dirty for skipping parallel exports
		{
			// CASE 1: write matrix to fname 
			// (at this point, obj should never be in state empty)

			getCache();
			acquire( false, true ); //incl. read matrix if evicted
			
			try
			{
				if (DMLScript.DEBUG)
					System.out.println("Exporting " + this.getDebugName() + " to " + fName + " in format " + outputFormat);
				
				writeMetaData (fName, outputFormat);
				writeMatrixToHDFS (fName, outputFormat);
				if ( !pWrite )
					_dirtyFlag = false;
			}
			catch (Exception e)
			{
				throw new CacheIOException (fName + " : Export failed.", e);
			}
			finally
			{
				release();
			}
		}
		else if(DMLScript.DEBUG)  
		{
			//CASE 3: data already in hdfs (do nothing, no need for export)
			System.out.println(this.getDebugName() + ": Skip export to hdfs since data already exists.");
		}
		
		
		/* TODO replace implementation as soon as empty_state semantics (file existence, output format) are clarified 
		//prevent concurrent modifications
		if (! isAvailableToRead ())
			throw new CacheStatusException ("MatrixObject not available to read.");
				
		//check status, 
		boolean pWrite = !fName.equals(_hdfsFileName); //persistent write flag

		if( isDirty() )
		{
			// CASE 1: write matrix to fname 
			// (at this point, obj should never be in state empty)
			
			getCache();
			acquire (false); //incl. read matrix if evicted
			try
			{
				writeMetaData (fName);
				writeMatrixToHDFS (fName, outputFormat);
				if ( !pWrite )
					_dirtyFlag = false;
			}
			catch (Exception e)
			{
				throw new CacheIOException (fName + " : Export failed.", e);
			}
			finally
			{
				release();
			}
		}
		else //not dirty (same content as in _hdfsFileName or empty) 
		{
			if( pWrite )
			{
				//CASE 2: copy data from one hdfs location to another
				//(for scalability this is done w/o trying to read if empty)
				
				//TODO: investigate if this case of empty MatrixObjectNew can happen at all.
				if( isEmpty() ) //if not empty (and not dirty) the respective file should exist
					System.out.println("Warning: input file for persistent write might not exist on hdfs or has a different output format.");
	
				getCache();
				acquire (false);
				try
				{
					writeMetaData (fName);
					MapReduceTool.copyFileOnHDFS(_hdfsFileName, fName); //keep original data format
					//TODO investigate data output format problem 
				}
				catch (Exception e)
				{
					throw new CacheIOException (fName + " : Export failed.", e);
				}
				finally
				{
					release();
				}
			}
			else if(DMLScript.DEBUG) 
			{
				//CASE 3: data already in hdfs (do nothing, no need for export)
				System.out.println("Skip export to hdfs since data already exists.");
			}
		}
		*/
	}

	
	// *********************************************
	// ***                                       ***
	// ***       HIGH-LEVEL PUBLIC METHODS       ***
	// ***     FOR PARTITIONED MATRIX ACCESS     ***
	// ***   (all other methods still usable)    ***
	// ***                                       ***
	// *********************************************
	
	/**
	 * 
	 */
	public void setPartitioned()
	{
		_partitioned = true;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isPartitioned()
	{
		return _partitioned;
	}
	
	/**
	 * NOTE: for reading matrix partitions, we could cache (in its real sense) the read block
	 * with soft references (no need for eviction, as partitioning only applied for read-only matrices).
	 * However, since we currently only support row- and column-wise partitioning caching is not applied yet.
	 * This could be changed once we also support column-block-wise and row-block-wise. Furthermore,
	 * as we reject to partition vectors and support only full row or column indexing, no metadata (apart from
	 * the partition flag) is required.  
	 * 
	 * @param pred
	 * @return
	 * @throws CacheException
	 */
	public synchronized MatrixBlock readMatrixPartition( IndexRange pred ) 
		throws CacheException
	{
		if ( !_partitioned )
			throw new CacheStatusException ("MatrixObject not available to indexed read.");
		
		String fname = getFileName( pred );
		MatrixBlock mb = null;
		
		try
		{
			//System.out.println("Reading partitioned matrix block "+fname);
			mb = readMatrixFromHDFS( fname );
			
			//TODO MB> check if special treatment of non-existing partitions necessary (e.g., for very sparse datasets)
		}
		catch(Exception ex)
		{
			throw new CacheException(ex);
		}
		
		return mb;
	}
	
	
	/**
	 * 
	 * @param pred
	 * @return
	 * @throws CacheStatusException 
	 */
	public String getFileName( IndexRange pred ) 
		throws CacheStatusException
	{
		if ( !_partitioned )
			throw new CacheStatusException ("MatrixObject not available to indexed read.");
		
		String fname = _hdfsFileName;
		if( pred.colStart == pred.colEnd )
			fname = fname + "/" + pred.colStart;
		else if( pred.rowStart == pred.rowEnd )
			fname = fname + "/" + pred.rowStart;
		else
			throw new CacheStatusException ("MatrixObject not available to indexed read.");

		return fname;
	}	
	

	// *********************************************
	// ***                                       ***
	// ***      LOW-LEVEL PROTECTED METHODS      ***
	// ***         EXTEND CACHEABLE DATA         ***
	// ***     ONLY CALLED BY THE SUPERCLASS     ***
	// ***                                       ***
	// *********************************************
	

	@Override
	protected boolean isBlobPresent()
	{
		return (_data != null);
	}

	@Override
	protected void evictBlobFromMemory ( MatrixBlock mb ) 
		throws CacheIOException
	{
		if( LDEBUG )
			System.out.println("EVICTION of Matrix "+_varName+" "+ _hdfsFileName +" (status="+getStatusAsString()+") at "+Runtime.getRuntime().freeMemory()/(1024*1024)+"MB free");

		_data = mb; //reference to garbage-collected matrix block
			
		String filePath = getCacheFilePathAndName ();
		long begin=0;
		if (DMLScript.DEBUG) 
		{
			System.out.println ("\t CACHE: Evicting matrix...  " + _varName + " HDFS path: " + 
					(_hdfsFileName == null ? "null" : _hdfsFileName));
			System.out.println ("\t        Eviction path: " + filePath);
			begin = System.currentTimeMillis();
		}

		try
		{
			writeMatrix (filePath);
		}
		catch (IOException e)
		{
			throw new CacheIOException (filePath + " : Eviction failed.", e);
		}
		catch (DMLRuntimeException e)
		{
			throw new CacheIOException (filePath + " : Eviction failed.", e);
		}
		
		//clear data and cache
		_data.clearEnvelope ();
		_data = null;
		clearCache();
		
		if (DMLScript.DEBUG) 
		{
			System.out.println ("\t\tEvicting matrix COMPLETED ... " + (System.currentTimeMillis()-begin) + " msec.");
		}
	}
	
	@Override
	protected void restoreBlobIntoMemory () 
		throws CacheIOException, CacheAssignmentException
	{
		if( LDEBUG ) 
			System.out.println("RESTORE of Matrix "+_varName+", "+_hdfsFileName);
		
		String filePath = getCacheFilePathAndName();
		long begin = 0;
		if (DMLScript.DEBUG) 
		{
			System.out.println ("\t CACHE: Restoring matrix...  " + _varName + "  HDFS path: " + 
					(_hdfsFileName == null ? "null" : _hdfsFileName));
			System.out.println ("\t        Restore from path: " + filePath);
			begin = System.currentTimeMillis();
		}
		
		if (_data != null)
			throw new CacheIOException (filePath + " : Cannot restore on top of existing in-memory data.");
		MatrixBlock newData = null;
		try
		{
			newData = readMatrix (filePath);
		}
		catch (IOException e)
		{
			throw new CacheIOException (filePath + " : Restore failed.", e);
		}
	    if (newData != null)
	    {
	    	newData.setEnvelope (this);
			_data = newData;
	    	freeEvictedBlob();
	    }
	    else
			throw new CacheIOException (filePath + " : Restore failed.");
	    
		if (DMLScript.DEBUG) 
		{
			System.out.println ("\t\tRestoring matrix - COMPLETED ... " + (System.currentTimeMillis()-begin) + " msec.");
		}
	}		

	@Override
	protected void freeEvictedBlob ()
	{
		String cacheFilePathAndName = getCacheFilePathAndName ();
		long begin = 0;
		if (DMLScript.DEBUG) 
		{
			System.out.println ("\t CACHE: Freeing evicted matrix...  " + _varName + "  HDFS path: " + 
					(_hdfsFileName == null ? "null" : _hdfsFileName));
			System.out.println ("\t        Eviction path: " + cacheFilePathAndName);
			begin = System.currentTimeMillis();
		}
		
		switch (CacheableData.cacheEvictionStorageType)
		{
			case LOCAL:
				new File(cacheFilePathAndName).delete();
				break;
			case HDFS:
				try
				{
					(FileSystem.get (new Configuration())).delete (new Path (cacheFilePathAndName), false);
				}
				catch (IOException e)
				{
					;
				}
				break;
		}
		
		if (DMLScript.DEBUG) 
		{
			System.out.println ("\t\tFreeing evicted matrix - COMPLETED ... " + (System.currentTimeMillis()-begin) + " msec.");
		}
	}
	
	// *******************************************
	// ***                                     ***
	// ***      LOW-LEVEL PRIVATE METHODS      ***
	// ***           FOR MATRIX I/O            ***
	// ***                                     ***
	// *******************************************
	
	
	private String getCacheFilePathAndName ()
	{
		if( _cacheFileName==null )
		{
			StringBuffer sb = new StringBuffer();
			switch (CacheableData.cacheEvictionStorageType)
			{
				case LOCAL:
					sb.append(CacheableData.cacheEvictionLocalFilePath); 
					sb.append(CacheableData.cacheEvictionLocalFilePrefix);
					sb.append(String.format ("%09d", getUniqueCacheID()));
					sb.append(CacheableData.cacheEvictionLocalFileExtension);
					break;
					
				case HDFS:
					sb.append(CacheableData.cacheEvictionHDFSFilePath);
					sb.append(CacheableData.cacheEvictionHDFSFilePrefix);
					sb.append(String.format ("%09d", getUniqueCacheID()));
					sb.append(CacheableData.cacheEvictionHDFSFileExtension);
					break;
			}
			
			_cacheFileName = sb.toString();
		}
		
		return _cacheFileName;
	}
	
	private MatrixBlock readMatrix (String filePathAndName)
		throws IOException
	{
		//System.out.println("read matrix");
		
		MatrixBlock newData = null;
		switch (CacheableData.cacheEvictionStorageType)
		{
			case LOCAL:
				newData = readMatrixFromLocal (filePathAndName);
				break;			
			case HDFS:
				newData = readMatrixFromHDFS (filePathAndName);
				break;
			default:
				throw new IOException (filePathAndName + 
						" : Cannot read matrix from unsupported storage type \""
						+ CacheableData.cacheEvictionStorageType + "\"");				
		}
		return newData;
	}
	
	private MatrixBlock readMatrixFromHDFS (String filePathAndName)
		throws IOException
	{
		//System.out.println("read hdfs "+filePathAndName);
		
		long begin = 0;
		if (DMLScript.DEBUG) 
		{
			System.out.println ("    Reading matrix from HDFS...  " + _varName + "  Path: " + filePathAndName);
			begin = System.currentTimeMillis();
		}

		MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;
		MatrixCharacteristics mc = iimd.getMatrixCharacteristics ();
		MatrixBlock newData = DataConverter.readMatrixFromHDFS(filePathAndName, iimd.getInputInfo(),
				                           mc.get_rows(), mc.get_cols(), mc.numRowsPerBlock, mc.get_cols_per_block());
		
		if( newData == null )
		{
			throw new IOException("Unable to load matrix from file "+filePathAndName);
		}
		
		newData.clearEnvelope ();
   		
		if (DMLScript.DEBUG) 
		{
			//System.out.println ("    Reading Completed: read ~" + newData.getObjectSizeInMemory() + " bytes, " + (System.currentTimeMillis()-begin) + " msec.");
			System.out.println ("    Reading Completed: " + (System.currentTimeMillis()-begin) + " msec.");
		}
		return newData;
	}

	private MatrixBlock readMatrixFromLocal (String filePathAndName)
		throws FileNotFoundException, IOException
	{
		//System.out.println("read local");
		
		DataInputStream in = new DataInputStream (new FileInputStream (filePathAndName));
		MatrixBlock newData = new MatrixBlock ();
		try {
			newData.readFields (in);
		}
		finally{
			in.close ();
		}
   		newData.clearEnvelope ();
   		
		return newData;
	}
	
	private void writeMatrix (String filePathAndName)
		throws DMLRuntimeException, IOException
	{
		switch (CacheableData.cacheEvictionStorageType)
		{
			case LOCAL:
				writeMatrixToLocal (filePathAndName);
				break;			
			case HDFS:
				writeMatrixToHDFS (filePathAndName, null);
				break;
			default:
				throw new IOException (filePathAndName + 
						" : Cannot write matrix to unsupported storage type \""
						+ CacheableData.cacheEvictionStorageType + "\"");				
		}
	}

	/**
	 * Writes in-memory matrix to HDFS in a specified format.
	 * 
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private void writeMatrixToHDFS (String filePathAndName, String outputFormat)
		throws DMLRuntimeException, IOException
	{
		//System.out.println("write matrix "+_varName+" "+filePathAndName);
		
		long begin = 0;
		if (DMLScript.DEBUG) 
		{
			System.out.println ("    Writing matrix to HDFS...  " + _varName + "  Path: " + filePathAndName + ", Format: " +
					(outputFormat != null ? outputFormat : "inferred from metadata"));
			begin = System.currentTimeMillis();
		}
		
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;

		if (_data != null)
		{
			// Get the dimension information from the metadata stored within MatrixObject
			MatrixCharacteristics mc = iimd.getMatrixCharacteristics ();
			// Write the matrix to HDFS in requested format
			OutputInfo oinfo = (outputFormat != null ? OutputInfo.stringToOutputInfo (outputFormat) 
					                                 : InputInfo.getMatchingOutputInfo (iimd.getInputInfo ()));
			DataConverter.writeMatrixToHDFS (_data, filePathAndName, oinfo, mc.get_rows(), mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block());

			if (DMLScript.DEBUG) 
			{
				System.out.println ("    Writing matrix to HDFS ("+filePathAndName+") - COMPLETED... " + (System.currentTimeMillis()-begin) + " msec.");
			}
		}
		else if (DMLScript.DEBUG) 
		{
			System.out.println ("Writing matrix to HDFS ("+filePathAndName+") - NOTHING TO WRITE (_data == null).");
		}
	}
	
	private synchronized void writeMatrixToLocal (String filePathAndName)
		throws FileNotFoundException, IOException
	{
		//System.out.println("write matrix local");
		
		DataOutputStream out = new DataOutputStream (new FileOutputStream (filePathAndName));
		try {
			_data.write (out);
		}
		finally{
			out.close ();	
		}
   		
	}

	private void writeMetaData (String filePathAndName, String outputFormat)
		throws DMLRuntimeException, IOException
	{
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;
	
		if (_data != null)
		{
			// Get the dimension information from the metadata stored within MatrixObject
			MatrixCharacteristics mc = iimd.getMatrixCharacteristics ();
			// Write the matrix to HDFS in requested format			
			OutputInfo oinfo = (outputFormat != null ? OutputInfo.stringToOutputInfo (outputFormat) 
                    : InputInfo.getMatchingOutputInfo (iimd.getInputInfo ()));
			MapReduceTool.writeMetaDataFile (filePathAndName + ".mtd", valueType, mc, oinfo);
		}
	}
	
	@Override
	public synchronized String getDebugName()
	{
		final int maxLength = 23;
		String debugNameEnding = (_hdfsFileName == null ? "null" : 
			(_hdfsFileName.length() < maxLength ? _hdfsFileName : "..." + 
				_hdfsFileName.substring (_hdfsFileName.length() - maxLength + 3)));
		return _varName + " " + debugNameEnding;
	}

	
	// *******************************************
	// ***                                     ***
	// ***      LOW-LEVEL PRIVATE METHODS      ***
	// ***       FOR SOFTREFERENCE CACHE       ***
	// ***                                     ***
	// *******************************************
	
	
	private void createCache( ) 
	{
		_cache = new CacheReference( _data );	
	}

	private void getCache()
	{
		if( _data == null && _cache !=null )
		{
			_data = _cache.get();
			
			if( _data != null ) 
			{
				clearCache();
			}
		}
	}
	
	private void clearCache()
	{
		if( _cache != null )
		{
			_cache.clear();
			_cache = null;
		}
	}
	
	/**
	 * see clear data
	 * 
	 * @param flag
	 */
	public void enableCleanup(boolean flag) 
	{
		//System.out.println("enable cleanup "+_varName+": "+flag);
		_cleanupFlag = flag;
	}

	/**
	 * see clear data
	 * 
	 * @return
	 */
	public boolean isCleanupEnabled() 
	{
		return _cleanupFlag;
	}
	
	public class CacheReference extends SoftReference<MatrixBlock>
	{
		private MatrixBlock _imb = null;
		
		public CacheReference(MatrixBlock o) 
		{
			//put object into cache
			super( o );
			
			//prepare recovery object
			_imb = o.createShallowCopy();
		}
		
		/**
		 * Guarantees to return the cached matrix block, if not cleared/evicted before.
		 * If the matrixblock is already collected by the garbage collector, but eviction
		 * did not happen so far, a recovery object is returned. 
		 */
		@Override
		public MatrixBlock get() 
		{
			// get the cache referent
			MatrixBlock ret = super.get();
			
			//get the recovery object if required
			if( ret == null )
			{
				ret = _imb; 
			}
			
			return ret;
		}

		@Override
		public void clear() 
		{
			//clear referent in soft reference
			super.clear();
			
			//clear recovery object
			_imb.clearEnvelope();
			_imb = null;
		}	
	}
}

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.CacheableData;
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
import com.ibm.bi.dml.utils.CacheOutOfMemoryException;
import com.ibm.bi.dml.utils.CacheStatusException;
import com.ibm.bi.dml.utils.DMLRuntimeException;


/**
 * Represents a matrix in controlprogram. This class contains method to read
 * matrices from HDFS and convert them to a specific format/representation. It
 * is also able to write several formats/representation of matrices to HDFS.
 * 
 * IMPORTANT: Preserve one-to-one correspondence between {@link MatrixObjectNew}
 * and {@link MatrixBlock} objects, for cache purposes.  Do not change a
 * {@link MatrixBlock} object without informing its {@link MatrixObjectNew} object.
 * 
 */
public class MatrixObjectNew extends CacheableData
{
	/**
	 * Container object that holds the actual data.
	 */
	protected MatrixBlock _data;

	/**
	 * The name of HDFS file by which the data is backed up.
	 */
	private String _hdfsFileName = null; // file name and path
	
	/**
	 * <code>true</code> if the in-memory or evicted matrix may be different from
	 * the matrix located at {@link #_hdfsFileName}; <code>false</code> if the two
	 * matrices should be the same.
	 */
	private boolean dirtyFlag = false;

	/**
	 * Object that holds the metadata associated with the matrix, which
	 * includes: 1) Matrix dimensions, if available 2) Number of non-zeros, if
	 * available 3) Block dimensions, if applicable 4) InputInfo -- subsequent
	 * operations that use this Matrix expect it to be in this format.
	 * 
	 * When the matrix is written to HDFS (local file system, as well?), one
	 * must get the OutputInfo that matches with InputInfo stored inside _mtd.
	 */
	protected MetaData _metaData;

	/**
	 * Default constructor
	 */
	public MatrixObjectNew ()
	{
		super (DataType.MATRIX, ValueType.DOUBLE); // DOUBLE is the default value type
		_data = null;
		_metaData = null;
		dirtyFlag = false;
	}

	/**
	 * Constructor that takes both HDFS filename and associated metadata.
	 * @throws CacheStatusException 
	 * @throws CacheOutOfMemoryException
	 */
	public MatrixObjectNew (ValueType vt, String file, MetaData mtd)
	throws CacheOutOfMemoryException, CacheStatusException
	{
		super (DataType.MATRIX, vt);
		_metaData = mtd; // Metadata
		_data = null;
		_hdfsFileName = file;
		dirtyFlag = false;
	}

	/**
	 * Constructor that takes only the HDFS filename.
	 */
	public MatrixObjectNew (ValueType vt, String file)
	{
		super (DataType.MATRIX, vt);
		_hdfsFileName = file; // HDFS file path
		_metaData = null;
		_data = null;
		dirtyFlag = false;
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
	 */
	public void refreshMetaData ()
	{
		MatrixCharacteristics mcOld = (_metaData == null ? null : 
			((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics ());
		
		MatrixBlock theData = _data;
		if (theData == null)
			return;
		
		long numRows = theData.getNumRows ();
		long numColumns = theData.getNumColumns ();
		long nonZeros = theData.getNonZeros ();
		
		int numRowsPerBlock = 
			(mcOld == null ? -1 : mcOld.get_rows_per_block ());  // TODO Set according to theData
		int numColumnsPerBlock = 
			(mcOld == null ? -1 : mcOld.get_cols_per_block ());  // TODO Set according to theData
		
		MatrixCharacteristics mcNew = 
			new MatrixCharacteristics (numRows, numColumns, numRowsPerBlock, numColumnsPerBlock, nonZeros);
		
		if (_metaData == null)
			_metaData = new  MatrixDimensionsMetaData (mcNew);
		else if (! mcNew.equals (mcOld))
			((MatrixDimensionsMetaData) _metaData).setMatrixCharacteristics (mcNew);
	}

	public String getFileName ()
	{
		return _hdfsFileName;
	}

	public synchronized void setFileName (String file)
	{
		if (! this._hdfsFileName.equals (file))
		{
			this._hdfsFileName = file;
			if (! isEmpty ())
				dirtyFlag = true;
		}
	}

	public int getNumRows () throws DMLRuntimeException
	{
		if (_metaData == null)
			refreshMetaData ();
		MatrixCharacteristics mc = ((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics ();
		long numRows = mc.get_rows ();
		if (numRows <= 2147483647L)
			return (int) numRows;
		else
			throw new DMLRuntimeException ("Number of rows is " + numRows + ", which is too large for an \"int\"");
	}

	public int getNumColumns() throws DMLRuntimeException
	{
		if (_metaData == null)
			refreshMetaData ();
		MatrixCharacteristics mc = ((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics ();
		long numColumns = mc.get_cols ();
		if (numColumns <= 2147483647L)
			return (int) numColumns;
		else
			throw new DMLRuntimeException ("Number of columns is " + numColumns + ", which is too large for an \"int\"");
	}
	
	/**
	 * <code>true</code> if the in-memory or evicted matrix may be different from
	 * the matrix located at {@link #_hdfsFileName}; <code>false</code> if the two
	 * matrices are supposed to be the same.
	 */
	public boolean isDirty ()
	{
		return dirtyFlag;
	}
	
	public String toString()
	{ 
		StringBuilder str = new StringBuilder();
		str.append("MatrixObject: ");
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
							// TODO Auto-generated catch block
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
	 * In-Status:  EMPTY, EVICTABLE, EVICTED, READ;
	 * Out-Status: READ(+1).
	 * 
	 * @return the matrix data reference
	 * @throws CacheException 
	 */
	public MatrixBlock acquireRead () throws CacheException
	{
		if (! isAvailableToRead ())
			throw new CacheStatusException ();
		MatrixBlock newData = null;
		String fName = _hdfsFileName;
		if (isEmpty () && fName != null)
		{
			try
			{
				newData = readMatrix (fName);
			}
			catch (IOException e)
			{
				throw new CacheIOException (fName + " : Reading failed.", e);
			}
		}
		synchronized (this)
		{
			if (isEmpty () && newData != null)
			{
				newData.setEnvelope (this);
				_data = newData;
				dirtyFlag = false;
			}
			acquire (false);
		}
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
	public MatrixBlock acquireModify ()
	throws CacheException
	{
		if (! isAvailableToModify ())
			throw new CacheStatusException ();
		MatrixBlock newData = null;
		String fName = _hdfsFileName;
		if (isEmpty () && fName != null)
		{
			try
			{
				newData = readMatrix (fName);
			}
			catch (IOException e)
			{
				throw new CacheIOException (fName + " : Reading failed.", e);
			}
		}
		synchronized (this)
		{
			if (isEmpty () && newData != null)
			{
				newData.setEnvelope (this);
				_data = newData;
			}
			acquire (true);
			dirtyFlag = true;
		}
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
	public MatrixBlock acquireModify (MatrixBlock newData) 
	throws CacheException
	{
		if (! isAvailableToModify ())
			throw new CacheStatusException ();
		clearData ();
		synchronized (this)
		{
			acquire (true);
			dirtyFlag = true;
			if (newData != null)
			{
				newData.setEnvelope (this);
				_data = newData;
				refreshMetaData ();
			}
		}
		return _data;
	}

	/**
	 * Releases the shared ("read-only") or exclusive ("write") lock.  Updates
	 * the matrix size, last-access time, metadata, etc.
	 * 
	 * In-Status:  READ, MODIFY;
	 * Out-Status: READ(-1), EVICTABLE, EMPTY.
	 * 
	 * @throws CacheOutOfMemoryException 
	 * @throws CacheStatusException
	 */
	@Override
	public synchronized void release () 
	throws CacheOutOfMemoryException, CacheStatusException
	{
		if (isModify ())
		{
			dirtyFlag = true;
			refreshMetaData ();
		}
		super.release ();
	}

	/**
	 * Downgrades the exclusive ("write") lock down to a shared ("read-only") lock.
	 * May be used, for example, prior to exporting a newly computed matrix to HDFS.
	 * Updates the matrix size, last-access time, metadata, etc.

	 * In-Status:  MODIFY;
	 * Out-Status: READ(1).
	 * 
	 * @throws CacheOutOfMemoryException 
	 * @throws CacheStatusException
	 */
	@Override
	public synchronized void downgrade ()
	throws CacheOutOfMemoryException, CacheStatusException
	{
		dirtyFlag = true;
		refreshMetaData ();
		super.downgrade ();
	}

	/**
	 * Sets the matrix data reference to <code>null</code>, abandons the old matrix.
	 * Makes the "envelope" empty.  Run it to finalize the matrix (otherwise the
	 * evicted matrix file may remain undeleted).
	 * 
	 * In-Status:  EMPTY, EVICTABLE, EVICTED;
	 * Out-Status: EMPTY.
	 * 
	 * @throws CacheOutOfMemoryException 
	 * @throws CacheStatusException
	 */
	public synchronized void clearData () 
	throws CacheOutOfMemoryException, CacheStatusException
	{
		if (! isAvailableToModify ())
			throw new CacheStatusException ();
		if (_data != null)
			_data.clearEnvelope ();
		if (! isEmpty())
		{
			registerBlobDeletion ();
			_data = null;
			dirtyFlag = false;
		}
	}

	/**
	 * Same as {@link #clearData()}, but in addition, sets the HDFS source file name
	 * for the matrix.  The next {@link #acquireRead()} will read it from HDFS.  So,
	 * this is a "lazy" import.
	 * 
	 * In-Status:  EMPTY, EVICTABLE, EVICTED;
	 * Out-Status: EMPTY.
	 * 
	 * @param filePath : the new HDFS path and file name
	 * @throws CacheOutOfMemoryException 
	 * @throws CacheStatusException
	 */
	public void importData (String filePath) 
	throws CacheOutOfMemoryException, CacheStatusException
	{
		if (! isAvailableToModify ())
			throw new CacheStatusException ();
		_hdfsFileName = filePath;
		clearData ();
	}

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
		if (! isAvailableToRead ())
			throw new CacheStatusException ();
		if (! isEmpty ())
		{
			String fName = _hdfsFileName;
			acquire (false);
			try
			{
				writeMetaData (fName);
				writeMatrix (fName);
				dirtyFlag = false;
			}
			catch (IOException e)
			{
				throw new CacheIOException (fName + " : Export failed.", e);
			}
			catch (DMLRuntimeException e)
			{
				throw new CacheIOException (fName + " : Export failed.", e);
			}
			finally
			{
				release ();
			}
		}
	}
	

	// *********************************************
	// ***                                       ***
	// ***      LOW-LEVEL PROTECTED METHODS      ***
	// ***         EXTEND CACHEABLE DATA         ***
	// ***     ONLY CALLED BY THE SUPERCLASS     ***
	// ***                                       ***
	// *********************************************
	
	
	@Override
	protected synchronized long getBlobSize ()
	{
		return (_data == null ? 0 : _data.getObjectSizeInMemory ());
	}

	@Override
	protected synchronized boolean isBlobPresent()
	{
		return (_data != null);
	}

	@Override
	protected synchronized void evictBlobFromMemory () throws CacheIOException
	{
		String filePath = getCacheFilePathAndName ();
		if (DMLScript.DEBUG) 
		{
			System.out.println ("CACHE: Evicting matrix...  Original HDFS path: " + 
					(_hdfsFileName == null ? "null" : _hdfsFileName));
			System.out.println ("       Eviction path: " + filePath);
		}
		
		if (_data == null)
			throw new CacheIOException (filePath + " : Nothing to evict.");
		try
		{
			switch (DMLScript.cacheEvictionStorageType)
			{
			case LOCAL:
				writeMatrixToLocal (filePath);
				break;			
			case HDFS:
				writeMatrix (filePath);
				break;
			default:
				throw new CacheIOException (filePath + 
						" : Cannot evict to unsupported storage type \""
						+ DMLScript.cacheEvictionStorageType + "\"");				
			}
		}
		catch (IOException e)
		{
			throw new CacheIOException (filePath + " : Eviction failed.", e);
		}
		catch (DMLRuntimeException e)
		{
			throw new CacheIOException (filePath + " : Eviction failed.", e);
		}
		_data.clearEnvelope ();
		_data = null;
		
		if (DMLScript.DEBUG) 
		{
			System.out.println ("CACHE: Evicting matrix - COMPLETED.");
		}
	}
	
	@Override
	protected synchronized void restoreBlobIntoMemory () 
	throws CacheIOException, CacheAssignmentException
	{
		String filePath = getCacheFilePathAndName ();
		if (DMLScript.DEBUG) 
		{
			System.out.println ("CACHE: Restoring matrix...  Original HDFS path: " + 
					(_hdfsFileName == null ? "null" : _hdfsFileName));
			System.out.println ("       Restore from path: " + filePath);
		}
		
		if (_data != null)
			throw new CacheIOException (filePath + " : Cannot restore on top of existing in-memory data.");
		MatrixBlock newData = null;
		try
		{
			switch (DMLScript.cacheEvictionStorageType)
			{
			case LOCAL:
				newData = readMatrixFromLocal (filePath);
				break;			
			case HDFS:
				newData = readMatrix (filePath);
				break;
			default:
				throw new CacheIOException (filePath + 
						" : Cannot restore from unsupported storage type \""
						+ DMLScript.cacheEvictionStorageType + "\"");				
			}
		}
		catch (IOException e)
		{
			throw new CacheIOException (filePath + " : Restore failed.", e);
		}
	    if (newData != null)
	    {
	    	newData.setEnvelope (this);
			_data = newData;
	    	freeEvictedBlob ();
	    }
	    else
			throw new CacheIOException (filePath + " : Restore failed.");
	    
		if (DMLScript.DEBUG) 
		{
			System.out.println ("CACHE: Restoring matrix - COMPLETED.");
		}
	}		
	
	@Override
	protected synchronized void freeEvictedBlob ()
	{
		String cacheFilePathAndName = getCacheFilePathAndName ();
		
		if (DMLScript.DEBUG) 
		{
			System.out.println ("CACHE: Freeing evicted matrix...  Original HDFS path: " + 
					(_hdfsFileName == null ? "null" : _hdfsFileName));
			System.out.println ("       Eviction path: " + cacheFilePathAndName);
		}
		
		switch (DMLScript.cacheEvictionStorageType)
		{
		case LOCAL:
			(new File (cacheFilePathAndName)).delete ();
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
			System.out.println ("CACHE: Freeing evicted matrix - COMPLETED.");
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
		String fileName = "";
		switch (DMLScript.cacheEvictionStorageType)
		{
		case LOCAL:
			fileName += 
				DMLScript.cacheEvictionLocalFilePath + 
				DMLScript.cacheEvictionLocalFilePrefix +
				String.format ("%09d", getUniqueCacheID ()) +
				DMLScript.cacheEvictionLocalFileExtension;
			break;
			
		case HDFS:
			fileName += 
				DMLScript.cacheEvictionHDFSFilePath +
				DMLScript.cacheEvictionHDFSFilePrefix +
				String.format ("%09d", getUniqueCacheID ()) +
				DMLScript.cacheEvictionHDFSFileExtension;
			break;
		}
		return fileName;
	}
	
	private MatrixBlock readMatrix (String filePathAndName)
	throws IOException
	{
		if (DMLScript.DEBUG) 
		{
			System.out.println ("Reading matrix from HDFS...  Path: " + filePathAndName);
		}

		MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;
		MatrixCharacteristics mc = iimd.getMatrixCharacteristics ();
		MatrixBlock newData = null;
		newData = DataConverter.readMatrixFromHDFS 
			(filePathAndName, iimd.getInputInfo(), mc.get_rows(), mc.get_cols(), mc.numRowsPerBlock, mc.get_cols_per_block());
   		newData.clearEnvelope ();
   		
		if (DMLScript.DEBUG) 
		{
			System.out.println ("Reading matrix from HDFS - COMPLETED.");
		}
		return newData;
	}

	/**
	 * Writes in-memory matrix to disk.
	 * 
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private void writeMatrix (String filePathAndName)
	throws DMLRuntimeException, IOException
	{
		if (DMLScript.DEBUG) 
		{
			System.out.println ("Writing matrix to HDFS...  Path: " + filePathAndName);
		}
   		
		MatrixFormatMetaData iimd;
		MatrixBlock theData;
		synchronized (this)
		{
			if (_metaData == null)
				refreshMetaData ();
			iimd = (MatrixFormatMetaData) _metaData;
			theData = _data;
		}
		if (theData != null)
		{
			// Get the dimension information from the metadata stored within MatrixObject
			MatrixCharacteristics mc = iimd.getMatrixCharacteristics ();
			// Write the matrix to HDFS in requested format
			OutputInfo oinfo = InputInfo.getMatchingOutputInfo (iimd.getInputInfo ());
			DataConverter.writeMatrixToHDFS (theData, filePathAndName, oinfo, mc.get_rows(), mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block());

			if (DMLScript.DEBUG) 
			{
				System.out.println ("Writing matrix to HDFS - COMPLETED.");
			}
		}
		else if (DMLScript.DEBUG) 
		{
			System.out.println ("Writing matrix to HDFS - NOTHING TO WRITE (_data == null).");
		}
	}
	
	private void writeMetaData (String filePathAndName)
	throws DMLRuntimeException, IOException
	{
		MatrixFormatMetaData iimd;
		MatrixBlock theData;
		synchronized (this)
		{
			if (_metaData == null)
				refreshMetaData ();
			iimd = (MatrixFormatMetaData) _metaData;
			theData = _data;
		}
		if (theData != null)
		{
			// Get the dimension information from the metadata stored within MatrixObject
			MatrixCharacteristics mc = iimd.getMatrixCharacteristics ();
			// Write the matrix to HDFS in requested format
			OutputInfo oinfo = InputInfo.getMatchingOutputInfo (iimd.getInputInfo ());
			MapReduceTool.writeMetaDataFile (filePathAndName + ".mtd", valueType, mc, oinfo);
		}
	}
	
	private MatrixBlock readMatrixFromLocal (String filePathAndName)
	throws FileNotFoundException, IOException
	{
		MatrixBlock newData = null;
		DataInputStream in = null;
    	// in = new ObjectInputStream (new FileInputStream (filePathAndName));
    	// newData = (MatrixBlock) in.readObject ();
    	in = new DataInputStream (new FileInputStream (filePathAndName));
		newData = new MatrixBlock ();
		newData.readFields (in);
   		in.close ();
   		newData.clearEnvelope ();
		return newData;
	}

	private void writeMatrixToLocal (String filePathAndName)
	throws FileNotFoundException, IOException
	{
		DataOutputStream out = null;
		// out = new ObjectOutputStream (new FileOutputStream (filePathAndName));
		// out.writeObject (_data);	
		out = new DataOutputStream (new FileOutputStream (filePathAndName));
		_data.write (out);
   		out.close ();
	}
}

package dml.runtime.instructions.CPInstructions;

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

import dml.api.DMLScript;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.runtime.controlprogram.CacheableData;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MatrixDimensionsMetaData;
import dml.runtime.matrix.MatrixFormatMetaData;
import dml.runtime.matrix.MetaData;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.util.DataConverter;
import dml.runtime.util.MapReduceTool;
import dml.utils.CacheOutOfMemoryException;
import dml.utils.CacheIOException;
import dml.utils.CacheStatusException;
import dml.utils.DMLRuntimeException;

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
	/* The name of HDFS file by which the data is backed up.
	 */
	private String _hdfsFileName; // file name and path

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
		_hdfsFileName = null;
	}

	/**
	 * Constructor that takes both HDFS filename and associated metadata.
	 */
	public MatrixObjectNew (ValueType vt, String file, MetaData mtd)
	{
		super (DataType.MATRIX, vt);
		_hdfsFileName = file; // HDFS file path
		_metaData = mtd; // Metadata
		_data = null;
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
		MatrixBlock theData = _data;
		if (theData == null)
			return;
		MatrixCharacteristics mc = ((_metaData == null) ?
			new MatrixCharacteristics (theData.getNumRows(), theData.getNumColumns(), -1, -1, theData.getNonZeros()) :
				((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics () );
		
		int numRows = theData.getNumRows ();
		int numColumns = theData.getNumColumns();
		
		mc.setDimension (numRows, numColumns);
		mc.setNonZeros (theData.getNonZeros());
			
		if ( _metaData == null )
			_metaData = new  MatrixDimensionsMetaData (mc);
		else
			((MatrixDimensionsMetaData) _metaData).setMatrixCharacteristics (mc);
	}

	public String getFileName ()
	{
		return _hdfsFileName;
	}

	public void setFileName (String file)
	{
		this._hdfsFileName = file;
	}

	public int getNumRows () throws DMLRuntimeException
	{
		MatrixCharacteristics mc = ((MatrixDimensionsMetaData)_metaData).getMatrixCharacteristics();
		long numRows = mc.get_rows ();
		if (numRows <= 2147483647L)
			return (int) numRows;
		else
			throw new DMLRuntimeException ("Number of rows is " + numRows + ", which is too large for an \"int\"");
	}

	public int getNumColumns() throws DMLRuntimeException
	{
		MatrixCharacteristics mc = ((MatrixDimensionsMetaData)_metaData).getMatrixCharacteristics();
		long numColumns = mc.get_cols ();
		if (numColumns <= 2147483647L)
			return (int) numColumns;
		else
			throw new DMLRuntimeException ("Number of columns is " + numColumns + ", which is too large for an \"int\"");
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
	 * @throws CacheOutOfMemoryException 
	 * @throws CacheIOException
	 * @throws CacheStatusException
	 */
	public MatrixBlock acquireRead ()
	throws CacheOutOfMemoryException, CacheIOException, CacheStatusException
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
				_data = newData;
				refreshMetaData ();
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
	 * @throws CacheOutOfMemoryException 
	 * @throws CacheIOException
	 * @throws CacheStatusException
	 */
	public MatrixBlock acquireModify ()
	throws CacheOutOfMemoryException, CacheIOException, CacheStatusException
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
				_data = newData;
				refreshMetaData ();
			}
			acquire (true);
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
	 * @throws CacheOutOfMemoryException 
	 * @throws CacheIOException
	 * @throws CacheStatusException
	 */
	public MatrixBlock acquireModify (MatrixBlock newData) 
	throws CacheOutOfMemoryException, CacheIOException, CacheStatusException
	{
		if (! isAvailableToModify ())
			throw new CacheStatusException ();
		clearData ();
		synchronized (this)
		{
			acquire (true);
			if (newData != null)
			{
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
		refreshMetaData ();
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
		registerBlobDeletion ();
		_data = null;
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
	 * @throws CacheOutOfMemoryException 
	 * @throws CacheIOException
	 * @throws CacheStatusException
	 */
	public void exportData ()
	throws CacheOutOfMemoryException, CacheIOException, CacheStatusException
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
		_data = null;
	}
	
	@Override
	protected synchronized void restoreBlobIntoMemory () throws CacheIOException
	{
		String filePath = getCacheFilePathAndName ();
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
			_data = newData;
			refreshMetaData ();
	    	freeEvictedBlob ();
	    }
	    else
			throw new CacheIOException (filePath + " : Restore failed.");
	}		
	
	@Override
	protected synchronized void freeEvictedBlob ()
	{
		String cacheFilePathAndName = getCacheFilePathAndName ();
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
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;
		MatrixCharacteristics mc = iimd.getMatrixCharacteristics ();
		MatrixBlock newData = null;
		newData = DataConverter.readMatrixFromHDFS 
			(filePathAndName, iimd.getInputInfo(), mc.get_rows(), mc.get_cols(), mc.numRowsPerBlock, mc.get_cols_per_block());
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
			MapReduceTool.writeMetaDataFile (filePathAndName + ".mtd", mc, oinfo);
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

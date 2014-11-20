/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.File;
import java.util.ArrayList;

import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.io.MatrixReaderFactory;
import com.ibm.bi.dml.runtime.io.ReaderBinaryBlock;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

public class DistributedCacheInput 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//internal partitioning parameter (threshold and partition size) 
	public static final long PARTITION_SIZE = 4000000; //32MB
	//public static final String PARTITION_SUFFIX = "_dp";
	
	//meta data of cache input
	private Path _localFilePath = null;
	private long _rlen = -1;
	private long _clen = -1;
	private int _brlen = -1;
	private int _bclen = -1;
	private PDataPartitionFormat _pformat = null;
	
	//data cached input
	private IndexedMatrixValue[][] dataBlocks = null;
	
	
	public DistributedCacheInput(Path p, long rows, long cols, int brlen, int bclen, PDataPartitionFormat pformat) 
	{
		_localFilePath = p;
		_rlen  = rows;
		_clen  = cols;
		_brlen = brlen;
		_bclen = bclen;
		_pformat = pformat;
	}

	public long getNumRows() {
		return _rlen;
	}
	
	public long getNumCols() {
		return _clen;
	}
	
	public int getNumRowsPerBlock(){
		return _brlen;
	}
	
	public int getNumColsPerBlock(){
		return _bclen;
	}
	
	/**
	 * 
	 */
	public void reset() 
	{
		_localFilePath = null;
		_rlen  = -1;
		_clen  = -1;
		_brlen = -1;
		_bclen = -1;
		_pformat = null;
	}

	/**
	 * 
	 * @param rowBlockIndex
	 * @param colBlockIndex
	 * @return
	 * @throws DMLRuntimeException
	 */
	public IndexedMatrixValue getDataBlock(int rowBlockIndex, int colBlockIndex)
		throws DMLRuntimeException 
	{
		//probe missing block (read on-demand)
		if( dataBlocks==null || dataBlocks[rowBlockIndex-1][colBlockIndex-1]==null )
			readDataBlocks( rowBlockIndex, colBlockIndex );
		
		//return read or existing block
		return dataBlocks[rowBlockIndex-1][colBlockIndex-1];
	}
	
	/**
	 * 
	 * @param rowBlockSize
	 * @param colBlockSize
	 * @throws DMLRuntimeException
	 */
	private void readDataBlocks( int rowBlockIndex, int colBlockIndex )
		throws DMLRuntimeException 
	{
		//get filename for rowblock/colblock
		String fname = _localFilePath.toString();
		if( isPartitioned() ) 
			fname = getPartitionFileName(rowBlockIndex, colBlockIndex);
			
		//read matrix partition (or entire vector)
		try 
		{		
			ReaderBinaryBlock reader = (ReaderBinaryBlock) MatrixReaderFactory.createMatrixReader(InputInfo.BinaryBlockInputInfo);
			reader.setLocalFS( !MRBaseForCommonInstructions.isJobLocal );
			ArrayList<IndexedMatrixValue> tmp = reader.readIndexedMatrixBlocksFromHDFS(fname, _rlen, _clen, _brlen, _bclen);
			
			int rowBlocks = (int) Math.ceil(_rlen / (double) _brlen);
			int colBlocks = (int) Math.ceil(_clen / (double) _bclen);

			if( dataBlocks==null )
				dataBlocks = new IndexedMatrixValue[rowBlocks][colBlocks];

			for (IndexedMatrixValue val : tmp) {
				MatrixIndexes idx = val.getIndexes();
				dataBlocks[(int) idx.getRowIndex() - 1][(int) idx.getColumnIndex() - 1] = val;
			}
		} 
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		} 
	}
	
	/**
	 * 
	 * @return
	 */
	private boolean isPartitioned()
	{
		return (_pformat != PDataPartitionFormat.NONE);
	}
	

	/**
	 * 
	 * @param rowBlockIndex
	 * @param colBlockIndex
	 * @return
	 * @throws DMLRuntimeException
	 */
	private String getPartitionFileName( int rowBlockIndex, int colBlockIndex  ) 
		throws DMLRuntimeException
	{
		long partition = -1;
		switch( _pformat )
		{
			case ROW_BLOCK_WISE_N:
			{
				long numRowBlocks = (long)Math.ceil(((double)PARTITION_SIZE)/_clen/_brlen); 
				partition = (rowBlockIndex-1)/numRowBlocks + 1;	
				break;
			}
			case COLUMN_BLOCK_WISE_N:
			{
				long numColBlocks = (long)Math.ceil(((double)PARTITION_SIZE)/_rlen/_bclen); 
				partition = (colBlockIndex-1)/numColBlocks + 1;
				break;
			}
			
			default: 
				throw new DMLRuntimeException("Unsupported partition format for distributed cache input: "+_pformat);
		}
		
		return _localFilePath.toString() + File.separator + partition;
	}
}

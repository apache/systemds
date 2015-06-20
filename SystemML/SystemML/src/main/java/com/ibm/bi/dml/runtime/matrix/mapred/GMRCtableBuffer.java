/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.io.MatrixWriter;
import com.ibm.bi.dml.runtime.matrix.data.CTableMap;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;


public class GMRCtableBuffer 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//buffer size is tradeoff between preaggregation and efficient hash probes
	//4k entries * ~64byte = 256KB (common L2 cache size)
	public static final int MAX_BUFFER_SIZE = 4096; 
	
	private HashMap<Byte, CTableMap> _mapBuffer = null;
	private HashMap<Byte, MatrixBlock> _blockBuffer = null;
	private CollectMultipleConvertedOutputs _collector = null;

	private byte[] _resultIndexes = null;
	private long[] _resultNonZeros = null;
	private byte[] _resultDimsUnknown = null;
	private long[] _resultMaxRowDims = null;
	private long[] _resultMaxColDims = null;
	
	
	public GMRCtableBuffer( CollectMultipleConvertedOutputs collector, boolean outputDimsKnown )
	{
		if ( outputDimsKnown )
			_blockBuffer = new HashMap<Byte, MatrixBlock>();
		else
			_mapBuffer = new HashMap<Byte, CTableMap>();
		_collector = collector;
	}
	
	/**
	 * 
	 * @param resultIndexes
	 * @param resultsNonZeros
	 * @param resultDimsUnknown
	 * @param resultsMaxRowDims
	 * @param resultsMaxColDims
	 */
	public void setMetadataReferences(byte[] resultIndexes, long[] resultsNonZeros, byte[] resultDimsUnknown, long[] resultsMaxRowDims, long[] resultsMaxColDims)
	{
		_resultIndexes = resultIndexes;
		_resultNonZeros = resultsNonZeros;
		_resultDimsUnknown = resultDimsUnknown;
		_resultMaxRowDims = resultsMaxRowDims;
		_resultMaxColDims = resultsMaxColDims;
	}

	/**
	 * 
	 * @return
	 */
	public int getBufferSize()
	{
		if ( _mapBuffer != null ) {
			int ret = 0;
			for( Entry<Byte, CTableMap> ctable : _mapBuffer.entrySet() )
				ret += ctable.getValue().size();
			return ret;
		}
		else if ( _blockBuffer != null ) {
			int ret = 0;
			for( Entry<Byte, MatrixBlock> ctable: _blockBuffer.entrySet()) {
				ctable.getValue().recomputeNonZeros();
				ret += MatrixBlock.estimateSizeInMemory(
						ctable.getValue().getNumRows(), 
						ctable.getValue().getNumColumns(), 
						((double)ctable.getValue().getNonZeros()/ctable.getValue().getNumRows())*ctable.getValue().getNumColumns());
			}
			return ret;
		}
		else {
			return 0;
		}
	}
	
	/**
	 * 
	 * @return
	 */
	public HashMap<Byte, CTableMap> getMapBuffer()
	{
		return _mapBuffer;
	}
	
	public HashMap<Byte, MatrixBlock> getBlockBuffer() 
	{
		return _blockBuffer;
	}
	
	/**
	 * 
	 * @param reporter
	 * @throws RuntimeException
	 */
	@SuppressWarnings("deprecation")
	public void flushBuffer( Reporter reporter ) 
		throws RuntimeException 
	{
		try
		{
			if ( _mapBuffer != null ) {
				MatrixIndexes key=null;//new MatrixIndexes();
				MatrixCell value=new MatrixCell();
				for(Entry<Byte, CTableMap> ctable: _mapBuffer.entrySet())
				{
					ArrayList<Integer> resultIDs = ReduceBase.getOutputIndexes(ctable.getKey(), _resultIndexes);
					CTableMap resultMap = ctable.getValue();
					
					//maintain result dims and nonzeros
					for(Integer i: resultIDs) {
						_resultNonZeros[i] += resultMap.size();
						if( _resultDimsUnknown[i] == (byte) 1 ) {
							_resultMaxRowDims[i] = Math.max( resultMap.getMaxRow(), _resultMaxRowDims[i]);
							_resultMaxColDims[i] = Math.max( resultMap.getMaxColumn(), _resultMaxColDims[i]);
						}
					}
					
					//output result data 
					for(Entry<MatrixIndexes, Double> e: resultMap.entrySet()) {
						key = e.getKey();
						value.setValue(e.getValue());
						for(Integer i: resultIDs) {
							_collector.collectOutput(key, value, i, reporter);
						}
					}
				}
			}
			else if ( _blockBuffer != null ) {
				MatrixIndexes key=new MatrixIndexes(1,1);
				//DataConverter.writeBinaryBlockMatrixToHDFS(path, job, mat, mc.get_rows(), mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block(), replication);
				for(Entry<Byte, MatrixBlock> ctable: _blockBuffer.entrySet())
				{
					ArrayList<Integer> resultIDs=ReduceBase.getOutputIndexes(ctable.getKey(), _resultIndexes);
					MatrixBlock outBlock = ctable.getValue();
					outBlock.recomputeNonZeros();
					
					// TODO: change hard coding of 1000
					int brlen = 1000, bclen = 1000;
					int rlen = outBlock.getNumRows();
					int clen = outBlock.getNumColumns();
					
					// final output matrix is smaller than a single block
					if(rlen <= brlen && clen <= brlen ) {
						key = new MatrixIndexes(1,1);
						for(Integer i: resultIDs)
						{
							_collector.collectOutput(key, outBlock, i, reporter);
							_resultNonZeros[i]+= outBlock.getNonZeros();
						}
					}
					
					else {
						//Following code is similar to that in DataConverter.DataConverter.writeBinaryBlockMatrixToHDFS
						//initialize blocks for reuse (at most 4 different blocks required)
						MatrixBlock[] blocks = MatrixWriter.createMatrixBlocksForReuse(rlen, clen, brlen, bclen, true, outBlock.getNonZeros());  
						
						//create and write subblocks of matrix
						for(int blockRow = 0; blockRow < (int)Math.ceil(rlen/(double)brlen); blockRow++) {
							for(int blockCol = 0; blockCol < (int)Math.ceil(clen/(double)bclen); blockCol++)
							{
								int maxRow = (blockRow*brlen + brlen < rlen) ? brlen : rlen - blockRow*brlen;
								int maxCol = (blockCol*bclen + bclen < clen) ? bclen : clen - blockCol*bclen;
						
								int row_offset = blockRow*brlen;
								int col_offset = blockCol*bclen;
								
								//get reuse matrix block
								MatrixBlock block = MatrixWriter.getMatrixBlockForReuse(blocks, maxRow, maxCol, brlen, bclen);
			
								//copy submatrix to block
								outBlock.sliceOperations( row_offset+1, row_offset+maxRow, 
										             col_offset+1, col_offset+maxCol, 
										             block );
								
								// TODO: skip empty "block"
								
								//append block to sequence file
								key.setIndexes(blockRow+1, blockCol+1);
								for(Integer i: resultIDs)
								{
									_collector.collectOutput(key, block, i, reporter);
									_resultNonZeros[i]+= block.getNonZeros();
								}
								
								//reset block for later reuse
								block.reset();
							}
						}
					}
				}
			}
			else {
				throw new DMLRuntimeException("Unexpected.. both ctable buffers are empty.");
			}
		}
		catch(Exception ex)
		{
			throw new RuntimeException("Failed to flush ctable buffer.", ex);
		}
		//remove existing partial ctables
		if (_mapBuffer != null ) 
			_mapBuffer.clear();
		else 
			_blockBuffer.clear();
	}
}

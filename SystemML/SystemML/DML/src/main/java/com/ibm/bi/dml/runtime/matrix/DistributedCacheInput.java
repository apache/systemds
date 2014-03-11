package com.ibm.bi.dml.runtime.matrix;

import java.util.ArrayList;

import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import com.ibm.bi.dml.runtime.util.DataConverter;

public class DistributedCacheInput {
	
	private Path localFilePath;
	private long numRows, numCols;
	
	private boolean isPartitioned;
	private PDataPartitionFormat pFormat;
	private int pSize;
	
	IndexedMatrixValue[][] dataBlocks;
	
	
	private void init() {
		localFilePath = null;
		numRows = numCols = -1;
		isPartitioned = false;
		pFormat = PDataPartitionFormat.NONE;
		pSize = -1;
		dataBlocks = null;
	}
	
	public DistributedCacheInput() {
		init();
	}
	
	public DistributedCacheInput(Path p, long rows, long cols) {
		init();
		localFilePath = p;
		numRows = rows;
		numCols = cols;
	}
	
	public DistributedCacheInput(Path p, long rows, long cols, boolean partitioned, PDataPartitionFormat pfmt, int psize) {
		init();
		localFilePath = p;
		numRows = rows;
		numCols = cols;
		isPartitioned = partitioned;
		pFormat = pfmt;
		pSize = psize;
	}
	
	public void reset() {
		init();
	}

	public IndexedMatrixValue getDataBlock(long rowBlockIndex,
			long colBlockIndex, int rowBlockSize, int colBlockSize)
			throws DMLRuntimeException {
		if (dataBlocks == null)
			readDataBlocks(rowBlockSize, colBlockSize);
		return dataBlocks[(int) rowBlockIndex - 1][(int) colBlockIndex - 1];
	}
	
	private void readDataBlocks(int rowBlockSize, int colBlockSize)
			throws DMLRuntimeException {
		if (!isPartitioned) // entire vector
		{
			try {
				ArrayList<IndexedMatrixValue> tmp = DataConverter
						.readMatrixBlocksFromHDFS(localFilePath.toString(),
								InputInfo.BinaryBlockInputInfo,
								numRows, // use rlens
								numCols, rowBlockSize, colBlockSize,
								!MRBaseForCommonInstructions.isJobLocal);

				int rowBlocks = (int) Math.ceil(numRows / (double) rowBlockSize);
				int colBlocks = (int) Math.ceil(numCols / (double) colBlockSize);

				dataBlocks = new IndexedMatrixValue[rowBlocks][colBlocks];

				for (IndexedMatrixValue val : tmp) {
					MatrixIndexes idx = val.getIndexes();
					dataBlocks[(int) idx.getRowIndex() - 1][(int) idx.getColumnIndex() - 1] = val;
				}
			} catch (Exception ex) {
				throw new DMLRuntimeException(ex);
			}

		} else // partitioned input
		{
			throw new DMLRuntimeException(
					"Partitioned inputs are not supported.");
			/*
			 * MatrixValue mv = getPartitionFromDistributedCache(input,
			 * distCache_index, rowBlockIndex, colBlockIndex, rowBlockSize,
			 * colBlockSize);
			 * 
			 * int part_rl, st, end; part_rl = (int)
			 * ((rowBlockIndex-1)*rowBlockSize
			 * /inputPartitionSizes[input])*inputPartitionSizes[input]; st =
			 * (int) ((rowBlockIndex-1)*rowBlockSize - part_rl); end = (int)
			 * Math.min(rowBlockIndex*rowBlockSize,
			 * distCacheNumRows[distCache_index])-part_rl-1;
			 * 
			 * MatrixBlock mb = new MatrixBlock( (int)Math.min(rowBlockSize,
			 * (distCacheNumRows
			 * [distCache_index]-(rowBlockIndex-1)*rowBlockSize)),
			 * (int)Math.min(colBlockSize,
			 * (distCacheNumColumns[distCache_index]-
			 * (colBlockIndex-1)*colBlockSize)), false); mb = (MatrixBlock)
			 * ((MatrixBlockDSM)mv).sliceOperations(st+1, end+1, 1, 1, mb); ret
			 * = new IndexedMatrixValue(new
			 * MatrixIndexes(rowBlockIndex,colBlockIndex),mb);
			 */
		}
	}
		
/*		private static MatrixValue getPartitionFromDistributedCache(byte input, int distCache_index, long rowBlockIndex, long colBlockIndex, int rowBlockSize, int colBlockSize) 
			throws DMLRuntimeException 
		{	
			IndexedMatrixValue imv = distCacheValues.get(input);

			int partID = computePartitionID(rowBlockIndex, colBlockIndex, rowBlockSize, colBlockSize, inputPartitionFormats[input], inputPartitionSizes[input]);
			//int cachedPartID = 
			boolean readNewPartition = true;
			if ( imv != null ) {
				MatrixIndexes partIdx = imv.getIndexes();
				
				// cached partition's range (from distCacheValues)
				//int part_st = (int) (partID-1)*inputPartitionSizes[input];
				//int part_end = part_st + (int) Math.min(partID*inputPartitionSizes[input], distCacheNumRows[distCache_index]-part_st)-1;
				
				// requested range
				//int req_st = (int) ((rowBlockIndex-1)*DMLTranslator.DMLBlockSize);
				//int req_end = (int) Math.min(rowBlockIndex*DMLTranslator.DMLBlockSize, distCacheNumRows[distCache_index])-1;
				//if ( req_st < req_end && req_st >= part_st && req_end <= part_end ) {
				//	// requested range can be served from distCacheValues, and no need to load a new partition
				//	readNewPartition = false; 
				//}
				
				int cachedPartID = (int) partIdx.getRowIndex();
				if(partID == cachedPartID || inputPartitionFlags[input] == false)
					readNewPartition = false;
				//System.out.println("reqIndex ["+rowBlockIndex+","+colBlockIndex+"] reqRange [" + req_st + "," + req_end +"]  partRange [" + part_st + "," + part_end + "] ... cachedPart " + cachedPartID + " reqPartID " + partID + " --> " + (readNewPartition ? "ReadNew" : "UseCached"));
			}
			if(imv == null || readNewPartition) {
				MatrixValue data = null;
				MatrixIndexes idx = null;

				// If the input data is not partitioned, read the entire matrix from HDFS.
				// Otherwise, read the required partition
				if(inputPartitionFlags[input] == false) {
					try {
						data = DataConverter.readMatrixFromHDFS(
				    			distCacheFiles[distCache_index].toString(), InputInfo.BinaryBlockInputInfo, 
				    			distCacheNumRows[distCache_index], // use rlens 
				    			distCacheNumColumns[distCache_index], 
				    			rowBlockSize, 
				    			colBlockSize, 1.0, !isJobLocal );
					} catch (IOException e) {
						throw new DMLRuntimeException(e);
					}
					idx = new MatrixIndexes(1,1);
				}
				else { 
					data = DataConverter.readPartitionFromDistCache(
							distCacheFiles[distCache_index].toString(), 
							true, 
							distCacheNumRows[distCache_index], distCacheNumColumns[distCache_index],
							rowBlockSize, colBlockSize,
							partID, inputPartitionSizes[input]);
					idx = new MatrixIndexes(partID,1);
				}
				//System.out.println(".... READ " + idx.toString());
				imv = new IndexedMatrixValue(idx, data);
				distCacheValues.put(input, imv);
			}
			
			return imv.getValue();
		}
		
		private static int computePartitionID(long rowBlockIndex, long colBlockIndex, int rowBlockSize, int colBlockSize, PDataPartitionFormat pformat, int psize) throws DMLRuntimeException {
			int pfile = -1; // partition file ID
			switch(pformat) {
			case NONE:
				return -1;
			case ROW_BLOCK_WISE_N:
				pfile = (int) (((rowBlockIndex-1)*rowBlockSize)/psize) + 1;
				break;
			
			case COLUMN_BLOCK_WISE_N:
				pfile = (int) (((colBlockIndex-1)*colBlockSize)/psize) + 1;
				break;
			
			default:
				throw new DMLRuntimeException("Unexpected partitioning format (" + pformat + ") in readPartitionFromDistCache");
			}
			
			return pfile;
		}*/

	
	public Path getLocalFilePath() {
		return localFilePath;
	}
	public void setLocalFilePath(Path localFilePath) {
		this.localFilePath = localFilePath;
	}
	public long getNumRows() {
		return numRows;
	}
	public void setNumRows(long numRows) {
		this.numRows = numRows;
	}
	public long getNumCols() {
		return numCols;
	}
	public void setNumCols(long numCols) {
		this.numCols = numCols;
	}
	public boolean isPartitioned() {
		return isPartitioned;
	}
	public void setPartitioned(boolean isPartitioned) {
		this.isPartitioned = isPartitioned;
	}
	public PDataPartitionFormat getpFormat() {
		return pFormat;
	}
	public void setpFormat(PDataPartitionFormat pFormat) {
		this.pFormat = pFormat;
	}
	public int getpSize() {
		return pSize;
	}
	public void setpSize(int pSize) {
		this.pSize = pSize;
	}
	
}

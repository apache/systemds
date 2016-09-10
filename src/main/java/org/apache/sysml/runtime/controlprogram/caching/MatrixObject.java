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

import java.io.IOException;
import java.lang.ref.SoftReference;

import org.apache.commons.lang.mutable.MutableBoolean;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.data.RDDObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.MetaData;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNN;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.runtime.util.MapReduceTool;


/**
 * Represents a matrix in control program. This class contains method to read
 * matrices from HDFS and convert them to a specific format/representation. It
 * is also able to write several formats/representation of matrices to HDFS.

 * IMPORTANT: Preserve one-to-one correspondence between {@link MatrixObject}
 * and {@link MatrixBlock} objects, for cache purposes.  Do not change a
 * {@link MatrixBlock} object without informing its {@link MatrixObject} object.
 * 
 */
public class MatrixObject extends CacheableData<MatrixBlock>
{
	private static final long serialVersionUID = 6374712373206495637L;
	
	public enum UpdateType {
		COPY,
		INPLACE,
		INPLACE_PINNED;
		public boolean isInPlace() {
			return (this != COPY);
		}
	}
	
	//additional matrix-specific flags
	private UpdateType _updateType = UpdateType.COPY; 
	
	//information relevant to partitioned matrices.
	private boolean _partitioned = false; //indicates if obj partitioned
	private PDataPartitionFormat _partitionFormat = null; //indicates how obj partitioned
	private int _partitionSize = -1; //indicates n for BLOCKWISE_N
	private String _partitionCacheName = null; //name of cache block
	private MatrixBlock _partitionInMemory = null;

	/**
	 * Constructor that takes only the HDFS filename.
	 */
	public MatrixObject (ValueType vt, String file) {
		this (vt, file, null); //HDFS file path
	}
	
	/**
	 * Constructor that takes both HDFS filename and associated metadata.
	 */
	public MatrixObject( ValueType vt, String file, MetaData mtd ) {
		super (DataType.MATRIX, vt);
		_metaData = mtd; 
		_hdfsFileName = file;		
		_cache = null;
		_data = null;
	}
	
	/**
	 * Copy constructor that copies meta data but NO data.
	 * 
	 * @param mo
	 */
	public MatrixObject( MatrixObject mo )
	{
		//base copy constructor
		super(mo);

		MatrixFormatMetaData metaOld = (MatrixFormatMetaData)mo.getMetaData();
		_metaData = new MatrixFormatMetaData(new MatrixCharacteristics(metaOld.getMatrixCharacteristics()),
				                             metaOld.getOutputInfo(), metaOld.getInputInfo());
		
		_updateType = mo._updateType;
		_partitioned = mo._partitioned;
		_partitionFormat = mo._partitionFormat;
		_partitionSize = mo._partitionSize;
		_partitionCacheName = mo._partitionCacheName;
	}
	

	/**
	 * 
	 * @param flag
	 */
	public void setUpdateType(UpdateType flag) {
		_updateType = flag;
	}
	
	/**
	 * 
	 * @return
	 */
	public UpdateType getUpdateType() {
		return _updateType;
	}
	
	@Override
	public void updateMatrixCharacteristics (MatrixCharacteristics mc) {
		((MatrixDimensionsMetaData)_metaData).setMatrixCharacteristics( mc );
	}

	/**
	 * Make the matrix metadata consistent with the in-memory matrix data
	 * @throws CacheException 
	 */
	@Override
	public void refreshMetaData() 
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
	
	/**
	 * 
	 * @return
	 */
	public long getNumRows() {
		MatrixCharacteristics mc = getMatrixCharacteristics();
		return mc.getRows ();
	}

	/**
	 * 
	 * @return
	 */
	public long getNumColumns() {
		MatrixCharacteristics mc = getMatrixCharacteristics();
		return mc.getCols ();
	}

	/**
	 * 
	 * @return
	 */
	public long getNumRowsPerBlock() {
		MatrixCharacteristics mc = getMatrixCharacteristics();
		return mc.getRowsPerBlock();
	}
	
	/**
	 * 
	 * @return
	 */
	public long getNumColumnsPerBlock() {
		MatrixCharacteristics mc = getMatrixCharacteristics();
		return mc.getColsPerBlock();
	}
	
	/**
	 * 
	 * @return
	 */
	public long getNnz() {
		MatrixCharacteristics mc = getMatrixCharacteristics();
		return mc.getNonZeros();
	}
	
	/**
	 * 
	 * @return
	 */
	public double getSparsity() {
		MatrixCharacteristics mc = getMatrixCharacteristics();		
		return ((double)mc.getNonZeros())/mc.getRows()/mc.getCols();
	}
	
	@Override
	protected void clearReusableData() {
		if(DMLScript.REUSE_NONZEROED_OUTPUT) {
			if(_data == null)
				getCache();
			if( _data != null && !_data.isVector() )
				LibMatrixDNN.cacheReuseableData(_data.getDenseBlock());
		}
	}
	
	// *********************************************
	// ***                                       ***
	// ***       HIGH-LEVEL PUBLIC METHODS       ***
	// ***     FOR PARTITIONED MATRIX ACCESS     ***
	// ***   (all other methods still usable)    ***
	// ***                                       ***
	// *********************************************
	
	/**
	 * @param n 
	 * 
	 */
	public void setPartitioned( PDataPartitionFormat format, int n )
	{
		_partitioned = true;
		_partitionFormat = format;
		_partitionSize = n;
	}
	

	public void unsetPartitioned() 
	{
		_partitioned = false;
		_partitionFormat = null;
		_partitionSize = -1;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isPartitioned()
	{
		return _partitioned;
	}
	
	public PDataPartitionFormat getPartitionFormat()
	{
		return _partitionFormat;
	}
	
	public int getPartitionSize()
	{
		return _partitionSize;
	}
	
	public synchronized void setInMemoryPartition(MatrixBlock block)
	{
		_partitionInMemory = block;
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
		if( LOG.isTraceEnabled() )
			LOG.trace("Acquire partition "+getVarName()+" "+pred);
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		if ( !_partitioned )
			throw new CacheException ("MatrixObject not available to indexed read.");
		
		//return static partition of set from outside of the program
		if( _partitionInMemory != null )
			return _partitionInMemory;
		
		MatrixBlock mb = null;
		
		try
		{
			boolean blockwise = (_partitionFormat==PDataPartitionFormat.ROW_BLOCK_WISE || _partitionFormat==PDataPartitionFormat.COLUMN_BLOCK_WISE);
			
			//preparations for block wise access
			MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;
			MatrixCharacteristics mc = iimd.getMatrixCharacteristics();
			int brlen = mc.getRowsPerBlock();
			int bclen = mc.getColsPerBlock();
			
			//get filename depending on format
			String fname = getPartitionFileName( pred, brlen, bclen );
			
			//probe cache
			if( blockwise && _partitionCacheName != null && _partitionCacheName.equals(fname) )
			{
				mb = _cache.get(); //try getting block from cache
			}
			
			if( mb == null ) //block not in cache
			{
				//get rows and cols
				long rows = -1;
				long cols = -1;
				switch( _partitionFormat )
				{
					case ROW_WISE:
						rows = 1;
						cols = mc.getCols();
						break;
					case ROW_BLOCK_WISE: 
						rows = brlen;
						cols = mc.getCols();
						break;
					case COLUMN_WISE:
						rows = mc.getRows();
						cols = 1;
						break;
					case COLUMN_BLOCK_WISE: 
						rows = mc.getRows();
						cols = bclen;
						break;
					default:
						throw new CacheException("Unsupported partition format: "+_partitionFormat);
				}
				
				
				//read the 
				if( MapReduceTool.existsFileOnHDFS(fname) )
					mb = readBlobFromHDFS( fname, rows, cols );
				else
				{
					mb = new MatrixBlock((int)rows, (int)cols, true);
					LOG.warn("Reading empty matrix partition "+fname);
				}
			}
			
			//post processing
			if( blockwise )
			{
				//put block into cache
				_partitionCacheName = fname;
				_cache = new SoftReference<MatrixBlock>(mb);
				
				if( _partitionFormat == PDataPartitionFormat.ROW_BLOCK_WISE )
				{
					int rix = (int)((pred.rowStart-1)%brlen);
					mb = mb.sliceOperations(rix, rix, (int)(pred.colStart-1), (int)(pred.colEnd-1), new MatrixBlock());
				}
				if( _partitionFormat == PDataPartitionFormat.COLUMN_BLOCK_WISE )
				{
					int cix = (int)((pred.colStart-1)%bclen);
					mb = mb.sliceOperations((int)(pred.rowStart-1), (int)(pred.rowEnd-1), cix, cix, new MatrixBlock());
				}
			}
			
			//NOTE: currently no special treatment of non-existing partitions necessary 
			//      because empty blocks are written anyway
		}
		catch(Exception ex)
		{
			throw new CacheException(ex);
		}
		
		if( DMLScript.STATISTICS ){
			long t1 = System.nanoTime();
			CacheStatistics.incrementAcquireRTime(t1-t0);
		}
		
		return mb;
	}
	
	
	/**
	 * 
	 * @param pred
	 * @return
	 * @throws CacheException 
	 */
	public String getPartitionFileName( IndexRange pred, int brlen, int bclen ) 
		throws CacheException
	{
		if ( !_partitioned )
			throw new CacheException ("MatrixObject not available to indexed read.");
		
		StringBuilder sb = new StringBuilder();
		sb.append(_hdfsFileName);
		
		switch( _partitionFormat )
		{
			case ROW_WISE:
				sb.append(Lop.FILE_SEPARATOR);
				sb.append(pred.rowStart); 
				break;
			case ROW_BLOCK_WISE:
				sb.append(Lop.FILE_SEPARATOR);
				sb.append((pred.rowStart-1)/brlen+1);
				break;
			case COLUMN_WISE:
				sb.append(Lop.FILE_SEPARATOR);
				sb.append(pred.colStart);
				break;
			case COLUMN_BLOCK_WISE:
				sb.append(Lop.FILE_SEPARATOR);
				sb.append((pred.colStart-1)/bclen+1);
				break;
			default:
				throw new CacheException ("MatrixObject not available to indexed read.");
		}

		return sb.toString();
	}	
	
	

	// *********************************************
	// ***                                       ***
	// ***      LOW-LEVEL PROTECTED METHODS      ***
	// ***         EXTEND CACHEABLE DATA         ***
	// ***     ONLY CALLED BY THE SUPERCLASS     ***
	// ***                                       ***
	// *********************************************
		
	@Override
	protected boolean isBelowCachingThreshold() {
		return super.isBelowCachingThreshold()
			|| getUpdateType() == UpdateType.INPLACE_PINNED;
	}
	
	@Override
	protected MatrixBlock readBlobFromCache(String fname) throws IOException {
		return (MatrixBlock)LazyWriteBuffer.readBlock(fname, true);
	}
	

	@Override
	protected MatrixBlock readBlobFromHDFS(String fname, long rlen, long clen)
		throws IOException
	{
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;
		MatrixCharacteristics mc = iimd.getMatrixCharacteristics();
		long begin = 0;
		
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Reading matrix from HDFS...  " + getVarName() + "  Path: " + fname 
					+ ", dimensions: [" + mc.getRows() + ", " + mc.getCols() + ", " + mc.getNonZeros() + "]");
			begin = System.currentTimeMillis();
		}
			
		double sparsity = ( mc.getNonZeros() >= 0 ? ((double)mc.getNonZeros())/(mc.getRows()*mc.getCols()) : 1.0d) ; 
		MatrixBlock newData = DataConverter.readMatrixFromHDFS(fname, iimd.getInputInfo(), rlen, clen,
				mc.getRowsPerBlock(), mc.getColsPerBlock(), sparsity, getFileFormatProperties());
		
		//sanity check correct output
		if( newData == null )
			throw new IOException("Unable to load matrix from file: "+fname);
		
		if( LOG.isTraceEnabled() )
			LOG.trace("Reading Completed: " + (System.currentTimeMillis()-begin) + " msec.");
		
		return newData;
	}
	
	/**
	 * 
	 * @param rdd
	 * @return
	 * @throws IOException 
	 */
	@Override
	protected MatrixBlock readBlobFromRDD(RDDObject rdd, MutableBoolean writeStatus) 
		throws IOException
	{
		//note: the read of a matrix block from an RDD might trigger
		//lazy evaluation of pending transformations.
		RDDObject lrdd = rdd;

		//prepare return status (by default only collect)
		writeStatus.setValue(false);
		
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;
		MatrixCharacteristics mc = iimd.getMatrixCharacteristics();
		InputInfo ii = iimd.getInputInfo();
		MatrixBlock mb = null;
		try 
		{
			//prevent unnecessary collect through rdd checkpoint
			if( rdd.allowsShortCircuitCollect() ) {
				lrdd = (RDDObject)rdd.getLineageChilds().get(0);
			}
			
			//obtain matrix block from RDD
			int rlen = (int)mc.getRows();
			int clen = (int)mc.getCols();
			int brlen = (int)mc.getRowsPerBlock();
			int bclen = (int)mc.getColsPerBlock();
			long nnz = mc.getNonZeros();
			
			//guarded rdd collect 
			if( ii == InputInfo.BinaryBlockInputInfo && //guarded collect not for binary cell
				!OptimizerUtils.checkSparkCollectMemoryBudget(rlen, clen, brlen, bclen, nnz, getPinnedSize()) ) {
				//write RDD to hdfs and read to prevent invalid collect mem consumption 
				//note: lazy, partition-at-a-time collect (toLocalIterator) was significantly slower
				if( !MapReduceTool.existsFileOnHDFS(_hdfsFileName) ) { //prevent overwrite existing file
					long newnnz = SparkExecutionContext.writeRDDtoHDFS(lrdd, _hdfsFileName, iimd.getOutputInfo());
					((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics().setNonZeros(newnnz);
					((RDDObject)rdd).setHDFSFile(true); //mark rdd as hdfs file (for restore)
					writeStatus.setValue(true);         //mark for no cache-write on read
				}
				mb = readBlobFromHDFS(_hdfsFileName);
			}
			else if( ii == InputInfo.BinaryCellInputInfo ) {
				//collect matrix block from binary block RDD
				mb = SparkExecutionContext.toMatrixBlock(lrdd, rlen, clen, nnz);		
			}
			else {
				//collect matrix block from binary cell RDD
				mb = SparkExecutionContext.toMatrixBlock(lrdd, rlen, clen, brlen, bclen, nnz);	
			}
		}
		catch(DMLRuntimeException ex) {
			throw new IOException(ex);
		}
		
		//sanity check correct output
		if( mb == null ) {
			throw new IOException("Unable to load matrix from rdd: "+lrdd.getVarName());
		}
		
		return mb;
	}
	
	/**
	 * Writes in-memory matrix to HDFS in a specified format.
	 * 
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	@Override
	protected void writeBlobToHDFS(String fname, String ofmt, int rep, FileFormatProperties fprop)
		throws IOException, DMLRuntimeException
	{
		long begin = 0;
		if( LOG.isTraceEnabled() ){
			LOG.trace (" Writing matrix to HDFS...  " + getVarName() + "  Path: " + fname + ", Format: " +
						(ofmt != null ? ofmt : "inferred from metadata"));
			begin = System.currentTimeMillis();
		}
		
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;

		if (_data != null)
		{
			// Get the dimension information from the metadata stored within MatrixObject
			MatrixCharacteristics mc = iimd.getMatrixCharacteristics ();
			// Write the matrix to HDFS in requested format
			OutputInfo oinfo = (ofmt != null ? OutputInfo.stringToOutputInfo (ofmt) : 
					InputInfo.getMatchingOutputInfo (iimd.getInputInfo ()));
			
			// when outputFormat is binaryblock, make sure that matrixCharacteristics has correct blocking dimensions
			// note: this is only required if singlenode (due to binarycell default) 
			if ( oinfo == OutputInfo.BinaryBlockOutputInfo && DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE &&
				(mc.getRowsPerBlock() != ConfigurationManager.getBlocksize() || mc.getColsPerBlock() != ConfigurationManager.getBlocksize()) ) 
			{
				DataConverter.writeMatrixToHDFS(_data, fname, oinfo, new MatrixCharacteristics(mc.getRows(), mc.getCols(), ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize(), mc.getNonZeros()), rep, fprop);
			}
			else {
				DataConverter.writeMatrixToHDFS(_data, fname, oinfo, mc, rep, fprop);
			}

			if( LOG.isTraceEnabled() )
				LOG.trace("Writing matrix to HDFS ("+fname+") - COMPLETED... " + (System.currentTimeMillis()-begin) + " msec.");
		}
		else if( LOG.isTraceEnabled() ) {
			LOG.trace ("Writing matrix to HDFS ("+fname+") - NOTHING TO WRITE (_data == null).");
		}
		
		if( DMLScript.STATISTICS )
			CacheStatistics.incrementHDFSWrites();
	}
	
	@Override
	protected void writeBlobFromRDDtoHDFS(RDDObject rdd, String fname, String outputFormat) 
	    throws IOException, DMLRuntimeException
	{
	    //prepare output info
        MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;
	    OutputInfo oinfo = (outputFormat != null ? OutputInfo.stringToOutputInfo (outputFormat) 
                : InputInfo.getMatchingOutputInfo (iimd.getInputInfo ()));
	    
		//note: the write of an RDD to HDFS might trigger
		//lazy evaluation of pending transformations.				
		long newnnz = SparkExecutionContext.writeRDDtoHDFS(rdd, fname, oinfo);	
		((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics().setNonZeros(newnnz);
	}
}

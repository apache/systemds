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

package org.apache.sysds.runtime.controlprogram.caching;

import java.io.IOException;
import java.lang.ref.SoftReference;
import java.util.List;
import java.util.concurrent.Future;

import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.fed.InitFEDInstruction;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.instructions.spark.data.RDDObject;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.ReaderWriterFederated;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageRecomputeUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaData;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.utils.Explain;

/**
 * Represents a matrix in control program. This class contains method to read matrices from HDFS and convert them to a
 * specific format/representation. It is also able to write several formats/representation of matrices to HDFS.
 * 
 * IMPORTANT: Preserve one-to-one correspondence between {@link MatrixObject} and {@link MatrixBlock} objects, for cache
 * purposes. Do not change a {@link MatrixBlock} object without informing its {@link MatrixObject} object.
 * 
 */
public class MatrixObject extends CacheableData<MatrixBlock> {
	private static final long serialVersionUID = 6374712373206495637L;

	public enum UpdateType {
		COPY, INPLACE, INPLACE_PINNED;

		public boolean isInPlace() {
			return(this != COPY);
		}
	}

	// additional matrix-specific flags
	private UpdateType _updateType = UpdateType.COPY;
	private boolean _diag = false;
	private boolean _markForLinCache = false;

	// information relevant to partitioned matrices.
	private boolean _partitioned = false; // indicates if obj partitioned
	private PDataPartitionFormat _partitionFormat = null; // indicates how obj partitioned
	private int _partitionSize = -1; // indicates n for BLOCKWISE_N
	private String _partitionCacheName = null; // name of cache block
	private MatrixBlock _partitionInMemory = null;

	/**
	 * Constructor that takes the value type and the HDFS filename.
	 * 
	 * @param vt   value type
	 * @param file file name
	 */
	public MatrixObject(ValueType vt, String file) {
		this(vt, file, null, null); // HDFS file path
	}

	/**
	 * Constructor that takes the value type, HDFS filename and associated metadata.
	 * 
	 * @param vt   value type
	 * @param file file name
	 * @param mtd  metadata
	 */
	public MatrixObject(ValueType vt, String file, MetaData mtd) {
		this(vt, file, mtd, null);
	}

	/**
	 * Constructor that takes the value type, HDFS filename and associated metadata and a MatrixBlock used for creation
	 * after serialization
	 *
	 * @param vt   value type
	 * @param file file name
	 * @param mtd  metadata
	 * @param data matrix block data
	 */
	public MatrixObject(ValueType vt, String file, MetaData mtd, MatrixBlock data) {
		super(DataType.MATRIX, vt);
		_metaData = mtd;
		_hdfsFileName = file;
		_cache = null;
		if(data != null) {
			acquireModify(data);
			release();
		}
	}

	/**
	 * Copy constructor that copies meta data but NO data.
	 * 
	 * @param mo matrix object
	 */
	public MatrixObject(MatrixObject mo) {
		// base copy constructor
		super(mo);

		MetaDataFormat metaOld = (MetaDataFormat) mo.getMetaData();
		_metaData = new MetaDataFormat(new MatrixCharacteristics(metaOld.getDataCharacteristics()),
			metaOld.getFileFormat());
		_updateType = mo._updateType;
		_diag = mo._diag;
		_partitioned = mo._partitioned;
		_partitionFormat = mo._partitionFormat;
		_partitionSize = mo._partitionSize;
		_partitionCacheName = mo._partitionCacheName;
		_markForLinCache = mo._markForLinCache;
	}

	public void setUpdateType(UpdateType flag) {
		_updateType = flag;
	}

	public UpdateType getUpdateType() {
		return _updateType;
	}

	public boolean isDiag() {
		return _diag;
	}

	public void setDiag(boolean diag) {
		_diag = diag;
	}

	public void setMarkForLinCache(boolean mark) {
		_markForLinCache = mark;
	}

	public boolean isMarked() {
		return _markForLinCache;
	}

	@Override
	public void updateDataCharacteristics(DataCharacteristics dc) {
		_metaData.getDataCharacteristics().set(dc);
	}

	/**
	 * Make the matrix metadata consistent with the in-memory matrix data
	 */
	@Override
	public void refreshMetaData() {
		if(_data == null || _metaData == null) // refresh only for existing data
			throw new DMLRuntimeException("Cannot refresh meta data because there is no data or meta data. ");
		// we need to throw an exception, otherwise input/output format cannot be inferred

		DataCharacteristics mc = _metaData.getDataCharacteristics();
		mc.setDimension(_data.getNumRows(), _data.getNumColumns());
		mc.setNonZeros(_data.getNonZeros());
	}

	public int getBlocksize() {
		return getDataCharacteristics().getBlocksize();
	}

	public long getNnz() {
		return getDataCharacteristics().getNonZeros();
	}

	public double getSparsity() {
		return OptimizerUtils.getSparsity(getDataCharacteristics());
	}

	// *********************************************
	// *** ***
	// *** HIGH-LEVEL PUBLIC METHODS ***
	// *** FOR PARTITIONED MATRIX ACCESS ***
	// *** (all other methods still usable) ***
	// *** ***
	// *********************************************

	public void setPartitioned(PDataPartitionFormat format, int n) {
		_partitioned = true;
		_partitionFormat = format;
		_partitionSize = n;
	}

	public void unsetPartitioned() {
		_partitioned = false;
		_partitionFormat = null;
		_partitionSize = -1;
	}

	public boolean isPartitioned() {
		return _partitioned;
	}

	public PDataPartitionFormat getPartitionFormat() {
		return _partitionFormat;
	}

	public int getPartitionSize() {
		return _partitionSize;
	}

	public synchronized void setInMemoryPartition(MatrixBlock block) {
		_partitionInMemory = block;
	}

	/**
	 * NOTE: for reading matrix partitions, we could cache (in its real sense) the read block with soft references (no
	 * need for eviction, as partitioning only applied for read-only matrices). However, since we currently only support
	 * row- and column-wise partitioning caching is not applied yet. This could be changed once we also support
	 * column-block-wise and row-block-wise. Furthermore, as we reject to partition vectors and support only full row or
	 * column indexing, no metadata (apart from the partition flag) is required.
	 * 
	 * @param pred index range
	 * @return matrix block
	 */
	public synchronized MatrixBlock readMatrixPartition(IndexRange pred) {
		if(LOG.isTraceEnabled())
			LOG.trace("Acquire partition " + hashCode() + " " + pred);
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;

		if(!_partitioned)
			throw new DMLRuntimeException("MatrixObject not available to indexed read.");

		// return static partition of set from outside of the program
		if(_partitionInMemory != null)
			return _partitionInMemory;

		MatrixBlock mb = null;

		try {
			boolean blockwise = (_partitionFormat == PDataPartitionFormat.ROW_BLOCK_WISE ||
				_partitionFormat == PDataPartitionFormat.COLUMN_BLOCK_WISE);

			// preparations for block wise access
			MetaDataFormat iimd = (MetaDataFormat) _metaData;
			DataCharacteristics mc = iimd.getDataCharacteristics();
			int blen = mc.getBlocksize();

			// get filename depending on format
			String fname = getPartitionFileName(pred, blen);

			// probe cache
			if(blockwise && _partitionCacheName != null && _partitionCacheName.equals(fname)) {
				mb = _cache.get(); // try getting block from cache
			}

			if(mb == null) // block not in cache
			{
				// get rows and cols
				long rows = -1;
				long cols = -1;
				switch(_partitionFormat) {
					case ROW_WISE:
						rows = 1;
						cols = mc.getCols();
						break;
					case ROW_BLOCK_WISE:
						rows = blen;
						cols = mc.getCols();
						break;
					case ROW_BLOCK_WISE_N:
						rows = _partitionSize;
						cols = mc.getCols();
						break;
					case COLUMN_WISE:
						rows = mc.getRows();
						cols = 1;
						break;
					case COLUMN_BLOCK_WISE:
						rows = mc.getRows();
						cols = blen;
						break;
					case COLUMN_BLOCK_WISE_N:
						rows = mc.getRows();
						cols = _partitionSize;
						break;
					default:
						throw new DMLRuntimeException("Unsupported partition format: " + _partitionFormat);
				}

				// read the
				if(HDFSTool.existsFileOnHDFS(fname))
					mb = readBlobFromHDFS(fname, new long[] {rows, cols});
				else {
					mb = new MatrixBlock((int) rows, (int) cols, true);
					LOG.warn("Reading empty matrix partition " + fname);
				}
			}

			// post processing
			if(blockwise) {
				// put block into cache
				_partitionCacheName = fname;
				_cache = new SoftReference<>(mb);

				if(_partitionFormat == PDataPartitionFormat.ROW_BLOCK_WISE) {
					int rix = (int) ((pred.rowStart - 1) % blen);
					mb = mb.slice(rix, rix, (int) (pred.colStart - 1), (int) (pred.colEnd - 1), new MatrixBlock());
				}
				if(_partitionFormat == PDataPartitionFormat.COLUMN_BLOCK_WISE) {
					int cix = (int) ((pred.colStart - 1) % blen);
					mb = mb.slice((int) (pred.rowStart - 1), (int) (pred.rowEnd - 1), cix, cix, new MatrixBlock());
				}
			}

			// NOTE: currently no special treatment of non-existing partitions necessary
			// because empty blocks are written anyway
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}

		if(DMLScript.STATISTICS) {
			long t1 = System.nanoTime();
			CacheStatistics.incrementAcquireRTime(t1 - t0);
		}

		return mb;
	}

	public String getPartitionFileName(IndexRange pred, int blen) {
		if(!_partitioned)
			throw new DMLRuntimeException("MatrixObject not available to indexed read.");

		StringBuilder sb = new StringBuilder();
		sb.append(_hdfsFileName);

		switch(_partitionFormat) {
			case ROW_WISE:
				sb.append(Lop.FILE_SEPARATOR);
				sb.append(pred.rowStart);
				break;
			case ROW_BLOCK_WISE:
				sb.append(Lop.FILE_SEPARATOR);
				sb.append((pred.rowStart - 1) / blen + 1);
				break;
			case ROW_BLOCK_WISE_N:
				sb.append(Lop.FILE_SEPARATOR);
				sb.append((pred.rowStart - 1) / _partitionSize + 1);
				break;
			case COLUMN_WISE:
				sb.append(Lop.FILE_SEPARATOR);
				sb.append(pred.colStart);
				break;
			case COLUMN_BLOCK_WISE:
				sb.append(Lop.FILE_SEPARATOR);
				sb.append((pred.colStart - 1) / blen + 1);
				break;
			case COLUMN_BLOCK_WISE_N:
				sb.append(Lop.FILE_SEPARATOR);
				sb.append((pred.colStart - 1) / _partitionSize + 1);
				break;
			default:
				throw new DMLRuntimeException("MatrixObject not available to indexed read.");
		}

		return sb.toString();
	}

	// *********************************************
	// *** ***
	// *** LOW-LEVEL PROTECTED METHODS ***
	// *** EXTEND CACHEABLE DATA ***
	// *** ONLY CALLED BY THE SUPERCLASS ***
	// *** ***
	// *********************************************

	@Override
	protected boolean isBelowCachingThreshold() {
		return LazyWriteBuffer.getCacheBlockSize(_data) <= CACHING_THRESHOLD ||
			getUpdateType() == UpdateType.INPLACE_PINNED;
	}

	@Override
	protected MatrixBlock readBlobFromCache(String fname) throws IOException {
		MatrixBlock mb = null;
		if (OptimizerUtils.isUMMEnabled())
			mb = (MatrixBlock) UnifiedMemoryManager.readBlock(fname, true);
		else
			mb = (MatrixBlock) LazyWriteBuffer.readBlock(fname, true);
		return mb;
	}

	@Override
	protected MatrixBlock readBlobFromHDFS(String fname, long[] dims) throws IOException {
		long rlen = dims[0];
		long clen = dims[1];
		MetaDataFormat iimd = (MetaDataFormat) _metaData;
		DataCharacteristics mc = iimd.getDataCharacteristics();
		long begin = 0;

		if(LOG.isTraceEnabled()) {
			LOG.trace("Reading matrix from HDFS...  " + hashCode() + "  Path: " + fname + ", dimensions: ["
				+ mc.getRows() + ", " + mc.getCols() + ", " + mc.getNonZeros() + "]");
			begin = System.currentTimeMillis();
		}

		// If the file format is Federated use the federated reader.
		if(iimd.getFileFormat() == FileFormat.FEDERATED) {
			InitFEDInstruction.federateMatrix(this, ReaderWriterFederated.read(fname, mc));
		}

		// Read matrix and maintain meta data,
		// if the MatrixObject is federated there is nothing extra to read, and therefore only acquire read and release
		int blen = mc.getBlocksize() <= 0 ? ConfigurationManager.getBlocksize() : mc.getBlocksize();
		MatrixBlock newData = 
			isFederated() ? acquireReadAndRelease() : 
			DataConverter.readMatrixFromHDFS(fname, iimd.getFileFormat(),
				rlen, clen, blen, mc.getNonZeros(), getFileFormatProperties());

		if(iimd.getFileFormat() == FileFormat.CSV) {
			_metaData = _metaData instanceof MetaDataFormat ? new MetaDataFormat(newData.getDataCharacteristics(),
				iimd.getFileFormat()) : new MetaData(newData.getDataCharacteristics());
		}

		setHDFSFileExists(true);

		// sanity check correct output
		if(newData == null)
			throw new IOException("Unable to load matrix from file: " + fname);

		if(LOG.isTraceEnabled())
			LOG.trace("Reading Completed: " + (System.currentTimeMillis() - begin) + " msec.");

		return newData;
	}

	@Override
	protected MatrixBlock readBlobFromRDD(RDDObject rdd, MutableBoolean writeStatus) throws IOException {
		// note: the read of a matrix block from an RDD might trigger
		// lazy evaluation of pending transformations.
		RDDObject lrdd = rdd;

		// prepare return status (by default only collect)
		writeStatus.setValue(false);

		MetaDataFormat iimd = (MetaDataFormat) _metaData;
		DataCharacteristics mc = iimd.getDataCharacteristics();
		FileFormat fmt = iimd.getFileFormat();
		MatrixBlock mb = null;
		try {
			// prevent unnecessary collect through rdd checkpoint (unless lineage cached)
			if(rdd.allowsShortCircuitCollect()) {
				lrdd = (RDDObject) rdd.getLineageChilds().get(0);
			}

			// obtain matrix block from RDD
			int rlen = (int) mc.getRows();
			int clen = (int) mc.getCols();
			int blen = mc.getBlocksize() > 0 ? mc.getBlocksize() : ConfigurationManager.getBlocksize();
			long nnz = mc.getNonZerosBound();

			// guarded rdd collect
			if(fmt == FileFormat.BINARY && // guarded collect not for binary cell
				!OptimizerUtils.checkSparkCollectMemoryBudget(mc, getPinnedSize() + getBroadcastSize(), true)) {
				// write RDD to hdfs and read to prevent invalid collect mem consumption
				// note: lazy, partition-at-a-time collect (toLocalIterator) was significantly slower
				if(!HDFSTool.existsFileOnHDFS(_hdfsFileName)) { // prevent overwrite existing file
					long newnnz = SparkExecutionContext.writeMatrixRDDtoHDFS(lrdd, _hdfsFileName, iimd.getFileFormat());
					_metaData.getDataCharacteristics().setNonZeros(newnnz);
					rdd.setPending(false); // mark rdd as non-pending (for export)
					rdd.setHDFSFile(true); // mark rdd as hdfs file (for restore)
					writeStatus.setValue(true); // mark for no cache-write on read
					// note: the flag hdfsFile is actually not entirely correct because we still hold an rdd
					// reference to the input not to an rdd of the hdfs file but the resulting behavior is correct
				}
				mb = readBlobFromHDFS(_hdfsFileName);
			}
			else {
				// collect matrix block from binary cell RDD
				mb = SparkExecutionContext.toMatrixBlock(lrdd, rlen, clen, blen, nnz);
			}
		}
		catch(DMLRuntimeException ex) {
			throw new IOException(ex);
		}

		// sanity check correct output
		if(mb == null)
			throw new IOException("Unable to load matrix from rdd.");

		return mb;
	}
	

	@Override
	protected MatrixBlock readBlobFromStream(LocalTaskQueue<IndexedMatrixValue> stream) throws IOException {
		MatrixBlock ret = new MatrixBlock((int)getNumRows(), (int)getNumColumns(), false);
		IndexedMatrixValue tmp = null;
		try {
			int blen = getBlocksize(), lnnz = 0;
			while( (tmp = stream.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS ) {
				// compute row/column block offsets
				final int row_offset = (int) (tmp.getIndexes().getRowIndex() - 1) * blen;
				final int col_offset = (int) (tmp.getIndexes().getColumnIndex() - 1) * blen;

				// Add the values of this block into the output block.
				((MatrixBlock)tmp.getValue()).putInto(ret, row_offset, col_offset, true);
				
				// incremental maintenance nnz
				lnnz += tmp.getValue().getNonZeros();
			}
			ret.setNonZeros(lnnz);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		return ret;
	}

	@Override
	protected MatrixBlock readBlobFromFederated(FederationMap fedMap, long[] dims) throws IOException {
		// TODO sparse optimization
		List<Pair<FederatedRange, Future<FederatedResponse>>> readResponses = fedMap.requestFederatedData();
		try {
			if(fedMap.getType() == FType.PART)
				return FederationUtils.aggregateResponses(readResponses);
			else
				return FederationUtils.bindResponses(readResponses, dims);
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Federated matrix read failed.", e);
		}
	}

	/**
	 * Writes in-memory matrix to HDFS in a specified format.
	 */
	@Override
	protected void writeBlobToHDFS(String fname, String ofmt, int rep, FileFormatProperties fprop)
		throws IOException, DMLRuntimeException {
		long begin = 0;
		if(LOG.isTraceEnabled()) {
			LOG.trace(" Writing matrix to HDFS...  " + hashCode() + "  Path: " + fname + ", Format: "
				+ (ofmt != null ? ofmt : "inferred from metadata"));
			begin = System.currentTimeMillis();
		}

		if(this.isFederated() && FileFormat.safeValueOf(ofmt) == FileFormat.FEDERATED) {
			ReaderWriterFederated.write(fname, this._fedMapping);
		}
		else if(_data != null) {
			MetaDataFormat iimd = (MetaDataFormat) _metaData;
			// Get the dimension information from the metadata stored within MatrixObject
			DataCharacteristics mc = iimd.getDataCharacteristics();
			// Write the matrix to HDFS in requested format
			FileFormat fmt = (ofmt != null ? FileFormat.safeValueOf(ofmt) : iimd.getFileFormat());
			if( fmt == FileFormat.BINARY && fprop != null )
				mc = new MatrixCharacteristics(mc).setBlocksize(fprop.getBlocksize());
			DataConverter.writeMatrixToHDFS(_data, fname, fmt, mc, rep, fprop, _diag);

			if(LOG.isTraceEnabled())
				LOG.trace("Writing matrix to HDFS (" + fname + ") - COMPLETED... "
					+ (System.currentTimeMillis() - begin) + " msec.");
		}
		else if(LOG.isTraceEnabled()) {
			LOG.trace("Writing matrix to HDFS (" + fname + ") - NOTHING TO WRITE (_data == null).");
		}

		if(DMLScript.STATISTICS)
			CacheStatistics.incrementHDFSWrites();
	}

	@Override
	protected void writeBlobFromRDDtoHDFS(RDDObject rdd, String fname, String outputFormat)
		throws IOException, DMLRuntimeException {
		// prepare output info
		MetaDataFormat iimd = (MetaDataFormat) _metaData;
		FileFormat fmt = (outputFormat != null ? FileFormat.safeValueOf(outputFormat) : iimd.getFileFormat());

		// note: the write of an RDD to HDFS might trigger
		// lazy evaluation of pending transformations.
		long newnnz = SparkExecutionContext.writeMatrixRDDtoHDFS(rdd, fname, fmt);
		_metaData.getDataCharacteristics().setNonZeros(newnnz);
	}

	@Override
	protected MatrixBlock reconstructByLineage(LineageItem li) throws IOException {
		return ((MatrixObject) LineageRecomputeUtils
			.parseNComputeLineageTrace(Explain.explain(li)))
			.acquireReadAndRelease();
	}

	@Override 
	public boolean isCompressed(){
		if(super.isCompressed())
			return true;
		else if(_partitionInMemory instanceof CompressedMatrixBlock){
			setCompressedSize(_partitionInMemory.estimateSizeInMemory());
			return true;
		}
		else
			return false;
	}

	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder(super.toString());
		if(_partitioned){
			sb.append("\n");
			sb.append("partitioned:" + _partitionFormat );
			sb.append(", partitionSize: " + _partitionSize);
			sb.append(", partitionCacheName:" + _partitionCacheName);
			sb.append(_partitionInMemory);
			sb.append("\n");
		}
		return sb.toString();
	}
}

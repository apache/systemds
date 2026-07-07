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

package org.apache.sysds.runtime.io;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.ColumnReader;
import org.apache.parquet.column.impl.ColumnReadStoreImpl;
import org.apache.parquet.column.page.PageReadStore;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.io.api.Converter;
import org.apache.parquet.io.api.GroupConverter;
import org.apache.parquet.io.api.PrimitiveConverter;
import org.apache.parquet.schema.MessageType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.util.HDFSTool;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.delta.kernel.DataWriteContext;
import io.delta.kernel.Operation;
import io.delta.kernel.Scan;
import io.delta.kernel.Snapshot;
import io.delta.kernel.Table;
import io.delta.kernel.Transaction;
import io.delta.kernel.TransactionBuilder;
import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.data.ColumnarBatch;
import io.delta.kernel.data.FilteredColumnarBatch;
import io.delta.kernel.data.Row;
import io.delta.kernel.defaults.engine.DefaultEngine;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.internal.InternalScanFileUtils;
import io.delta.kernel.internal.ScanImpl;
import io.delta.kernel.internal.data.ScanStateRow;
import io.delta.kernel.internal.util.Utils;
import io.delta.kernel.types.BooleanType;
import io.delta.kernel.types.ByteType;
import io.delta.kernel.types.DataType;
import io.delta.kernel.types.DoubleType;
import io.delta.kernel.types.FloatType;
import io.delta.kernel.types.IntegerType;
import io.delta.kernel.types.LongType;
import io.delta.kernel.types.ShortType;
import io.delta.kernel.types.StringType;
import io.delta.kernel.types.StructType;
import io.delta.kernel.utils.CloseableIterable;
import io.delta.kernel.utils.CloseableIterator;
import io.delta.kernel.utils.DataFileStatus;
import io.delta.kernel.utils.FileStatus;

/**
 * Shared helpers for the native (Spark-free) Delta Lake read/write paths used
 * by both the matrix and frame readers/writers. Centralizes engine creation,
 * path qualification, the scan loop (snapshot -&gt; data files -&gt; logical
 * columnar batches, honoring deletion vectors), and the write transaction
 * (logical data -&gt; parquet -&gt; commit).
 */
public class DeltaKernelUtils {

	private static final Log LOG = LogFactory.getLog(DeltaKernelUtils.class.getName());

	private static final String ENGINE_INFO = "Apache SystemDS";

	/** Reused thread-safe JSON reader for the per-file Delta stats (numRecords). */
	private static final ObjectMapper JSON_MAPPER = new ObjectMapper();

	/** Delta Kernel config key: number of rows per parquet read batch, overridable via
	 * {@link org.apache.sysds.conf.DMLConfig#DELTA_READER_BATCH_SIZE}. */
	private static final String CONF_READER_BATCH_SIZE = "delta.kernel.default.parquet.reader.batch-size";
	/** Delta Kernel config key: target size (bytes) at which the writer rolls a new data file, overridable via
	 * {@link org.apache.sysds.conf.DMLConfig#DELTA_WRITER_TARGET_FILE_SIZE}. */
	private static final String CONF_WRITER_TARGET_FILE_SIZE = "delta.kernel.default.parquet.writer.targetMaxFileSize";

	/** Internal Delta column type codes shared by the matrix and frame readers to
	 * dispatch boxing-free primitive column access. */
	public static final int T_DOUBLE  = 0;
	public static final int T_FLOAT   = 1;
	public static final int T_LONG    = 2;
	public static final int T_INT     = 3;
	public static final int T_SHORT   = 4;
	public static final int T_BYTE    = 5;
	public static final int T_BOOLEAN = 6;
	public static final int T_STRING  = 7;

	//derived configuration cached to avoid copying the (large) base conf on every
	//engine creation (createEngine is called once per data file in parallel reads);
	//rebuilt whenever the base conf or the relevant SystemDS settings change.
	private static Configuration cachedConf;
	private static Configuration cachedConfBase;
	private static int cachedBatchSize;
	private static long cachedTargetFileSize;

	private DeltaKernelUtils() {}

	/**
	 * Consumes a whole columnar batch. {@code selected} is {@code null} when all
	 * {@code size} rows are live; otherwise {@code selected[r]} indicates whether
	 * row {@code r} survived the deletion/selection vector. Batch-level consumption
	 * lets callers extract data column-at-a-time (cache friendly, boxing free)
	 * instead of paying a per-row callback.
	 */
	@FunctionalInterface
	public interface BatchConsumer {
		void accept(ColumnVector[] cols, int size, boolean[] selected);
	}

	/**
	 * Map a Delta Kernel {@link DataType} to an internal type code (see the
	 * {@code T_*} constants). Returned once per column so the per-cell read loop
	 * can switch on a primitive int instead of repeating {@code instanceof} checks.
	 *
	 * @param dt the Delta column data type
	 * @return the matching {@code T_*} code, or {@code -1} if the type is not supported
	 */
	public static int typeCode(DataType dt) {
		if( dt instanceof DoubleType )  return T_DOUBLE;
		if( dt instanceof FloatType )   return T_FLOAT;
		if( dt instanceof LongType )    return T_LONG;
		if( dt instanceof IntegerType ) return T_INT;
		if( dt instanceof ShortType )   return T_SHORT;
		if( dt instanceof ByteType )    return T_BYTE;
		if( dt instanceof BooleanType ) return T_BOOLEAN;
		if( dt instanceof StringType )  return T_STRING;
		return -1;
	}

	/**
	 * @param size     number of rows in the batch
	 * @param selected per-row selection mask, or {@code null} if all rows are live
	 * @return the number of live rows in the batch
	 */
	public static int countSelected(int size, boolean[] selected) {
		if(selected == null)
			return size;
		int n = 0;
		for(int r = 0; r < size; r++)
			if(selected[r])
				n++;
		return n;
	}

	// ------------------------------------------
	// direct parquet decode of Delta data files
	// ------------------------------------------

	/** Physical-schema metadata key carrying the parquet field id (column mapping mode {@code id}). */
	private static final String PARQUET_FIELD_ID_KEY = "parquet.field.id";

	/**
	 * Whether data files can be decoded directly into pre-allocated output columns: the physical read schema must be a
	 * positional 1:1 image of the logical schema, i.e. no partition columns (not stored in the data files, spliced back
	 * in by the kernel) and no kernel metadata columns such as {@code row_index} (only requested for deletion-vector
	 * reads). Deletion vectors themselves are excluded separately via the exact-row-count check.
	 *
	 * @param logicalSchema  the table's logical schema
	 * @param physicalSchema the physical read schema from the scan state
	 * @return true if data files can be decoded directly
	 */
	public static boolean supportsDirectDecode(StructType logicalSchema, StructType physicalSchema) {
		if(physicalSchema.length() != logicalSchema.length())
			return false;
		for(int c = 0; c < physicalSchema.length(); c++)
			if(physicalSchema.at(c).isMetadataColumn())
				return false;
		return true;
	}

	/** @param scanFileRow a scan-file row @return the fully-qualified path of its data file */
	public static String dataFilePath(Row scanFileRow) {
		return InternalScanFileUtils.getAddFileStatus(scanFileRow).getPath();
	}

	/**
	 * Decode one Delta data file into pre-allocated typed column arrays at the given absolute row offset, through
	 * parquet-mr's column API ({@link ColumnReadStoreImpl}/{@link ColumnReader}) with no kernel engine or intermediate
	 * batch vectors in the path. Columns are resolved by parquet field id first (column mapping mode {@code id}) and
	 * physical name second; columns absent from the file (schema evolution) keep the array defaults (0 for numerics,
	 * null for strings), matching the kernel-path null semantics.
	 *
	 * @param filePath       fully-qualified path of the parquet data file
	 * @param physicalSchema physical read schema (positionally 1:1 with the output columns)
	 * @param readCodes      per-column type codes (see the {@code T_*} constants)
	 * @param dest           pre-allocated per-column backing arrays
	 * @param destOff        absolute row offset of this file's first row
	 * @param limit          exclusive upper row bound of this file's slice
	 * @param tablePath      table path for error messages
	 * @return the number of rows decoded
	 * @throws IOException on read failure
	 */
	public static int decodeDataFileInto(String filePath, StructType physicalSchema, int[] readCodes, Object[] dest,
		int destOff, int limit, String tablePath) throws IOException {
		final Configuration conf = ConfigurationManager.getCachedJobConf();
		final int ncol = physicalSchema.length();
		int off = destOff;
		try(ParquetFileReader reader = ParquetFileReader.open(HadoopInputFile.fromPath(new Path(filePath), conf))) {
			MessageType parquetSchema = reader.getFooter().getFileMetaData().getSchema();
			String createdBy = reader.getFooter().getFileMetaData().getCreatedBy();
			String[] colNames = resolveParquetColumns(physicalSchema, parquetSchema);
			GroupConverter root = dummyConverter(parquetSchema.getFieldCount());
			PageReadStore pages;
			while((pages = reader.readNextRowGroup()) != null) {
				int nrow = (int) pages.getRowCount();
				// guard before decoding: writing past the limit would overflow into the
				// next file's slice (or off the array) in the pre-allocated output
				if(off + nrow > limit)
					throw new DMLRuntimeException("Delta file produced more rows than its "
						+ "numRecords statistic; refusing direct read of " + tablePath);
				ColumnReadStoreImpl store = new ColumnReadStoreImpl(pages, root, parquetSchema, createdBy);
				for(int c = 0; c < ncol; c++) {
					if(colNames[c] == null)
						continue; // column absent from this data file -> keep defaults (nulls)
					ColumnDescriptor desc = parquetSchema.getColumnDescription(new String[] {colNames[c]});
					decodeColumnInto(store.getColumnReader(desc), desc.getMaxDefinitionLevel(), nrow, readCodes[c],
						dest[c], off);
				}
				off += nrow;
			}
		}
		return off - destOff;
	}

	/**
	 * Resolve each physical-schema column to the parquet column name of the given file: by parquet field id when the
	 * schema carries one, by name otherwise, or null when the file does not contain the column at all.
	 */
	private static String[] resolveParquetColumns(StructType schema, MessageType parquetSchema) {
		Map<Integer, String> idToName = new HashMap<>();
		Map<String, String> names = new HashMap<>();
		for(int i = 0; i < parquetSchema.getFieldCount(); i++) {
			org.apache.parquet.schema.Type t = parquetSchema.getType(i);
			names.put(t.getName(), t.getName());
			if(t.getId() != null)
				idToName.put(t.getId().intValue(), t.getName());
		}
		String[] resolved = new String[schema.length()];
		for(int c = 0; c < schema.length(); c++) {
			Object fid = schema.at(c).getMetadata().get(PARQUET_FIELD_ID_KEY);
			String byId = (fid instanceof Number) ? idToName.get(((Number) fid).intValue()) : null;
			resolved[c] = (byId != null) ? byId : names.get(schema.at(c).getName());
		}
		return resolved;
	}

	/** No-op converter tree; the column API requires one, but values are pulled via the typed getters. */
	private static GroupConverter dummyConverter(int nFields) {
		final PrimitiveConverter[] leaves = new PrimitiveConverter[nFields];
		for(int i = 0; i < nFields; i++)
			leaves[i] = new PrimitiveConverter() {
			};
		return new GroupConverter() {
			@Override
			public Converter getConverter(int fieldIndex) {
				return leaves[fieldIndex];
			}

			@Override
			public void start() {
			}

			@Override
			public void end() {
			}
		};
	}

	/**
	 * Decode one parquet column of the current row group into a pre-allocated typed array at the given offset. Null
	 * cells (definition level below max) keep the array default (0 for numerics, null for strings).
	 */
	private static void decodeColumnInto(ColumnReader creader, int maxDef, int nrow, int readCode, Object dest,
		int off) {
		switch(readCode) {
			case T_DOUBLE: {
				double[] a = (double[]) dest;
				for(int r = 0; r < nrow; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[off + r] = creader.getDouble();
					creader.consume();
				}
				break;
			}
			case T_FLOAT: {
				float[] a = (float[]) dest;
				for(int r = 0; r < nrow; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[off + r] = creader.getFloat();
					creader.consume();
				}
				break;
			}
			case T_LONG: {
				long[] a = (long[]) dest;
				for(int r = 0; r < nrow; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[off + r] = creader.getLong();
					creader.consume();
				}
				break;
			}
			case T_INT:
			case T_SHORT:
			case T_BYTE: {
				// delta short/byte columns are stored as annotated parquet INT32
				int[] a = (int[]) dest;
				for(int r = 0; r < nrow; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[off + r] = creader.getInteger();
					creader.consume();
				}
				break;
			}
			case T_BOOLEAN: {
				boolean[] a = (boolean[]) dest;
				for(int r = 0; r < nrow; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[off + r] = creader.getBoolean();
					creader.consume();
				}
				break;
			}
			case T_STRING: {
				String[] a = (String[]) dest;
				for(int r = 0; r < nrow; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[off + r] = creader.getBinary().toStringUsingUTF8();
					creader.consume();
				}
				break;
			}
			default:
				throw new DMLRuntimeException("Unsupported read code for direct decode: " + readCode);
		}
	}

	/** Floor on the adaptive writer target file size. Below this the per-file metadata/open
	 * overhead (and tiny-file proliferation) outweighs the extra read parallelism. */
	public static final long ADAPTIVE_WRITER_MIN_FILE_SIZE = 4L * 1024 * 1024;

	private static Configuration buildConf(Configuration base, int batchSize, long targetFileSize) {
		Configuration c = new Configuration(base);
		c.setInt(CONF_READER_BATCH_SIZE, batchSize);
		c.setLong(CONF_WRITER_TARGET_FILE_SIZE, targetFileSize);
		return c;
	}

	private static synchronized Configuration deltaConf() {
		Configuration base = ConfigurationManager.getCachedJobConf();
		int batchSize = ConfigurationManager.getDeltaReaderBatchSize();
		long targetFileSize = ConfigurationManager.getDeltaWriterTargetFileSize();
		if(cachedConf == null || cachedConfBase != base
			|| cachedBatchSize != batchSize || cachedTargetFileSize != targetFileSize)
		{
			cachedConf = buildConf(base, batchSize, targetFileSize);
			cachedConfBase = base;
			cachedBatchSize = batchSize;
			cachedTargetFileSize = targetFileSize;
		}
		return cachedConf;
	}

	public static Engine createEngine() {
		return DefaultEngine.create(deltaConf());
	}

	/**
	 * Compute the parquet target data-file size (bytes) for writing a table of the given
	 * estimated size. With adaptive sizing enabled the writer aims for roughly one data
	 * file per expected parallel reader (so the native per-file parallel read can use all
	 * threads): never above the configured target, and never below
	 * {@code ADAPTIVE_WRITER_MIN_FILE_SIZE} unless the configured target is itself smaller
	 * than that floor (in which case the configured target wins).
	 *
	 * @param estimatedBytes estimate of the table's size (the block in-memory size is a fine proxy)
	 * @return the target max parquet data-file size in bytes
	 */
	public static long adaptiveWriterTargetFileSize(long estimatedBytes) {
		long configured = ConfigurationManager.getDeltaWriterTargetFileSize();
		if(!ConfigurationManager.isDeltaWriterAdaptiveFileSize() || estimatedBytes <= 0)
			return configured;
		int par = Math.max(1, OptimizerUtils.getParallelBinaryReadParallelism());
		long perReader = Math.max(1, estimatedBytes / par);
		//never above the configured cap, never below the floor (unless the cap itself is lower)
		long target = Math.min(configured, Math.max(ADAPTIVE_WRITER_MIN_FILE_SIZE, perReader));
		if(LOG.isDebugEnabled())
			LOG.debug("Delta adaptive file size: est=" + estimatedBytes + "B par=" + par + " -> target=" + target
				+ "B (cap=" + configured + "B, floor=" + ADAPTIVE_WRITER_MIN_FILE_SIZE + "B)");
		return target;
	}

	/**
	 * Create an engine for writing a table of the given estimated size, configured with an
	 * adaptive target data-file size (see {@link #adaptiveWriterTargetFileSize(long)}). A fresh
	 * (uncached) configuration is built since writes happen once per table, not per data file.
	 *
	 * @param estimatedBytes estimate of the table's size (the block in-memory size is a fine proxy)
	 * @return a Delta Kernel engine for the write
	 */
	public static Engine createWriteEngine(long estimatedBytes) {
		//the reader batch size is irrelevant on the write path but is set to keep the
		//conf shape identical to deltaConf(); only the target file size matters here.
		Configuration c = buildConf(ConfigurationManager.getCachedJobConf(),
			ConfigurationManager.getDeltaReaderBatchSize(), adaptiveWriterTargetFileSize(estimatedBytes));
		return DefaultEngine.create(c);
	}

	/**
	 * Resolve a (possibly relative) path to a fully-qualified URI so the
	 * kernel's default engine can locate the table on the right filesystem.
	 *
	 * @param fname input path
	 * @return fully-qualified table path
	 */
	public static String qualify(String fname) {
		try {
			Configuration conf = ConfigurationManager.getCachedJobConf();
			Path path = new Path(fname);
			return path.getFileSystem(conf).makeQualified(path).toString();
		}
		catch(IOException ex) {
			throw new DMLRuntimeException("Failed to resolve Delta table path: " + fname, ex);
		}
	}

	/**
	 * Opened latest snapshot of a Delta table: the logical schema plus everything
	 * needed to (re)read its data files, including the list of per-data-file scan
	 * rows. Delta Kernel scan-file rows are self-contained (the kernel's
	 * distributed design serializes them to workers), so they can be retained and
	 * read independently / in parallel.
	 */
	public static final class ScanHandle {
		public final StructType schema;
		public final Row scanState;
		public final StructType physicalReadSchema;
		public final List<Row> scanFiles;
		/**
		 * Per-file record counts taken from the Delta {@code numRecords} statistic,
		 * aligned with {@link #scanFiles}; {@code -1} where the statistic is absent.
		 */
		public final long[] numRecords;
		/**
		 * Per-file flag indicating a deletion vector is present (so the live row
		 * count differs from {@link #numRecords}), aligned with {@link #scanFiles}.
		 */
		public final boolean[] hasDeletionVector;

		private ScanHandle(StructType schema, Row scanState, StructType physicalReadSchema,
			List<Row> scanFiles, long[] numRecords, boolean[] hasDeletionVector)
		{
			this.schema = schema;
			this.scanState = scanState;
			this.physicalReadSchema = physicalReadSchema;
			this.scanFiles = scanFiles;
			this.numRecords = numRecords;
			this.hasDeletionVector = hasDeletionVector;
		}

		/**
		 * @return true iff every data file carries a {@code numRecords} statistic
		 *         and none has a deletion vector, i.e. exact per-file row offsets
		 *         can be derived from metadata without reading the data.
		 */
		public boolean hasExactRowCounts() {
			for( int i=0; i<numRecords.length; i++ )
				if( numRecords[i] < 0 || hasDeletionVector[i] )
					return false;
			return true;
		}
	}

	/**
	 * Open the latest snapshot of a Delta table and enumerate its data files.
	 *
	 * @param engine    delta kernel engine
	 * @param tablePath fully-qualified table path
	 * @return a handle carrying the schema, scan state, physical read schema and
	 *         one scan-file row per data file
	 * @throws IOException on metadata read failure
	 */
	public static ScanHandle openScan(Engine engine, String tablePath) throws IOException {
		Table table = Table.forPath(engine, tablePath);
		Snapshot snapshot = table.getLatestSnapshot(engine);
		StructType schema = snapshot.getSchema(engine);

		Scan scan = snapshot.getScanBuilder(engine).build();
		Row scanState = scan.getScanState(engine);
		StructType physicalReadSchema = ScanStateRow.getPhysicalDataReadSchema(engine, scanState);

		//request the scan files WITH per-file statistics (numRecords) so callers can
		//pre-size output and place rows without reading the data; harmless extra
		//column for the data-read path. Fall back to the stats-less iterator if the
		//concrete scan does not support it.
		CloseableIterator<FilteredColumnarBatch> scanFileIter = (scan instanceof ScanImpl)
			? ((ScanImpl) scan).getScanFiles(engine, true)
			: scan.getScanFiles(engine);

		List<Row> files = new ArrayList<>();
		List<Long> recs = new ArrayList<>();
		List<Boolean> dvs = new ArrayList<>();
		try( CloseableIterator<FilteredColumnarBatch> scanFiles = scanFileIter ) {
			while( scanFiles.hasNext() ) {
				FilteredColumnarBatch scanFileBatch = scanFiles.next();
				try( CloseableIterator<Row> scanFileRows = scanFileBatch.getRows() ) {
					while( scanFileRows.hasNext() ) {
						Row scanFileRow = scanFileRows.next();
						files.add(scanFileRow);
						recs.add(numRecords(scanFileRow));
						dvs.add(InternalScanFileUtils.getDeletionVectorDescriptorFromRow(scanFileRow) != null);
					}
				}
			}
		}
		long[] numRecords = new long[recs.size()];
		boolean[] hasDv = new boolean[dvs.size()];
		for( int i=0; i<numRecords.length; i++ ) {
			numRecords[i] = recs.get(i);
			hasDv[i] = dvs.get(i);
		}
		return new ScanHandle(schema, scanState, physicalReadSchema, files, numRecords, hasDv);
	}

	/**
	 * Extract the {@code numRecords} statistic from a scan-file row, or {@code -1}
	 * if the stats string is absent / does not contain the field. The stats are a
	 * small JSON document such as {@code {"numRecords":1048310,...}}.
	 */
	private static long numRecords(Row scanFileRow) {
		Row add = scanFileRow.getStruct(InternalScanFileUtils.ADD_FILE_ORDINAL);
		int statsOrd = add.getSchema().fieldNames().indexOf("stats");
		if( statsOrd < 0 || add.isNullAt(statsOrd) )
			return -1;
		String stats = add.getString(statsOrd);
		if( stats == null )
			return -1;
		try {
			JsonNode node = JSON_MAPPER.readTree(stats).get("numRecords");
			return (node != null && node.canConvertToLong()) ? node.asLong() : -1;
		}
		catch(JsonProcessingException ex) {
			return -1;
		}
	}

	/**
	 * Read a single Delta data file (identified by its scan-file row), decoding
	 * its parquet batches and applying any deletion vector, invoking the consumer
	 * once per (logical) batch. Safe to call concurrently for distinct files as
	 * long as each call uses its own {@code engine}.
	 *
	 * @param engine             delta kernel engine
	 * @param scanState          scan state from {@link #openScan}
	 * @param physicalReadSchema physical read schema from {@link #openScan}
	 * @param scanFileRow        the data file's scan-file row
	 * @param consumer           batch consumer
	 * @throws IOException on read failure
	 */
	public static void readScanFile(Engine engine, Row scanState, StructType physicalReadSchema,
		Row scanFileRow, BatchConsumer consumer) throws IOException
	{
		FileStatus dataFile = InternalScanFileUtils.getAddFileStatus(scanFileRow);
		CloseableIterator<ColumnarBatch> physicalData = engine.getParquetHandler()
			.readParquetFiles(Utils.singletonCloseableIterator(dataFile), physicalReadSchema, Optional.empty());
		try( CloseableIterator<FilteredColumnarBatch> logicalData =
			Scan.transformPhysicalData(engine, scanState, scanFileRow, physicalData) )
		{
			while( logicalData.hasNext() )
				consumeBatch(logicalData.next(), consumer);
		}
	}

	/**
	 * Scan the latest snapshot of a Delta table sequentially, invoking the batch
	 * consumer for every data batch. The consumer is created lazily from the table
	 * schema (so callers can size buffers / derive per-column types up front).
	 *
	 * @param engine          delta kernel engine
	 * @param tablePath       fully-qualified table path
	 * @param consumerFactory builds the batch consumer from the table schema
	 * @return the logical table schema
	 * @throws IOException on read failure
	 */
	public static StructType scan(Engine engine, String tablePath, Function<StructType, BatchConsumer> consumerFactory)
		throws IOException
	{
		ScanHandle h = openScan(engine, tablePath);
		BatchConsumer consumer = consumerFactory.apply(h.schema);
		for( Row scanFileRow : h.scanFiles )
			readScanFile(engine, h.scanState, h.physicalReadSchema, scanFileRow, consumer);
		return h.schema;
	}

	private static void consumeBatch(FilteredColumnarBatch fcb, BatchConsumer consumer) {
		ColumnarBatch batch = fcb.getData();
		int ncol = batch.getSchema().length();
		ColumnVector[] cols = new ColumnVector[ncol];
		for( int c=0; c<ncol; c++ )
			cols[c] = batch.getColumnVector(c);
		int size = batch.getSize();

		//materialize the deletion/selection mask once (null => all rows live)
		Optional<ColumnVector> selVector = fcb.getSelectionVector();
		boolean[] selected = null;
		if( selVector.isPresent() ) {
			ColumnVector sv = selVector.get();
			selected = new boolean[size];
			for( int r=0; r<size; r++ )
				selected[r] = !sv.isNullAt(r) && sv.getBoolean(r);
		}
		consumer.accept(cols, size, selected);
	}

	/**
	 * Create a Delta table at the target path and commit the given logical data as
	 * parquet data files in a single transaction. Any existing file/directory at
	 * the target path is deleted first, so a write fully replaces what was there
	 * (matching the overwrite semantics of the other SystemDS writers). Append /
	 * incremental table updates are not supported.
	 *
	 * @param engine      delta kernel engine
	 * @param tablePath   fully-qualified table path
	 * @param schema      table schema to create
	 * @param logicalData logical (unpartitioned) data batches to write
	 * @throws IOException on write failure
	 */
	public static void commit(Engine engine, String tablePath, StructType schema,
		CloseableIterator<FilteredColumnarBatch> logicalData) throws IOException
	{
		//replace any existing table at the path (the other SystemDS writers delete
		//the output first; the caching layer does not do it on our behalf)
		HDFSTool.deleteFileIfExistOnHDFS(tablePath);

		Table table = Table.forPath(engine, tablePath);
		TransactionBuilder txnBuilder = table
			.createTransactionBuilder(engine, ENGINE_INFO, Operation.CREATE_TABLE)
			.withSchema(engine, schema);
		Transaction txn = txnBuilder.build(engine);
		Row txnState = txn.getTransactionState(engine);

		CloseableIterator<FilteredColumnarBatch> physicalData =
			Transaction.transformLogicalData(engine, txnState, logicalData, Collections.emptyMap());
		DataWriteContext writeContext =
			Transaction.getWriteContext(engine, txnState, Collections.emptyMap());
		CloseableIterator<DataFileStatus> dataFiles = engine.getParquetHandler()
			.writeParquetFiles(writeContext.getTargetDirectory(), physicalData, writeContext.getStatisticsColumns());
		CloseableIterator<Row> appendActions =
			Transaction.generateAppendActions(engine, txnState, dataFiles, writeContext);
		txn.commit(engine, CloseableIterable.inMemoryIterable(appendActions));
	}
}

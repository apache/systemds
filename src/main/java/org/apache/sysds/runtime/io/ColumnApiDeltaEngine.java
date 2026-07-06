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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.ColumnReader;
import org.apache.parquet.column.impl.ColumnReadStoreImpl;
import org.apache.parquet.column.page.PageReadStore;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.io.api.Converter;
import org.apache.parquet.io.api.GroupConverter;
import org.apache.parquet.io.api.PrimitiveConverter;
import org.apache.parquet.schema.MessageType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;

import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.data.ColumnarBatch;
import io.delta.kernel.data.FilteredColumnarBatch;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.engine.ExpressionHandler;
import io.delta.kernel.engine.FileSystemClient;
import io.delta.kernel.engine.JsonHandler;
import io.delta.kernel.engine.ParquetHandler;
import io.delta.kernel.expressions.Column;
import io.delta.kernel.expressions.Predicate;
import io.delta.kernel.types.DataType;
import io.delta.kernel.types.LongType;
import io.delta.kernel.types.StructField;
import io.delta.kernel.types.StructType;
import io.delta.kernel.utils.CloseableIterator;
import io.delta.kernel.utils.DataFileStatus;
import io.delta.kernel.utils.FileStatus;

/**
 * Delta Kernel {@link Engine} whose {@link ParquetHandler} decodes flat data files through
 * parquet-mr's low-level column API ({@link ColumnReadStoreImpl}/{@link ColumnReader}) instead of the default
 * engine's row-record path ({@code org.apache.parquet.hadoop.ParquetReader}).
 * Everything else delegates to the wrapped default engine, and deletion vectors / column mapping are still applied by the kernel.
 *
 * Beyond plain decoding the fast path honors the {@link ParquetHandler#readParquetFiles} contract cases the
 * kernel relies on: the {@code _metadata.row_index} metadata column (requested for every read of a table whose
 * protocol carries the {@code deletionVectors} feature) is synthesized from the row-group row offsets, columns are
 * resolved by parquet field id first and name second (column mapping mode {@code id}), and columns absent from a
 * data file are returned as all-null vectors. Reads with predicates, nested/unsupported types,
 * or metadata columns other than {@code row_index} fall back to the wrapped default engine.
 */
public class ColumnApiDeltaEngine implements Engine {

	private final Engine _delegate;

	public ColumnApiDeltaEngine(Engine delegate) {
		_delegate = delegate;
	}

	@Override
	public ExpressionHandler getExpressionHandler() {
		return _delegate.getExpressionHandler();
	}

	@Override
	public JsonHandler getJsonHandler() {
		return _delegate.getJsonHandler();
	}

	@Override
	public FileSystemClient getFileSystemClient() {
		return _delegate.getFileSystemClient();
	}

	@Override
	public ParquetHandler getParquetHandler() {
		return new ColumnApiParquetHandler(_delegate.getParquetHandler());
	}

	private static class ColumnApiParquetHandler implements ParquetHandler {
		private final ParquetHandler _fallback;

		ColumnApiParquetHandler(ParquetHandler fallback) {
			_fallback = fallback;
		}

		@Override
		public CloseableIterator<ColumnarBatch> readParquetFiles(CloseableIterator<FileStatus> files,
			StructType physicalSchema, Optional<Predicate> predicate) throws IOException {
			// fast path only for flat schemas of decodable primitives; predicates, nested
			// reads and unknown metadata columns keep the default row-record decode
			if(predicate.isPresent() || !supportsFastPath(physicalSchema))
				return _fallback.readParquetFiles(files, physicalSchema, predicate);
			return new ColumnApiBatchIterator(files, physicalSchema, ConfigurationManager.getCachedJobConf());
		}

		@Override
		public CloseableIterator<DataFileStatus> writeParquetFiles(String directoryPath,
			CloseableIterator<FilteredColumnarBatch> dataIter, List<Column> statsColumns) throws IOException {
			return _fallback.writeParquetFiles(directoryPath, dataIter, statsColumns);
		}

		@Override
		public void writeParquetFileAtomically(String filePath, CloseableIterator<FilteredColumnarBatch> data)
			throws IOException {
			_fallback.writeParquetFileAtomically(filePath, data);
		}

		private static boolean supportsFastPath(StructType schema) {
			for(int c = 0; c < schema.length(); c++) {
				StructField f = schema.at(c);
				if(isRowIndexColumn(f))
					continue; // synthesized, need not exist in the data files
				if(f.isMetadataColumn() || DeltaKernelUtils.typeCode(f.getDataType()) < 0)
					return false;
			}
			return true;
		}
	}

	/** The metadata column the kernel asks the parquet handler to fill with the file row index. */
	private static boolean isRowIndexColumn(StructField f) {
		return f.isMetadataColumn() && StructField.METADATA_ROW_INDEX_COLUMN_NAME.equals(f.getName());
	}

	/**
	 * Streams the requested files as one {@link ColumnarBatch} per parquet row group, decoding each column
	 * directly off {@link ColumnReader} into a pre-sized primitive array.
	 */
	private static class ColumnApiBatchIterator implements CloseableIterator<ColumnarBatch> {
		/** Physical-schema metadata key carrying the parquet field id (column mapping mode {@code id}). */
		private static final String PARQUET_FIELD_ID_KEY = "parquet.field.id";

		private final CloseableIterator<FileStatus> _files;
		private final StructType _schema;
		private final Configuration _conf;

		private ParquetFileReader _reader;
		private MessageType _parquetSchema;
		private String _createdBy;
		private GroupConverter _rootConverter;
		private ColumnarBatch _next;
		/** Per schema ordinal the resolved parquet column name of the current file, or null if absent there. */
		private String[] _parquetColNames;
		/** File row index of the next row group's first row (fallback when the page store carries no offset). */
		private long _fileRowOffset;

		ColumnApiBatchIterator(CloseableIterator<FileStatus> files, StructType schema, Configuration conf) {
			_files = files;
			_schema = schema;
			_conf = conf;
		}

		@Override
		public boolean hasNext() {
			if(_next == null)
				_next = advance();
			return _next != null;
		}

		@Override
		public ColumnarBatch next() {
			if(!hasNext())
				throw new java.util.NoSuchElementException();
			ColumnarBatch b = _next;
			_next = null;
			return b;
		}

		private ColumnarBatch advance() {
			try {
				while(true) {
					if(_reader != null) {
						PageReadStore pages = _reader.readNextRowGroup();
						if(pages != null)
							return decodeRowGroup(pages);
						_reader.close();
						_reader = null;
					}
					if(!_files.hasNext())
						return null;
					openFile(_files.next());
				}
			}
			catch(IOException ex) {
				throw new DMLRuntimeException("Column-API parquet decode failed", ex);
			}
		}

		private void openFile(FileStatus file) throws IOException {
			_reader = ParquetFileReader.open(HadoopInputFile.fromPath(new Path(file.getPath()), _conf));
			ParquetMetadata meta = _reader.getFooter();
			_parquetSchema = meta.getFileMetaData().getSchema();
			_createdBy = meta.getFileMetaData().getCreatedBy();
			_parquetColNames = resolveParquetColumns(_schema, _parquetSchema);
			_fileRowOffset = 0;
			final int n = _parquetSchema.getFieldCount();
			final PrimitiveConverter[] leaves = new PrimitiveConverter[n];
			for(int i = 0; i < n; i++)
				leaves[i] = new PrimitiveConverter() {};
			_rootConverter = new GroupConverter() {
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

		private ColumnarBatch decodeRowGroup(PageReadStore pages) {
			final int nrow = (int) pages.getRowCount();
			final int ncol = _schema.length();
			final long rowIndexBase = pages.getRowIndexOffset().orElse(_fileRowOffset);
			ColumnReadStoreImpl store = new ColumnReadStoreImpl(pages, _rootConverter, _parquetSchema, _createdBy);
			ColumnVector[] vectors = new ColumnVector[ncol];
			for(int c = 0; c < ncol; c++) {
				StructField field = _schema.at(c);
				if(isRowIndexColumn(field)) {
					vectors[c] = rowIndexVector(rowIndexBase, nrow);
					continue;
				}
				String name = _parquetColNames[c];
				if(name == null) { // column absent from this data file (schema evolution)
					vectors[c] = new NullVector(field.getDataType(), nrow);
					continue;
				}
				ColumnDescriptor desc = _parquetSchema.getColumnDescription(new String[] {name});
				vectors[c] = decodeColumn(store.getColumnReader(desc), desc.getMaxDefinitionLevel(), nrow,
					field.getDataType());
			}
			_fileRowOffset += nrow;
			return new ArrayBackedBatch(_schema, vectors, nrow);
		}

		/**
		 * Resolve each schema column to the parquet column name of the current file: by parquet field id when the
		 * physical schema carries one (column mapping mode {@code id}), by name otherwise, null when the file does
		 * not contain the column at all.
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
				StructField f = schema.at(c);
				if(isRowIndexColumn(f))
					continue; // synthesized, never read from the file
				Object fid = f.getMetadata().get(PARQUET_FIELD_ID_KEY);
				String byId = (fid instanceof Number) ? idToName.get(((Number) fid).intValue()) : null;
				resolved[c] = (byId != null) ? byId : names.get(f.getName());
			}
			return resolved;
		}

		private static ColumnVector rowIndexVector(long base, int nrow) {
			long[] a = new long[nrow];
			for(int r = 0; r < nrow; r++)
				a[r] = base + r;
			return TypedVector.longs(LongType.LONG, nrow, null, a);
		}

		private static ColumnVector decodeColumn(ColumnReader creader, int maxDef, int nrow, DataType dt) {
			boolean[] nulls = maxDef > 0 ? new boolean[nrow] : null;
			int code = DeltaKernelUtils.typeCode(dt);
			switch(code) {
				case DeltaKernelUtils.T_DOUBLE: {
					double[] a = new double[nrow];
					for(int r = 0; r < nrow; r++) {
						if(creader.getCurrentDefinitionLevel() == maxDef)
							a[r] = creader.getDouble();
						else
							nulls[r] = true;
						creader.consume();
					}
					return TypedVector.doubles(dt, nrow, nulls, a);
				}
				case DeltaKernelUtils.T_FLOAT: {
					float[] a = new float[nrow];
					for(int r = 0; r < nrow; r++) {
						if(creader.getCurrentDefinitionLevel() == maxDef)
							a[r] = creader.getFloat();
						else
							nulls[r] = true;
						creader.consume();
					}
					return TypedVector.floats(dt, nrow, nulls, a);
				}
				case DeltaKernelUtils.T_LONG: {
					long[] a = new long[nrow];
					for(int r = 0; r < nrow; r++) {
						if(creader.getCurrentDefinitionLevel() == maxDef)
							a[r] = creader.getLong();
						else
							nulls[r] = true;
						creader.consume();
					}
					return TypedVector.longs(dt, nrow, nulls, a);
				}
				case DeltaKernelUtils.T_INT:
				case DeltaKernelUtils.T_SHORT:
				case DeltaKernelUtils.T_BYTE: {
					// delta short/byte columns are stored as annotated parquet INT32
					int[] a = new int[nrow];
					for(int r = 0; r < nrow; r++) {
						if(creader.getCurrentDefinitionLevel() == maxDef)
							a[r] = creader.getInteger();
						else
							nulls[r] = true;
						creader.consume();
					}
					return TypedVector.ints(dt, nrow, nulls, a);
				}
				case DeltaKernelUtils.T_BOOLEAN: {
					boolean[] a = new boolean[nrow];
					for(int r = 0; r < nrow; r++) {
						if(creader.getCurrentDefinitionLevel() == maxDef)
							a[r] = creader.getBoolean();
						else
							nulls[r] = true;
						creader.consume();
					}
					return TypedVector.booleans(dt, nrow, nulls, a);
				}
				case DeltaKernelUtils.T_STRING: {
					String[] a = new String[nrow];
					for(int r = 0; r < nrow; r++) {
						if(creader.getCurrentDefinitionLevel() == maxDef)
							a[r] = creader.getBinary().toStringUsingUTF8();
						else
							nulls[r] = true;
						creader.consume();
					}
					return TypedVector.strings(dt, nrow, nulls, a);
				}
				default:
					throw new DMLRuntimeException("Unsupported delta type for column-API decode: " + dt);
			}
		}

		@Override
		public void close() throws IOException {
			if(_reader != null) {
				_reader.close();
				_reader = null;
			}
			_files.close();
		}
	}

	/** Columnar batch over decoded per-column arrays, with the with* methods the kernel may invoke. */
	private static class ArrayBackedBatch implements ColumnarBatch {
		private final StructType _schema;
		private final ColumnVector[] _vectors;
		private final int _size;

		ArrayBackedBatch(StructType schema, ColumnVector[] vectors, int size) {
			_schema = schema;
			_vectors = vectors;
			_size = size;
		}

		@Override
		public StructType getSchema() {
			return _schema;
		}

		@Override
		public ColumnVector getColumnVector(int ordinal) {
			return _vectors[ordinal];
		}

		@Override
		public int getSize() {
			return _size;
		}

		@Override
		public ColumnarBatch withNewColumn(int ordinal, StructField field, ColumnVector vector) {
			List<StructField> fields = new ArrayList<>(_schema.fields());
			fields.add(ordinal, field);
			ColumnVector[] vs = new ColumnVector[_vectors.length + 1];
			System.arraycopy(_vectors, 0, vs, 0, ordinal);
			vs[ordinal] = vector;
			System.arraycopy(_vectors, ordinal, vs, ordinal + 1, _vectors.length - ordinal);
			return new ArrayBackedBatch(new StructType(fields), vs, _size);
		}

		@Override
		public ColumnarBatch withDeletedColumnAt(int ordinal) {
			List<StructField> fields = new ArrayList<>(_schema.fields());
			fields.remove(ordinal);
			ColumnVector[] vs = new ColumnVector[_vectors.length - 1];
			System.arraycopy(_vectors, 0, vs, 0, ordinal);
			System.arraycopy(_vectors, ordinal + 1, vs, ordinal, _vectors.length - ordinal - 1);
			return new ArrayBackedBatch(new StructType(fields), vs, _size);
		}

		@Override
		public ColumnarBatch withNewSchema(StructType newSchema) {
			return new ArrayBackedBatch(newSchema, _vectors, _size);
		}
	}

	/** All-null vector for a column that is absent from a data file (added to the table after the file was written). */
	private static class NullVector implements ColumnVector {
		private final DataType _type;
		private final int _size;

		NullVector(DataType type, int size) {
			_type = type;
			_size = size;
		}

		@Override
		public DataType getDataType() {
			return _type;
		}

		@Override
		public int getSize() {
			return _size;
		}

		@Override
		public boolean isNullAt(int rowId) {
			return true;
		}

		@Override
		public void close() {
			// nothing to release
		}
	}

	/** Typed vector over one decoded primitive array; exactly one backing array is non-null. */
	private static class TypedVector implements ColumnVector {
		private final DataType _type;
		private final int _size;
		private final boolean[] _nulls; // null => column has no nulls
		private double[] _d;
		private float[] _f;
		private long[] _l;
		private int[] _i;
		private boolean[] _b;
		private String[] _s;

		private TypedVector(DataType type, int size, boolean[] nulls) {
			_type = type;
			_size = size;
			_nulls = nulls;
		}

		static TypedVector doubles(DataType t, int n, boolean[] nulls, double[] a) {
			TypedVector v = new TypedVector(t, n, nulls);
			v._d = a;
			return v;
		}

		static TypedVector floats(DataType t, int n, boolean[] nulls, float[] a) {
			TypedVector v = new TypedVector(t, n, nulls);
			v._f = a;
			return v;
		}

		static TypedVector longs(DataType t, int n, boolean[] nulls, long[] a) {
			TypedVector v = new TypedVector(t, n, nulls);
			v._l = a;
			return v;
		}

		static TypedVector ints(DataType t, int n, boolean[] nulls, int[] a) {
			TypedVector v = new TypedVector(t, n, nulls);
			v._i = a;
			return v;
		}

		static TypedVector booleans(DataType t, int n, boolean[] nulls, boolean[] a) {
			TypedVector v = new TypedVector(t, n, nulls);
			v._b = a;
			return v;
		}

		static TypedVector strings(DataType t, int n, boolean[] nulls, String[] a) {
			TypedVector v = new TypedVector(t, n, nulls);
			v._s = a;
			return v;
		}

		@Override
		public DataType getDataType() {
			return _type;
		}

		@Override
		public int getSize() {
			return _size;
		}

		@Override
		public boolean isNullAt(int rowId) {
			return _nulls != null && _nulls[rowId];
		}

		@Override
		public double getDouble(int rowId) {
			return _d[rowId];
		}

		@Override
		public float getFloat(int rowId) {
			return _f[rowId];
		}

		@Override
		public long getLong(int rowId) {
			return _l[rowId];
		}

		@Override
		public int getInt(int rowId) {
			return _i[rowId];
		}

		@Override
		public short getShort(int rowId) {
			return (short) _i[rowId];
		}

		@Override
		public byte getByte(int rowId) {
			return (byte) _i[rowId];
		}

		@Override
		public boolean getBoolean(int rowId) {
			return _b[rowId];
		}

		@Override
		public String getString(int rowId) {
			return _s[rowId];
		}

		@Override
		public void close() {
			// nothing to release
		}
	}
}

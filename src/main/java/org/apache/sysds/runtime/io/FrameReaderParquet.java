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
import java.io.InputStream;
import java.util.Arrays;
import java.util.Comparator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.ColumnReader;
import org.apache.parquet.column.impl.ColumnReadStoreImpl;
import org.apache.parquet.column.page.PageReadStore;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.FileMetaData;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.io.api.Converter;
import org.apache.parquet.io.api.GroupConverter;
import org.apache.parquet.io.api.PrimitiveConverter;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName;
import org.apache.parquet.schema.Type;
import org.apache.parquet.schema.Type.Repetition;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Single-threaded frame parquet reader.
 *
 * Decodes through parquet-mr's column API ({@link ColumnReadStoreImpl}/{@link ColumnReader}) directly into
 * pre-allocated typed column arrays. The output frame is constructed from the filled arrays without copying. Columns
 * whose parquet physical type does not match the requested frame value type are converted per cell instead.
 */
public class FrameReaderParquet extends FrameReader {

	/**
	 * Reads a Parquet file from HDFS and converts it into a FrameBlock.
	 *
	 * @param fname  The HDFS file path to the Parquet file.
	 * @param schema The expected data types of the columns.
	 * @param names  The names of the columns.
	 * @param rlen   The expected number of rows.
	 * @param clen   The expected number of columns.
	 * @return A FrameBlock containing the data read from the Parquet file.
	 */
	@Override
	public FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		Configuration conf = ConfigurationManager.getCachedJobConf();
		Path path = new Path(fname);
		if(!HDFSTool.existsFileOnHDFS(path.toString()))
			throw new IOException("File does not exist on HDFS: " + fname);

		ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNames(names, clen);

		Object[] dest = new Object[(int) clen];
		for(int c = 0; c < clen; c++)
			dest[c] = ArrayFactory.allocateBacking(lschema[c], (int) rlen);

		readParquetFrameFromHDFS(path, conf, dest, lschema, lnames, rlen);

		// zero-row output stays a schema-only frame (no zero-length column arrays)
		if(rlen == 0)
			return new FrameBlock(lschema, lnames, 0);

		Array<?>[] columns = new Array<?>[(int) clen];
		for(int c = 0; c < clen; c++)
			columns[c] = ArrayFactory.create(lschema[c], dest[c]);
		return new FrameBlock(columns, lnames);
	}

	/**
	 * Reads an entire Parquet file (or directory of part files) into the pre-allocated column backing arrays. Part
	 * files, if any, are read sequentially in name order.
	 *
	 * @param path   The HDFS path to the Parquet file or directory.
	 * @param conf   The Hadoop configuration.
	 * @param dest   The per-column backing arrays to populate.
	 * @param schema The value types of the output columns.
	 * @param names  The names of the output columns.
	 * @param rlen   The expected number of rows.
	 */
	protected void readParquetFrameFromHDFS(Path path, Configuration conf, Object[] dest, ValueType[] schema,
		String[] names, long rlen) throws IOException {
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		Path[] files = IOUtilFunctions.getSequenceFilePaths(fs, path);
		Arrays.sort(files, Comparator.comparing(Path::getName));

		long off = 0;
		for(Path file : files)
			off += readSingleParquetFile(file, conf, dest, schema, names, rlen, (int) off);
		if(off != rlen)
			throw new IOException("Mismatch in row count: expected " + rlen + ", but got " + off);
	}

	/**
	 * Decodes a single Parquet file into the column backing arrays at the given row offset, one row group at a time and
	 * column-at-a-time within each row group. Thread-safe for distinct row ranges, so the parallel reader assigns each
	 * file its own offset and shares the output arrays.
	 *
	 * @param path      The HDFS path to the Parquet file.
	 * @param conf      The Hadoop configuration.
	 * @param dest      The per-column backing arrays to populate.
	 * @param schema    The value types of the output columns.
	 * @param names     The names of the output columns.
	 * @param rlen      The total number of output rows (exclusive upper bound of this file's rows).
	 * @param rowOffset The row offset of this file's first row.
	 * @return The number of rows read.
	 */
	protected int readSingleParquetFile(Path path, Configuration conf, Object[] dest, ValueType[] schema,
		String[] names, long rlen, int rowOffset) throws IOException {
		final int ncol = schema.length;
		try(ParquetFileReader reader = ParquetFileReader.open(HadoopInputFile.fromPath(path, conf))) {
			FileMetaData meta = reader.getFooter().getFileMetaData();
			MessageType parquetSchema = meta.getSchema();
			String createdBy = meta.getCreatedBy();

			// map each requested frame column (by name) to its parquet column descriptor
			final ColumnDescriptor[] descs = new ColumnDescriptor[ncol];
			for(int c = 0; c < ncol; c++)
				descs[c] = validateDecodable(parquetSchema, names[c]);

			GroupConverter root = dummyConverter(parquetSchema.getFieldCount());
			int off = rowOffset;
			PageReadStore pages;
			while((pages = reader.readNextRowGroup()) != null) {
				int nrow = (int) pages.getRowCount();
				if(off + nrow > rlen)
					throw new IOException(
						"Mismatch in row count: expected " + rlen + ", but got at least " + (off + nrow));
				ColumnReadStoreImpl store = new ColumnReadStoreImpl(pages, root, parquetSchema, createdBy);
				for(int c = 0; c < ncol; c++) {
					ColumnReader creader = store.getColumnReader(descs[c]);
					int maxDef = descs[c].getMaxDefinitionLevel();
					PrimitiveTypeName ptype = descs[c].getPrimitiveType().getPrimitiveTypeName();
					if(directDecodable(ptype, schema[c]))
						decodeColumnInto(creader, maxDef, nrow, ptype, dest[c], off);
					else
						decodeColumnConvert(creader, maxDef, nrow, ptype, schema[c], dest[c], off);
				}
				off += nrow;
			}
			return off - rowOffset;
		}
	}

	/**
	 * Resolve the descriptor of one parquet column and verify it is decodable: a non-nested primitive of a physical
	 * type the reader supports (INT96 timestamps and nested/repeated groups are not).
	 */
	private static ColumnDescriptor validateDecodable(MessageType parquetSchema, String name) throws IOException {
		if(!parquetSchema.containsField(name))
			throw new IOException("Column not found in Parquet schema: " + name);
		Type t = parquetSchema.getType(name);
		if(!t.isPrimitive() || t.isRepetition(Repetition.REPEATED))
			throw new IOException("Nested Parquet columns are not supported: " + name);
		PrimitiveTypeName ptype = t.asPrimitiveType().getPrimitiveTypeName();
		switch(ptype) {
			case INT32:
			case INT64:
			case FLOAT:
			case DOUBLE:
			case BOOLEAN:
			case BINARY:
				return parquetSchema.getColumnDescription(new String[] {name});
			default:
				throw new IOException("Unsupported Parquet type " + ptype + " for column: " + name
					+ (ptype == PrimitiveTypeName.INT96 ? " (deprecated INT96 timestamps; re-encode as INT64)" : ""));
		}
	}

	/**
	 * Whether the parquet physical type is the one the frame value type's backing array stores, i.e. the column can be
	 * decoded through the typed direct path without per-value conversion.
	 */
	private static boolean directDecodable(PrimitiveTypeName ptype, ValueType vt) {
		switch(ptype) {
			case INT32:
				return vt == ValueType.INT32;
			case INT64:
				return vt == ValueType.INT64;
			case FLOAT:
				return vt == ValueType.FP32;
			case DOUBLE:
				return vt == ValueType.FP64;
			case BOOLEAN:
				return vt == ValueType.BOOLEAN;
			default:
				return vt == ValueType.STRING; // BINARY
		}
	}

	/**
	 * Decode one parquet column of the current row group into a pre-allocated typed array at the given offset. Null
	 * cells (definition level below max) keep the array default (0 for numerics, null for strings).
	 */
	private static void decodeColumnInto(ColumnReader creader, int maxDef, int nrow, PrimitiveTypeName ptype,
		Object dest, int off) {
		final int end = off + nrow;
		switch(ptype) {
			case INT32: {
				int[] a = (int[]) dest;
				for(int r = off; r < end; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[r] = creader.getInteger();
					creader.consume();
				}
				break;
			}
			case INT64: {
				long[] a = (long[]) dest;
				for(int r = off; r < end; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[r] = creader.getLong();
					creader.consume();
				}
				break;
			}
			case FLOAT: {
				float[] a = (float[]) dest;
				for(int r = off; r < end; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[r] = creader.getFloat();
					creader.consume();
				}
				break;
			}
			case DOUBLE: {
				double[] a = (double[]) dest;
				for(int r = off; r < end; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[r] = creader.getDouble();
					creader.consume();
				}
				break;
			}
			case BOOLEAN: {
				boolean[] a = (boolean[]) dest;
				for(int r = off; r < end; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[r] = creader.getBoolean();
					creader.consume();
				}
				break;
			}
			default: { // BINARY
				String[] a = (String[]) dest;
				for(int r = off; r < end; r++) {
					if(creader.getCurrentDefinitionLevel() == maxDef)
						a[r] = creader.getBinary().toStringUsingUTF8();
					creader.consume();
				}
				break;
			}
		}
	}

	/**
	 * Decode one parquet column whose physical type does not match the frame value type, converting each value per cell
	 * (same semantics as {@link FrameBlock#set(int, int, Object)}, e.g. a DOUBLE file column read into a STRING frame
	 * column).
	 */
	private static void decodeColumnConvert(ColumnReader creader, int maxDef, int nrow, PrimitiveTypeName ptype,
		ValueType vt, Object dest, int off) {
		final int end = off + nrow;
		for(int r = off; r < end; r++) {
			if(creader.getCurrentDefinitionLevel() == maxDef)
				setConverted(dest, vt, r, readValue(creader, ptype));
			creader.consume();
		}
	}

	private static Object readValue(ColumnReader creader, PrimitiveTypeName ptype) {
		switch(ptype) {
			case INT32:
				return creader.getInteger();
			case INT64:
				return creader.getLong();
			case FLOAT:
				return creader.getFloat();
			case DOUBLE:
				return creader.getDouble();
			case BOOLEAN:
				return creader.getBoolean();
			default:
				return creader.getBinary().toStringUsingUTF8(); // BINARY
		}
	}

	/**
	 * Store one converted value into the backing array of the given value type.
	 */
	private static void setConverted(Object dest, ValueType vt, int r, Object val) {
		Object converted = UtilFunctions.objectToObject(vt, val);
		if(converted == null)
			return;
		switch(vt) {
			case FP64:
				((double[]) dest)[r] = (Double) converted;
				break;
			case FP32:
				((float[]) dest)[r] = (Float) converted;
				break;
			case INT64:
			case HASH64:
				((long[]) dest)[r] = (Long) converted;
				break;
			case UINT4:
			case UINT8:
			case INT32:
			case HASH32:
				((int[]) dest)[r] = (Integer) converted;
				break;
			case BOOLEAN:
				((boolean[]) dest)[r] = (Boolean) converted;
				break;
			case CHARACTER:
				((char[]) dest)[r] = (Character) converted;
				break;
			default:
				((String[]) dest)[r] = (String) converted;
				break;
		}
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

	//not implemented
	@Override
	public FrameBlock readFrameFromInputStream(InputStream is, ValueType[] schema, String[] names, long rlen, long clen)
			throws IOException, DMLRuntimeException {
		throw new UnsupportedOperationException("Unimplemented method 'readFrameFromInputStream'");
	}
}

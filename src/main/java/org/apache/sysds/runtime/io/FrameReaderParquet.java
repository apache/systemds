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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.TimeUnit;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.ColumnReader;
import org.apache.parquet.column.impl.ColumnReadStoreImpl;
import org.apache.parquet.column.page.PageReadStore;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.io.api.Converter;
import org.apache.parquet.io.api.GroupConverter;
import org.apache.parquet.io.api.PrimitiveConverter;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.HDFSTool;

/**
 * Single-threaded frame parquet reader.
 * 
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
	public FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names, long rlen, long clen) throws IOException, DMLRuntimeException {
		// Prepare file access
		Configuration conf = ConfigurationManager.getCachedJobConf();
		Path path = new Path(fname);

		// Check existence and non-empty file
		if (!HDFSTool.existsFileOnHDFS(path.toString())) {
			throw new IOException("File does not exist on HDFS: " + fname);
		}

		// Allocate output frame block
		ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNames(names, clen);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);

		// Read Parquet file
		readParquetFrameFromHDFS(path, conf, ret, lschema, rlen, clen);

		return ret;
	}

	/**
	 * Reads data from a Parquet file on HDFS and fills the provided FrameBlock.
	 * The method retrieves the Parquet schema from the file footer, maps the required column names
	 * to their corresponding indices, and then uses the column API to iterate over each column.
	 * Data is extracted based on the column type and set into the output FrameBlock.
	 *
	 * @param path   The HDFS path to the Parquet file.
	 * @param conf   The Hadoop configuration.
	 * @param dest   The FrameBlock to populate with data.
	 * @param schema The expected value types for the output columns.
	 * @param rlen   The expected number of rows.
	 * @param clen   The expected number of columns.
	 */
	protected void readParquetFrameFromHDFS(Path path, Configuration conf, FrameBlock dest, ValueType[] schema, long rlen, long clen) throws IOException {
		int row = readSingleParquetFile(path, conf, dest, clen, 0);

		// Check frame dimensions
		if (row != rlen) {
			throw new IOException("Mismatch in row count: expected " + rlen + ", but got " + row);
		}
	}

	// Constants for decoding legacy INT96 timestamps
	private static final int JULIAN_EPOCH_OFFSET_DAYS = 2_440_588;
	private static final long MILLIS_IN_DAY = TimeUnit.DAYS.toMillis(1);
	private static final long NANOS_PER_MILLISECOND = TimeUnit.MILLISECONDS.toNanos(1);

	/**
	 * Reads a single Parquet file into the destination FrameBlock using the column API.
	 * Iterates row groups; within each row group reads each requested column and writes the
	 * value into the FrameBlock.
	 *
	 * @param path      The HDFS path to the Parquet file.
	 * @param conf      The Hadoop configuration.
	 * @param dest      The FrameBlock to populate.
	 * @param clen      The number of columns.
	 * @param rowOffset The starting row offset in the destination FrameBlock.
	 * @return The number of rows read.
	 */
	protected int readSingleParquetFile(Path path, Configuration conf, FrameBlock dest,
		long clen, long rowOffset) throws IOException
	{
		String[] columnNames = dest.getColumnNames();

		try (ParquetFileReader reader = ParquetFileReader.open(HadoopInputFile.fromPath(path, conf))) {
			ParquetMetadata metadata = reader.getFooter();
			MessageType parquetSchema = metadata.getFileMetaData().getSchema();
			String createdBy = metadata.getFileMetaData().getCreatedBy();

			// Map each requested frame column (by name) to its Parquet column descriptor.
			ColumnDescriptor[] descriptors = new ColumnDescriptor[(int) clen];
			for (int col = 0; col < clen; col++) {
				ColumnDescriptor desc = parquetSchema.getColumnDescription(new String[]{ columnNames[col] });
				// Nested columns cannot be represented by a flat FrameBlock column
				if (desc.getMaxRepetitionLevel() > 0)
					throw new IOException("Nested Parquet columns are not supported: " + columnNames[col]);
				descriptors[col] = desc;
			}

			// no-op converter tree, only used by ColumnReadStoreImpl to resolve a converter per column
			GroupConverter rootConverter = newNoOpRootConverter(parquetSchema);

			int row = (int) rowOffset;
			PageReadStore pages;

			while (true) {
				pages = reader.readNextRowGroup();

				if (pages == null) 
					break;

				int rowsInGroup = (int) pages.getRowCount();
				ColumnReadStoreImpl colStore = new ColumnReadStoreImpl(pages, rootConverter, parquetSchema, createdBy);

				for (int col = 0; col < clen; col++) {
					ColumnDescriptor desc = descriptors[col];
					ColumnReader creader = colStore.getColumnReader(desc);
					int maxDef = desc.getMaxDefinitionLevel();
					PrimitiveType.PrimitiveTypeName ptype = desc.getPrimitiveType().getPrimitiveTypeName();
					readColumn(dest, creader, col, row, rowsInGroup, maxDef, ptype);
				}
				row += rowsInGroup;
			}
			return row - (int) rowOffset;
		}
	}

	/**
	 * Reads one column of a row group, writing each value (or null) into the destination FrameBlock.
	 */
	private void readColumn(FrameBlock dest, ColumnReader creader, int col, int rowStart,
		int rowsInGroup, int maxDef, PrimitiveType.PrimitiveTypeName ptype) throws IOException
	{
		for (int i = 0; i < rowsInGroup; i++) {
			int row = rowStart + i;
			if (creader.getCurrentDefinitionLevel() == maxDef) {
				switch (ptype) {
					case INT32:
						dest.set(row, col, creader.getInteger());
						break;
					case INT64:
						dest.set(row, col, creader.getLong());
						break;
					case FLOAT:
						dest.set(row, col, creader.getFloat());
						break;
					case DOUBLE:
						dest.set(row, col, creader.getDouble());
						break;
					case BOOLEAN:
						dest.set(row, col, creader.getBoolean());
						break;
					case INT96: {
						// Legacy INT96 timestamp, narrowed to epoch millis.
						// See https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#timestamp
						Binary binary = creader.getBinary();
						ByteBuffer buf = ByteBuffer.wrap(binary.getBytes()).order(ByteOrder.LITTLE_ENDIAN);
						long nanosOfDay = buf.getLong();
						int julianDay = buf.getInt();
						long millis = (julianDay - JULIAN_EPOCH_OFFSET_DAYS) * MILLIS_IN_DAY
							+ nanosOfDay / NANOS_PER_MILLISECOND;
						dest.set(row, col, millis);
						break;
					}
					case BINARY:
						dest.set(row, col, creader.getBinary().toStringUsingUTF8());
						break;
					default:
						throw new IOException("Unsupported data type: " + ptype);
				}
			}
			else {
				dest.set(row, col, null);
			}
			creader.consume();
		}
	}

	/**
	 * Builds a no-op converter tree matching the (flat) Parquet schema. The converter
	 * callbacks are never invoked because values are read through the typed creader.getX accessors in readColumn().
	 * The tree merely serves to satisfy the ColumnReadStoreImpl constructor
	 */
	private static GroupConverter newNoOpRootConverter(MessageType schema) {
		final int n = schema.getFieldCount();
		final PrimitiveConverter[] leaves = new PrimitiveConverter[n];
		for (int i = 0; i < n; i++)
			leaves[i] = new PrimitiveConverter() {};
		return new GroupConverter() {
			@Override public Converter getConverter(int fieldIndex) { return leaves[fieldIndex]; }
			@Override public void start() {}
			@Override public void end() {}
		};
	}

	//not implemented
	@Override
	public FrameBlock readFrameFromInputStream(InputStream is, ValueType[] schema, String[] names, long rlen, long clen)
			throws IOException, DMLRuntimeException {
		throw new UnsupportedOperationException("Unimplemented method 'readFrameFromInputStream'");
	}
}
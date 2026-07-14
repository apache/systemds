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
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.parquet.hadoop.ParquetOutputFormat;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.api.WriteSupport;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.io.api.RecordConsumer;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName;
import org.apache.parquet.schema.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.common.Types.ValueType;

/**
 * Single-threaded frame parquet writer.
 * 
 */
public class FrameWriterParquet extends FrameWriter {

	public enum DictEncoding {
		ALL_ON, ALL_OFF, STRING_ONLY
	}

	private final CompressionCodecName codec;
	private final DictEncoding dictEncoding;
	private final long rowGroupSize;

	public FrameWriterParquet() {
		this(CompressionCodecName.ZSTD, DictEncoding.STRING_ONLY, ParquetWriter.DEFAULT_BLOCK_SIZE);
	}

	public FrameWriterParquet(CompressionCodecName codec, DictEncoding dictEncoding) {
		this(codec, dictEncoding, ParquetWriter.DEFAULT_BLOCK_SIZE);
	}

	public FrameWriterParquet(CompressionCodecName codec, DictEncoding dictEncoding, long rowGroupSize) {
		this.codec = codec;
		this.dictEncoding = dictEncoding;
		this.rowGroupSize = rowGroupSize;
	}

	/**
	 * Writes a FrameBlock to a Parquet file on HDFS.
	 *
	 * @param src   The FrameBlock containing the data to write.
	 * @param fname The HDFS file path where the Parquet file will be stored.
	 * @param rlen  The expected number of rows.
	 * @param clen  The expected number of columns.
	 */
	@Override
	public final void writeFrameToHDFS(FrameBlock src, String fname, long rlen, long clen) throws IOException, DMLRuntimeException {
		// Prepare file access
		JobConf conf = ConfigurationManager.getCachedJobConf();
		Path path = new Path(fname);

		// If the file already exists on HDFS, remove it
		HDFSTool.deleteFileIfExistOnHDFS(path, conf);
		
		// Check frame dimensions
		if (src.getNumRows() != rlen || src.getNumColumns() != clen) {
			throw new IOException("Frame dimensions mismatch with metadata: " + src.getNumRows() + "x" + src.getNumColumns() + " vs " + rlen + "x" + clen + ".");
		}

		// Write parquet file
		writeParquetFrameToHDFS(path, conf, src);
	}

	/**
	 * Writes the FrameBlock data to a Parquet file using a ParquetWriter. The method generates a Parquet schema based
	 * on the metadata of the FrameBlock, initializes a ParquetWriter with specified configurations, iterates over each
	 * row and column, writing directly to the RecordConsumer, using type-specific conversions.
	 *
	 * @param path The HDFS path where the Parquet file will be written.
	 * @param conf The Hadoop configuration.
	 * @param src  The FrameBlock containing the data to write.
	 */
	protected void writeParquetFrameToHDFS(Path path, Configuration conf, FrameBlock src) 
		throws IOException 
	{
		FileSystem fs = IOUtilFunctions.getFileSystem(path, conf);

		// Create schema based on frame block metadata
		MessageType schema = createParquetSchema(src);

		String[] columnNames = src.getColumnNames();
		ValueType[] columnTypes = src.getSchema();

		FrameParquetWriterBuilder writerBuilder = new FrameParquetWriterBuilder(path, schema, src).withConf(conf)
			.withCompressionCodec(
				CompressionCodecName.fromConf(conf.get(ParquetOutputFormat.COMPRESSION, codec.name())))
			.withRowGroupSize(conf.getLong(ParquetOutputFormat.BLOCK_SIZE, rowGroupSize))
			.withPageSize(conf.getInt(ParquetOutputFormat.PAGE_SIZE, ParquetWriter.DEFAULT_PAGE_SIZE))
			.withDictionaryPageSize(
				conf.getInt(ParquetOutputFormat.DICTIONARY_PAGE_SIZE, ParquetWriter.DEFAULT_PAGE_SIZE))
			.withDictionaryEncoding(
				conf.getBoolean(ParquetOutputFormat.ENABLE_DICTIONARY, dictEncoding == DictEncoding.ALL_ON));

		if(dictEncoding == DictEncoding.STRING_ONLY)
			for(int j = 0; j < src.getNumColumns(); j++)
				if(columnTypes[j] == ValueType.STRING)
					writerBuilder = writerBuilder.withDictionaryEncoding(columnNames[j], true);

		try(ParquetWriter<Integer> writer = writerBuilder.build()) {
			for(int i = 0; i < src.getNumRows(); i++)
				writer.write(i);
		}
		
		// Delete CRC files created by Hadoop if necessary
		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	/**
	 * Creates a Parquet schema based on the metadata of a FrameBlock.
	 *
	 * @param src The FrameBlock whose metadata is used to create the Parquet schema.
	 * @return The generated Parquet MessageType schema.
	 */
	protected MessageType createParquetSchema(FrameBlock src) {
		String[] columnNames = src.getColumnNames();
		ValueType[] columnTypes = src.getSchema();
		Types.MessageTypeBuilder builder = Types.buildMessage();

		for (int i = 0; i < src.getNumColumns(); i++) {
			switch (columnTypes[i]) {
				case STRING:
					builder.optional(PrimitiveTypeName.BINARY).as(LogicalTypeAnnotation.stringType())
						.named(columnNames[i]);
					break;
				case INT32:
					builder.optional(PrimitiveTypeName.INT32).named(columnNames[i]);
					break;
				case INT64:
					builder.optional(PrimitiveTypeName.INT64).named(columnNames[i]);
					break;
				case FP32:
					builder.optional(PrimitiveTypeName.FLOAT).named(columnNames[i]);
					break;
				case FP64:
					builder.optional(PrimitiveTypeName.DOUBLE).named(columnNames[i]);
					break;
				case BOOLEAN:
					builder.optional(PrimitiveTypeName.BOOLEAN).named(columnNames[i]);
					break;
				default:
					throw new IllegalArgumentException("Unsupported data type: " + columnTypes[i]);
			}
		}
		return builder.named("FrameSchema");
	}

	/**
	 * WriteSupport implementation that writes rows from a FrameBlock directly to the Parquet RecordConsumer.
	 */
	private static class FrameWriteSupport extends WriteSupport<Integer> {
		private final MessageType schema;
		private final FrameBlock src;
		private RecordConsumer recordConsumer;
		// constant across all rows
		private String[] colNames;
		private ValueType[] colTypes;
		private int numCols;

		FrameWriteSupport(MessageType schema, FrameBlock src) {
			this.schema = schema;
			this.src = src;
		}

		@Override
		public WriteContext init(Configuration configuration) {
			Map<String, String> metadata = new HashMap<>();
			return new WriteContext(schema, metadata);
		}

		@Override
		public void prepareForWrite(RecordConsumer consumer) {
			this.recordConsumer = consumer;
			this.colNames = src.getColumnNames();
			this.colTypes = src.getSchema();
			this.numCols = src.getNumColumns();
		}

		@Override
		public void write(Integer rowIndex) {
			recordConsumer.startMessage();
			for(int j = 0; j < numCols; j++) {
				Object value = src.get(rowIndex, j);
				if(value != null) {
					recordConsumer.startField(colNames[j], j);
					switch(colTypes[j]) {
						case STRING:
							recordConsumer.addBinary(Binary.fromString(value.toString()));
							break;
						case INT32:
							recordConsumer.addInteger((int) value);
							break;
						case INT64:
							recordConsumer.addLong((long) value);
							break;
						case FP32:
							recordConsumer.addFloat((float) value);
							break;
						case FP64:
							recordConsumer.addDouble((double) value);
							break;
						case BOOLEAN:
							recordConsumer.addBoolean((boolean) value);
							break;
						default:
							throw new IllegalArgumentException("Unsupported value type: " + colTypes[j]);
					}
					recordConsumer.endField(colNames[j], j);
				}
			}
			recordConsumer.endMessage();
		}
	}

	/**
	 * ParquetWriter builder wired to FrameWriteSupport.
	 */
	private static class FrameParquetWriterBuilder extends ParquetWriter.Builder<Integer, FrameParquetWriterBuilder> {
		private final MessageType schema;
		private final FrameBlock src;

		FrameParquetWriterBuilder(Path path, MessageType schema, FrameBlock src) {
			super(path);
			this.schema = schema;
			this.src = src;
		}

		@Override
		protected FrameParquetWriterBuilder self() {
			return this;
		}

		@Override
		protected WriteSupport<Integer> getWriteSupport(Configuration conf) {
			return new FrameWriteSupport(schema, src);
		}
	}
}

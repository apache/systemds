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

	protected void writeParquetFrameToHDFS(Path path, Configuration conf, FrameBlock src) throws IOException{
		writeParquetFrameToHDFS(path, conf, src, 0, src.getNumRows());
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
	 * @param startRow The starting row index for the write operation.
	 * @param endRow The ending row index for the write operation.
	 */
	protected void writeParquetFrameToHDFS(Path path, Configuration conf, FrameBlock src, int startRow, int endRow) throws IOException {
		if(startRow < 0 || endRow < startRow || endRow > src.getNumRows())
			throw new IOException("Invalid row range for Parquet write: " + startRow + " to " + endRow);
		
		FileSystem fs = IOUtilFunctions.getFileSystem(path, conf);

		// Create schema based on frame block metadata
		MessageType schema = createParquetSchema(src);

		// TODO:Experiment with different batch sizes?
		//int batchSize = 1000;  
		//int rowCount = 0;

		// Write data using ParquetWriter //FIXME replace example writer? 
		try (ParquetWriter<Group> writer = ExampleParquetWriter.builder(path)
				.withConf(conf)
				.withType(schema)
				.withCompressionCodec(ParquetWriter.DEFAULT_COMPRESSION_CODEC_NAME)
				.withRowGroupSize((long) ParquetWriter.DEFAULT_BLOCK_SIZE)
				.withPageSize(ParquetWriter.DEFAULT_PAGE_SIZE)
				.withDictionaryEncoding(true)
				.build()) 
		{
			final int numCols = src.getNumColumns();
			final String[] columnNames = src.getColumnNames();
			final ValueType[] schemaTypes = src.getSchema();

			SimpleGroupFactory groupFactory = new SimpleGroupFactory(schema);
			
			//List<Group> rowBuffer = new ArrayList<>(batchSize);
			
			for (int i = startRow; i < endRow; i++) {
				Group group = groupFactory.newGroup();
				for (int j = 0; j < numCols; j++) {
					Object value = src.get(i, j);
					if (value != null) {
						ValueType type = schemaTypes[j];
						switch (type) {
							case STRING:
								group.add(columnNames[j], value.toString());
								break;
							case INT32:
								group.add(columnNames[j], (int) value);
								break;
							case INT64:
								group.add(columnNames[j], (long) value);
								break;
							case FP32:
								group.add(columnNames[j], (float) value);
								break;
							case FP64:
								group.add(columnNames[j], (double) value);
								break;
							case BOOLEAN:
								group.add(columnNames[j], (boolean) value);
								break;
							default:
								throw new IOException("Unsupported value type: " + type);
						}
					}
				}

				writer.write(group);
			}
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
		schemaBuilder.append("}");
		return MessageTypeParser.parseMessageType(schemaBuilder.toString());
	}
}
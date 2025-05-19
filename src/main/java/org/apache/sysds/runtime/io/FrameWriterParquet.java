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
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;
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
	 * Writes the FrameBlock data to a Parquet file using a ParquetWriter.
	 * The method generates a Parquet schema based on the metadata of the FrameBlock, initializes a ParquetWriter with specified configurations, 
	 * iterates over each row and column, adding values (in batches for improved performance) using type-specific conversions.
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

		// TODO:Experiment with different batch sizes?
		int batchSize = 1000;  
		int rowCount = 0;

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

			SimpleGroupFactory groupFactory = new SimpleGroupFactory(schema);
			
			List<Group> rowBuffer = new ArrayList<>(batchSize);
			
			for (int i = 0; i < src.getNumRows(); i++) {
				Group group = groupFactory.newGroup();
				for (int j = 0; j < src.getNumColumns(); j++) {
					Object value = src.get(i, j);
					if (value != null) {
						ValueType type = src.getSchema()[j];
						switch (type) {
							case STRING:
								group.add(src.getColumnNames()[j], value.toString());
								break;
							case INT32:
								group.add(src.getColumnNames()[j], (int) value);
								break;
							case INT64:
								group.add(src.getColumnNames()[j], (long) value);
								break;
							case FP32:
								group.add(src.getColumnNames()[j], (float) value);
								break;
							case FP64:
								group.add(src.getColumnNames()[j], (double) value);
								break;
							case BOOLEAN:
								group.add(src.getColumnNames()[j], (boolean) value);
								break;
							default:
								throw new IOException("Unsupported value type: " + type);
						}
					}
				}
				rowBuffer.add(group);
				rowCount++;

				if (rowCount >= batchSize) {
					for (Group g : rowBuffer) {
						writer.write(g);
					}
					rowBuffer.clear();
					rowCount = 0;
				}
			}
			
			for (Group g : rowBuffer) {
				writer.write(g);
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
		StringBuilder schemaBuilder = new StringBuilder("message FrameSchema {");
		String[] columnNames = src.getColumnNames();
		ValueType[] columnTypes = src.getSchema();

		for (int i = 0; i < src.getNumColumns(); i++) {
			schemaBuilder.append("optional ");
			switch (columnTypes[i]) {
				case STRING:
					schemaBuilder.append("binary ").append(columnNames[i]).append(" (UTF8);");
					break;
				case INT32:
					schemaBuilder.append("int32 ").append(columnNames[i]).append(";");
					break;
				case INT64:
					schemaBuilder.append("int64 ").append(columnNames[i]).append(";");
					break;
				case FP32:
					schemaBuilder.append("float ").append(columnNames[i]).append(";");
					break;
				case FP64:
					schemaBuilder.append("double ").append(columnNames[i]).append(";");
					break;
				case BOOLEAN:
					schemaBuilder.append("boolean ").append(columnNames[i]).append(";");
					break;
				default:
					throw new IllegalArgumentException("Unsupported data type: " + columnTypes[i]);
			}
		}
		schemaBuilder.append("}");
		return MessageTypeParser.parseMessageType(schemaBuilder.toString());
	}
}

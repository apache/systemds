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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.example.GroupReadSupport;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.hadoop.util.HadoopInputFile;
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
	 * to their corresponding indices, and then uses a ParquetReader to iterate over each row.
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
		// Retrieve schema from Parquet footer
		ParquetMetadata metadata = ParquetFileReader.open(HadoopInputFile.fromPath(path, conf)).getFooter();
		MessageType parquetSchema = metadata.getFileMetaData().getSchema();

		// Map column names to Parquet schema indices
		String[] columnNames = dest.getColumnNames();
		int[] columnIndices = new int[columnNames.length];
		for (int i = 0; i < columnNames.length; i++) {
			columnIndices[i] = parquetSchema.getFieldIndex(columnNames[i]);
		}

		// Read data usind ParquetReader
		try (ParquetReader<Group> rowReader = ParquetReader.builder(new GroupReadSupport(), path)
				.withConf(conf)
				.build()) {

			Group group;
			int row = 0;
			while ((group = rowReader.read()) != null) {
				for (int col = 0; col < clen; col++) {
					int colIndex = columnIndices[col];
					if (group.getFieldRepetitionCount(colIndex) > 0) {
						PrimitiveType.PrimitiveTypeName type = parquetSchema.getType(columnNames[col]).asPrimitiveType().getPrimitiveTypeName();
						switch (type) {
							case INT32:
								dest.set(row, col, group.getInteger(colIndex, 0));
								break;
							case INT64:
								dest.set(row, col, group.getLong(colIndex, 0));
								break;
							case FLOAT:
								dest.set(row, col, group.getFloat(colIndex, 0));
								break;
							case DOUBLE:
								dest.set(row, col, group.getDouble(colIndex, 0));
								break;
							case BOOLEAN:
								dest.set(row, col, group.getBoolean(colIndex, 0));
								break;
							case BINARY:
								dest.set(row, col, group.getBinary(colIndex, 0).toStringUsingUTF8());
								break;
							default:
								throw new IOException("Unsupported data type: " + type);
						}
					} else {
						dest.set(row, col, null);
					}
				}
				row++;
			}

			// Check frame dimensions
			if (row != rlen) {
				throw new IOException("Mismatch in row count: expected " + rlen + ", but got " + row);
			}
		}
	}

	//not implemented
	@Override
	public FrameBlock readFrameFromInputStream(InputStream is, ValueType[] schema, String[] names, long rlen, long clen)
			throws IOException, DMLRuntimeException {
		throw new UnsupportedOperationException("Unimplemented method 'readFrameFromInputStream'");
	}
}
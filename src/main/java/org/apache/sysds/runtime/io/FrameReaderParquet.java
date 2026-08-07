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
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.example.GroupReadSupport;
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

	protected PrimitiveType.PrimitiveTypeName[] getParquetColumnTypes(MessageType parquetSchema, int[] columnIndices) {
		PrimitiveType.PrimitiveTypeName[] columnTypes = new PrimitiveType.PrimitiveTypeName[columnIndices.length];
		for(int i = 0; i < columnIndices.length; i++)
			columnTypes[i] = parquetSchema.getType(columnIndices[i]).asPrimitiveType().getPrimitiveTypeName();
		return columnTypes;
	}

	protected int[] getParquetColumnIndices(MessageType parquetSchema, String[] columnNames) {
		int[] columnIndices = new int[columnNames.length];
		for (int i = 0; i < columnNames.length; i++) {
			columnIndices[i] = parquetSchema.getFieldIndex(columnNames[i]);
		}
		return columnIndices;
	}

	protected Object readTypedParquetValue(Group group, PrimitiveType.PrimitiveTypeName type, int columnIndex) throws IOException {
		if (group.getFieldRepetitionCount(columnIndex) == 0) {
			return null;
		}

		switch (type) {
			case INT32:
				return group.getInteger(columnIndex, 0);
			case INT64:
				return group.getLong(columnIndex, 0);
			case FLOAT:
				return group.getFloat(columnIndex, 0);
			case DOUBLE:
				return group.getDouble(columnIndex, 0);
			case BOOLEAN:
				return group.getBoolean(columnIndex, 0);
			case BINARY:
				return group.getBinary(columnIndex, 0).toStringUsingUTF8();
			default:
				throw new IOException("Unsupported data type: " + type);
		}
	}

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

		// Check existence
		if (!HDFSTool.existsFileOnHDFS(path.toString())) {
			throw new IOException("File does not exist on HDFS: " + fname);

		ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNames(names, clen);

		Object[] dest = new Object[(int) clen];
		for(int c = 0; c < clen; c++)
			dest[c] = ArrayFactory.allocateBacking(lschema[c], (int) rlen);

		// Read Parquet file
		readParquetFrameFromHDFS(path, conf, ret, rlen, clen);

		return ret;
	}

	/**
	 * Reads data from a Parquet file on HDFS and fills the provided FrameBlock.
	 * The method retrieves the Parquet schema from the file footer, maps the required column names
	 * to their corresponding indices, and then uses a ParquetReader to iterate over each row.
	 * Data is extracted based on the column type and set into the output FrameBlock.
	 *
	 * @param path   The HDFS path to the Parquet file or directory.
	 * @param conf   The Hadoop configuration.
	 * @param dest   The FrameBlock to populate with data.
	 * @param rlen   The expected number of rows.
	 * @param clen   The expected number of columns.
	 */
	protected void readParquetFrameFromHDFS(Path path, Configuration conf, FrameBlock dest, long rlen, long clen) throws IOException {
		// Retrieve schema from Parquet footer
		MessageType parquetSchema;
		try (ParquetFileReader fileReader = ParquetFileReader.open(HadoopInputFile.fromPath(path, conf))) {
			parquetSchema = fileReader.getFooter().getFileMetaData().getSchema();
		}

		// Map column names to Parquet schema indices
		String[] columnNames = dest.getColumnNames();
		int[] columnIndices = getParquetColumnIndices(parquetSchema, columnNames);
		PrimitiveType.PrimitiveTypeName[] columnTypes = getParquetColumnTypes(parquetSchema, columnIndices);

		// Read data using ParquetReader
		try (ParquetReader<Group> rowReader = ParquetReader.builder(new GroupReadSupport(), path)
				.withConf(conf)
				.build()) {

			Group group;
			int row = 0;
			while ((group = rowReader.read()) != null) {
				if(row >= rlen)
					throw new IOException("Mismatch in row count: expected " + rlen + ", but got more rows.");
				
				for (int col = 0; col < clen; col++) {
					int colIndex = columnIndices[col];
					dest.set(row, col, readTypedParquetValue(group, columnTypes[col], colIndex));
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

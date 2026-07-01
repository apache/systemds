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

package org.apache.sysds.test.functions.io.parquet;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;

class ParquetTestUtils {

	static class ParquetMetadataInfo {
		String[] names;
		ValueType[] schema;
		long rlen;
		long clen;
	}

	static ParquetMetadataInfo inferMetadata(String fname) throws IOException {
		Configuration conf = ConfigurationManager.getCachedJobConf();
		Path path = new Path(fname);

		ParquetMetadata metadata;
		try (ParquetFileReader r = ParquetFileReader.open(HadoopInputFile.fromPath(path, conf))) {
			metadata = r.getFooter();
		}
		MessageType parquetSchema = metadata.getFileMetaData().getSchema();

		int fieldCount = parquetSchema.getFieldCount();
		String[] names = new String[fieldCount];
		ValueType[] schema = new ValueType[fieldCount];

		for (int i = 0; i < fieldCount; i++) {
			names[i] = parquetSchema.getFieldName(i);
			PrimitiveType.PrimitiveTypeName type = parquetSchema.getType(i).asPrimitiveType().getPrimitiveTypeName();
			switch (type) {
				case INT32:    schema[i] = ValueType.INT32;    break;
				case INT64:    schema[i] = ValueType.INT64;    break;
				case FLOAT:    schema[i] = ValueType.FP32;     break;
				case DOUBLE:   schema[i] = ValueType.FP64;     break;
				case BOOLEAN:  schema[i] = ValueType.BOOLEAN;  break;
				case BINARY:   schema[i] = ValueType.STRING;   break;
				case INT96:    schema[i] = ValueType.INT64;    break;
				default:
					throw new IOException("Unsupported parquet type: " + type + " in column " + names[i]);
			}
		}

		long rlen = 0;
		for (BlockMetaData block : metadata.getBlocks())
			rlen += block.getRowCount();

		ParquetMetadataInfo info = new ParquetMetadataInfo();
		info.names = names;
		info.schema = schema;
		info.rlen = rlen;
		info.clen = fieldCount;
		return info;
	}
}

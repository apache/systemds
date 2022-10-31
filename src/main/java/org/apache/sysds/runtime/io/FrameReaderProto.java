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

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.protobuf.SysdsProtos;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class FrameReaderProto extends FrameReader {
	@Override
	public FrameBlock readFrameFromHDFS(String fname, Types.ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException {
		// prepare file access
		JobConf jobConf = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fileSystem = IOUtilFunctions.getFileSystem(path, jobConf);
		FileInputFormat.addInputPath(jobConf, path);

		// check existence and non-empty file
		checkValidInputFile(fileSystem, path);

		Types.ValueType[] outputSchema = createOutputSchema(schema, clen);
		String[] outputNames = createOutputNames(names, clen);
		FrameBlock outputFrameBlock = createOutputFrameBlock(outputSchema, outputNames, rlen);

		// core read (sequential/parallel)
		readProtoFrameFromHDFS(path, fileSystem, outputFrameBlock, rlen, clen);
		return outputFrameBlock;
	}

	private static void readProtoFrameFromHDFS(Path path, FileSystem fileSystem, FrameBlock dest, long rlen, long clen)
		throws IOException {
		SysdsProtos.Frame frame = readProtoFrameFromFile(path, fileSystem);
		for(int row = 0; row < rlen; row++) {
			for(int column = 0; column < clen; column++) {
				dest.set(row,
					column,
					UtilFunctions.stringToObject(Types.ValueType.STRING, frame.getRows(row).getColumnData(column)));
			}
		}
		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fileSystem, path);
	}

	private static SysdsProtos.Frame readProtoFrameFromFile(Path path, FileSystem fileSystem) throws IOException {
		FSDataInputStream fsDataInputStream = fileSystem.open(path);
		try {
			return SysdsProtos.Frame.newBuilder().mergeFrom(fsDataInputStream).build();
		}
		finally {
			IOUtilFunctions.closeSilently(fsDataInputStream);
		}
	}

	@Override
	public FrameBlock readFrameFromInputStream(InputStream is, Types.ValueType[] schema, String[] names, long rlen,
		long clen) {
		throw new DMLRuntimeException("Not implemented yet.");
	}
}

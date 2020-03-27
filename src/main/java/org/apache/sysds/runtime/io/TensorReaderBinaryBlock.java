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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.util.Arrays;

public class TensorReaderBinaryBlock extends TensorReader {
	@SuppressWarnings("resource")
	@Override
	public TensorBlock readTensorFromHDFS(String fname, long[] dims,
			int blen, ValueType[] schema) throws IOException, DMLRuntimeException {
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		//check existence and non-empty file
		checkValidInputFile(fs, path);

		//core read
		return readBinaryBlockTensorFromHDFS(path, job, fs, dims, blen, schema);
	}

	protected TensorBlock readBinaryBlockTensorFromHDFS(Path path, JobConf job, FileSystem fs, long[] dims,
			int blen, ValueType[] schema) throws IOException {
		int[] idims = Arrays.stream(dims).mapToInt(i -> (int) i).toArray();
		TensorBlock ret;
		if (schema.length == 1)
			ret = new TensorBlock(schema[0], idims).allocateBlock();
		else
			ret = new TensorBlock(schema, idims).allocateBlock();
		TensorIndexes key = new TensorIndexes();
		// TODO reuse blocks

		for (Path lpath : IOUtilFunctions.getSequenceFilePaths(fs, path)) {
			TensorBlock value = new TensorBlock();

			try (SequenceFile.Reader reader = new SequenceFile.Reader(job, SequenceFile.Reader.file(lpath))) {
				while (reader.next(key, value)) {
					if (value.isEmpty(false))
						continue;
					int[] lower = new int[dims.length];
					int[] upper = new int[lower.length];
					UtilFunctions.getBlockBounds(key, value.getLongDims(), blen, lower, upper);
					ret.copy(lower, upper, value);
				}
			}
		}
		return ret;
	}
}

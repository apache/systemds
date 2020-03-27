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
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

public class TensorWriterTextCell extends TensorWriter {
	@Override
	public void writeTensorToHDFS(TensorBlock src, String fname, int blen) throws IOException {
		int[] dims = src.getDims();
		//validity check matrix dimensions
		if (src.getNumDims() != dims.length)
			throw new IOException("Tensor number of dimensions mismatch with metadata: " + src.getNumDims() + " vs " + dims.length);
		for (int i = 0; i < dims.length; i++) {
			if (dims[i] != src.getDim(i))
				throw new IOException("Tensor dimension (" + (i + 1) + ") mismatch with metadata: " + src.getDim(i) + " vs " + dims[i]);
		}

		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		//if the file already exists on HDFS, remove it.
		HDFSTool.deleteFileIfExistOnHDFS(fname);

		//core write
		writeTextCellTensorToHDFS(path, fs, src);

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}
	
	protected void writeTextCellTensorToHDFS(Path path, FileSystem fs, TensorBlock src) throws IOException {
		writeTextCellTensorToFile(path, fs, src, 0, src.getNumRows());
	}
	
	protected static void writeTextCellTensorToFile(Path path, FileSystem fs, TensorBlock src, int rl, int ru) throws IOException {
		try (BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path, true)))) {
			int[] dims = src.getDims();
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();

			int[] ix = new int[dims.length];
			ix[0] = rl;
			for (long i = 0; i < (src.getLength() / src.getNumRows()) * (ru - rl); i++) {
				Object obj = src.get(ix);
				ValueType vt = src.isBasic() ? src.getValueType() : src.getSchema()[ix[1]];
				boolean skip;
				if( vt == ValueType.STRING )
					skip = obj == null || ((String) obj).isEmpty();
				else
					skip = UtilFunctions.objectToDouble(vt, obj) == 0.0;
				if (!skip) {
					for (int j : ix)
						sb.append(j + 1).append(' ');
					sb.append(src.get(ix)).append('\n');
					br.write(sb.toString());
					sb.setLength(0);
				}
				src.getNextIndexes(ix);
			}

			//handle empty result
			if (src.isEmpty(false)) {
				for (int i = 0; i < dims.length; i++)
					sb.append(0).append(' ');
				sb.append(0).append('\n');
				br.write(sb.toString());
			}
		}
	}
}

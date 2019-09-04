/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.runtime.io;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.util.HDFSTool;
import org.tugraz.sysds.runtime.util.UtilFunctions;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

public class TensorWriterTextCell extends TensorWriter {
	@Override
	public void writeTensorToHDFS(TensorBlock src, String fname, long[] dims, int blen) throws IOException {
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
		writeTextCellTensorToHDFS(path, fs, src, dims);

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	private static void writeTextCellTensorToHDFS(Path path, FileSystem fs, TensorBlock src,
			long[] dims) throws IOException {
		try (BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path, true)))) {
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();

			int[] ix = new int[dims.length];
			for (long i = 0; i < src.getLength(); i++) {
				Object obj = src.get(ix);
				boolean skip;
				if (!src.isBasic())
					skip = UtilFunctions.objectToDouble(src.getSchema()[ix[1]], obj) == 0.0;
				else
					skip = UtilFunctions.objectToDouble(src.getValueType(), obj) == 0.0;
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

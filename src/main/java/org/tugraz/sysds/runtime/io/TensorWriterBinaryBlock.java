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
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.data.TensorIndexes;
import org.tugraz.sysds.runtime.matrix.mapred.MRJobConfiguration;
import org.tugraz.sysds.runtime.util.HDFSTool;

import java.io.IOException;

public class TensorWriterBinaryBlock extends TensorWriter {
	//TODO replication

	@Override
	public void writeTensorToHDFS(TensorBlock src, String fname, long[] dims, int blen) throws IOException {
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		//if the file already exists on HDFS, remove it.
		HDFSTool.deleteFileIfExistOnHDFS(fname);

		//set up preferred custom serialization framework for binary block format
		if (MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION)
			MRJobConfiguration.addBinaryBlockSerializationFramework(job);

		//core write sequential
		writeBinaryBlockMatrixToHDFS(path, job, fs, src, dims, blen);

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	@SuppressWarnings("deprecation")
	private void writeBinaryBlockMatrixToHDFS(Path path, JobConf job, FileSystem fs, TensorBlock src, long[] dims,
			int blen) throws IOException {
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, TensorIndexes.class, TensorBlock.class);

		try {
			// bound check
			for (int i = 0; i < dims.length; i++) {
				if (src.getDim(i) > dims[i])
					throw new IOException("TensorBlock dimension " + i + " range [1:" + src.getDim(i) +
							"] out of range [1:" + dims[i] + "].");
			}
			long numBlocks = 1;
			for (long dim : dims) {
				numBlocks *= Math.max((long) Math.ceil((double) dim / blen), 1);
			}

			for (int i = 0; i < numBlocks; i++) {
				int[] offsets = new int[dims.length];
				long blockIndex = i;
				long[] tix = new long[dims.length];
				int[] blockDims = new int[dims.length];
				for (int j = dims.length - 1; j >= 0; j--) {
					long numDimBlocks = Math.max((long) Math.ceil((double)src.getDim(j) / blen), 1);
					tix[j] = 1 + (blockIndex % numDimBlocks);
					blockIndex /= numDimBlocks;
					offsets[j] = ((int) tix[j] - 1) * blen;
					blockDims[j] = (tix[j] * blen < src.getDim(j)) ? blen : src.getDim(j) - offsets[j];
				}
				TensorIndexes indx = new TensorIndexes(tix);
				TensorBlock block;
				if (!src.isBasic())
					block = new TensorBlock(src.getSchema(), blockDims).allocateBlock();
				else
					block = new TensorBlock(src.getValueType(), blockDims).allocateBlock();

				//copy submatrix to block
				src.slice(offsets, block);

				writer.append(indx, block);
			}
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
	}
}

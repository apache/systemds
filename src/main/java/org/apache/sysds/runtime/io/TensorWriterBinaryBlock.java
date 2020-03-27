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
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.util.HDFSTool;

import java.io.IOException;
import java.util.Arrays;

public class TensorWriterBinaryBlock extends TensorWriter {
	//TODO replication

	@Override
	public void writeTensorToHDFS(TensorBlock src, String fname, int blen) throws IOException {
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		//if the file already exists on HDFS, remove it.
		HDFSTool.deleteFileIfExistOnHDFS(fname);

		//set up preferred custom serialization framework for binary block format
		if (HDFSTool.USE_BINARYBLOCK_SERIALIZATION)
			HDFSTool.addBinaryBlockSerializationFramework(job);

		//core write sequential
		writeBinaryBlockTensorToHDFS(path, job, fs, src, blen);

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}
	
	protected void writeBinaryBlockTensorToHDFS(Path path, JobConf job, FileSystem fs, TensorBlock src,
			int blen) throws IOException {
		writeBinaryBlockTensorToSequenceFile(path, job, fs, src, blen, 0, src.getNumRows());
	}

	@SuppressWarnings("deprecation")
	protected static void writeBinaryBlockTensorToSequenceFile(Path path, JobConf job, FileSystem fs, TensorBlock src,
			int blen, int rl, int ru)
			throws IOException
	{
		try(SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, TensorIndexes.class, TensorBlock.class)) {
			int[] dims = src.getDims();
			// bound check
			for (int i = 0; i < dims.length; i++) {
				if (src.getDim(i) > dims[i])
					throw new IOException("TensorBlock dimension " + i + " range [1:" + src.getDim(i) +
							"] out of range [1:" + dims[i] + "].");
			}
			long numBlocks = Math.max((long) Math.ceil((double) (ru - rl) / blen), 1);
			for (int i = 1; i < dims.length; i++) {
				numBlocks *= Math.max((long) Math.ceil((double) dims[i] / blen), 1);
			}

			for (int i = 0; i < numBlocks; i++) {
				int[] offsets = new int[dims.length];
				long blockIndex = i;
				long[] tix = new long[dims.length];
				int[] blockDims = new int[dims.length];
				for (int j = dims.length - 1; j >= 0; j--) {
					long numDimBlocks = Math.max((long) Math.ceil((double)src.getDim(j) / blen), 1);
					tix[j] = 1 + (blockIndex % numDimBlocks);
					if (j == 0)
						tix[j] += rl / blen;
					blockIndex /= numDimBlocks;
					offsets[j] = ((int) tix[j] - 1) * blen;
					blockDims[j] = (tix[j] * blen < src.getDim(j)) ? blen : src.getDim(j) - offsets[j];
				}
				TensorIndexes indx = new TensorIndexes(tix);
				TensorBlock block;
				if( src.isBasic() )
					block = new TensorBlock(src.getValueType(), blockDims).allocateBlock();
				else {
					ValueType[] schema = src.getSchema();
					ValueType[] blockSchema = Arrays.copyOfRange(schema, offsets[1], offsets[1] + blockDims[1]);
					block = new TensorBlock(blockSchema, blockDims).allocateBlock();
				}
				
				//copy submatrix to block
				src.slice(offsets, block);

				writer.append(indx, block);
			}
		}
	}
}

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

package org.apache.sysds.runtime.compress.io;

import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import org.apache.commons.lang.NotImplementedException;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.HDFSTool;

public class WriterCompressed extends MatrixWriter {

	public static WriterCompressed create(FileFormatProperties props) {
		return new WriterCompressed();
	}

	public static void writeCompressedMatrixToHDFS(MatrixBlock src, String fname) throws IOException {
		create(null).writeMatrixToHDFS(src, fname, 0, 0, 0, 0, false);
	}

	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int blen, long nnz, boolean diag)
		throws IOException {
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		HDFSTool.deleteFileIfExistOnHDFS(fname);
		try {
			writeCompressedMatrixToHDFS(path, job, fs, src);
		}
		catch(DMLCompressionException ce) {
			fs.delete(path, true);
			throw ce;
		}
		finally {
			IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
		}
	}

	@Override
	public void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen) throws IOException {
		throw new NotImplementedException();
	}

	private void writeCompressedMatrixToHDFS(Path path, JobConf conf, FileSystem fs, MatrixBlock src)
		throws IOException {
		final OutputStream os = fs.create(path, true);
		final DataOutput out = new DataOutputStream(os);
		try {
			final MatrixBlock mb = src instanceof CompressedMatrixBlock ? // If compressed
				src : // Do not compress
				CompressedMatrixBlockFactory.compress(src).getLeft(); // otherwise compress

			if(!(mb instanceof CompressedMatrixBlock))
				throw new DMLCompressionException("Input was not compressed, therefore the file was not saved to disk");

			CompressedMatrixBlock cmb = (CompressedMatrixBlock) mb;
			cmb.write(out);
		}
		finally {
			os.close();
		}
	}
}

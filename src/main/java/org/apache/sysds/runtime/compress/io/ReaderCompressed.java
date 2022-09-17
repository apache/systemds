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

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;

import org.apache.commons.lang.NotImplementedException;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class ReaderCompressed extends MatrixReader {

	public static ReaderCompressed create() {
		return new ReaderCompressed();
	}

	public static MatrixBlock readCompressedMatrixFromHDFS(String fname) throws IOException {
		return create().readMatrixFromHDFS(fname, 0, 0, 0, 0);
	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {

		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		checkValidInputFile(fs, path);

		MatrixBlock cmb = readCompressedMatrix(path, job, fs);

		if(cmb.getNumRows() != rlen)
			LOG.warn("Metadata file does not correlate with compressed file, NRows : " + cmb.getNumRows() + " vs " + rlen);
		if(cmb.getNumColumns() != clen)
			LOG.warn(
				"Metadata file does not correlate with compressed file, NCols : " + cmb.getNumColumns() + " vs " + clen);

		return cmb;
	}

	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {
		throw new NotImplementedException("Not implemented reading compressedMatrix from input stream");
	}

	private static MatrixBlock readCompressedMatrix(Path path, JobConf job, FileSystem fs) throws IOException {
		if(fs.getFileStatus(path).isDirectory())
			return readCompressedMatrixFolder(path, job, fs);
		else
			return readCompressedMatrixSingleFile(path, job, fs);
	}

	private static MatrixBlock readCompressedMatrixFolder(Path path, JobConf job, FileSystem fs) {
		throw new NotImplementedException();
	}

	private static MatrixBlock readCompressedMatrixSingleFile(Path path, JobConf job, FileSystem fs) throws IOException {
		final InputStream is = fs.open(path);
		final DataInput in = new DataInputStream(is);
		MatrixBlock ret;
		try {
			ret = CompressedMatrixBlock.read(in);
		}
		finally {
			is.close();
		}
		return ret;
	}

}

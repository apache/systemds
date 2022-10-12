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

import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.util.HDFSTool;

public final class WriterCompressed extends MatrixWriter {

	protected static final Log LOG = LogFactory.getLog(ReaderCompressed.class.getName());

	public static WriterCompressed create(FileFormatProperties props) {
		return new WriterCompressed();
	}

	public static void writeCompressedMatrixToHDFS(MatrixBlock src, String fname) throws IOException {
		writeCompressedMatrixToHDFS(src, fname, src.getNumRows(), src.getNumColumns(), OptimizerUtils.DEFAULT_BLOCKSIZE,
			src.getNonZeros(), false);
	}

	public static void writeCompressedMatrixToHDFS(MatrixBlock src, String fname, int blen) throws IOException {
		writeCompressedMatrixToHDFS(src, fname, src.getNumRows(), src.getNumColumns(), blen, src.getNonZeros(), false);
	}

	public static void writeCompressedMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int blen,
		long nnz, boolean diag) throws IOException {
		create(null).writeMatrixToHDFS(src, fname, rlen, clen, blen, nnz, diag);
	}

	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int blen, long nnz, boolean diag)
		throws IOException {
		if(blen <= 0)
			throw new DMLRuntimeException("Invalid block size for writing to disk");
		if(diag)
			throw new DMLRuntimeException("Not supported diag for compressed writing.");
		if(fname == null)
			throw new DMLRuntimeException("Invalid missing path.");
		if(src == null)
			throw new DMLRuntimeException("Null matrix block invalid");
		if(src.getNumRows() != rlen || src.getNumColumns() != clen)
			throw new DMLRuntimeException("Invalid number of rows or columns specified not matching");
		write(src, fname, blen);
	}

	@Override
	public void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen) throws IOException {
		if(rlen <= 0)
			throw new RuntimeException("Invalid empty write with rlen : " + rlen);
		if(clen <= 0)
			throw new RuntimeException("Invalid empty write with clen : " + clen);
		if(blen <= 0)
			throw new RuntimeException("Invalid empty write with blen " + blen);
		if(rlen > Integer.MAX_VALUE || clen > Integer.MAX_VALUE)
			throw new RuntimeException("Unable to create compressed matrix block larger than IntMax");
		if(fname == null)
			throw new RuntimeException("Invalid null file name to write to");
		CompressedMatrixBlock m = CompressedMatrixBlockFactory.createConstant((int) rlen, (int) clen, 0.0);
		write(m, fname, blen);
	}

	private void write(MatrixBlock src, String fname, int blen) throws IOException {
		final int k = OptimizerUtils.getParallelTextWriteParallelism();
		final Path path = new Path(fname);
		final JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		// Delete previous
		HDFSTool.deleteFileIfExistOnHDFS(path, job);

		// Make Writer (New interface)
		Writer w = SequenceFile.createWriter(job, Writer.file(path), Writer.bufferSize(4096), Writer.blockSize(4096),
			Writer.keyClass(MatrixIndexes.class), Writer.valueClass(CompressedWriteBlock.class),
			Writer.compression(SequenceFile.CompressionType.RECORD), Writer.replication((short) 1));

		final int rlen = src.getNumRows();
		final int clen = src.getNumColumns();

		if(rlen <= blen && clen <= blen)
			writeSingleBlock(w, src, k);
		else
			writeMultiBlock(w, src, rlen, clen, blen, k);

		IOUtilFunctions.closeSilently(w);

		// Cleanup
		final FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	private void writeSingleBlock(Writer w, MatrixBlock b, int k) throws IOException {
		MatrixIndexes idx = new MatrixIndexes(1, 1);
		MatrixBlock mc = CompressedMatrixBlockFactory.compress(b, k).getLeft();
		w.append(idx, new CompressedWriteBlock(mc));
	}

	private void writeMultiBlock(Writer w, MatrixBlock b, int rlen, int clen, int blen, int k) throws IOException {
		final MatrixIndexes indexes = new MatrixIndexes();
		for(int br = 0; br * blen < rlen; br++) {
			for(int bc = 0; bc * blen < clen; bc++) {
				// Max Row and col in block
				int sR = br * blen;
				int sC = bc * blen;
				int mR = Math.min(br * blen + blen, rlen) - 1;
				int mC = Math.min(bc * blen + blen, clen) - 1;

				MatrixBlock mb = b.slice(sR, mR, sC, mC);
				MatrixBlock mc = CompressedMatrixBlockFactory.compress(mb, k).getLeft();
				indexes.setIndexes(br + 1, bc + 1);
				w.append(indexes, new CompressedWriteBlock(mc));
			}
		}
	}
}

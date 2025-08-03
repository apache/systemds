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

import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.util.HDFSTool;

import static org.apache.sysds.runtime.util.HDFSTool.getHDFSDataOutputStream;

public class WriterBinaryBlock extends MatrixWriter {
	protected int _replication = -1;

	protected static int jobUse = 0;
	protected static JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());

	public WriterBinaryBlock(int replication) {
		_replication = replication;

		jobUse ++;
		if(jobUse > 15){
			// job =  new JobConf();
			job = new JobConf(ConfigurationManager.getCachedJobConf());
			jobUse = 0;
		}
	}

	@Override
	public final void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int blen, long nnz,
		boolean diag) throws IOException, DMLRuntimeException {
		// prepare file access
		Path path = new Path(fname);

		// if the file already exists on HDFS, remove it.
		HDFSTool.deleteFileIfExistOnHDFS(path, job);

		// set up preferred custom serialization framework for binary block format
		if(HDFSTool.USE_BINARYBLOCK_SERIALIZATION)
			HDFSTool.addBinaryBlockSerializationFramework(job);

		if(src instanceof CompressedMatrixBlock) {
			if(ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS)) {
				LOG.debug("Multi threaded decompression");
				// parallel
				src = CompressedMatrixBlock.getUncompressed(src, "binary write",
					OptimizerUtils.getParallelBinaryWriteParallelism());
			}
			else {
				LOG.warn("Single threaded decompression");
				src = CompressedMatrixBlock.getUncompressed(src, "binary write");
			}
		}

		// core write sequential/parallel
		if(diag)
			writeDiagBinaryBlockMatrixToHDFS(path, job, src, rlen, clen, blen);
		else
			writeBinaryBlockMatrixToHDFS(path, job, src, rlen, clen, blen);

	}

	@Override
	public final void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen)
		throws IOException, DMLRuntimeException {
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		final Writer writer = IOUtilFunctions.getSeqWriter(path, job, _replication);
		try {
			MatrixIndexes index = new MatrixIndexes(1, 1);
			MatrixBlock block = new MatrixBlock((int) Math.max(Math.min(rlen, blen), 1),
				(int) Math.max(Math.min(clen, blen), 1), true);
			writer.append(index, block);
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	protected void writeBinaryBlockMatrixToHDFS(Path path, JobConf job,  MatrixBlock src, long rlen,
		long clen, int blen) throws IOException, DMLRuntimeException {
		// sequential write
		writeBinaryBlockMatrixToSequenceFile(path, job, src, blen, 0, (int) rlen);
		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(job, path);
	}

	protected final void writeBinaryBlockMatrixToSequenceFile(Path path, JobConf job,  MatrixBlock src,
		int blen, int rl, int ru) throws IOException {
		boolean sparse = src.isInSparseFormat();
		final int rlen = src.getNumRows();
		final int clen = src.getNumColumns();
		final Writer writer = IOUtilFunctions.getSeqWriter(path, job, _replication);

		try { 
	
			// 3) reblock and write
			MatrixIndexes indexes = new MatrixIndexes();

			if(rlen <= blen && clen <= blen && rl == 0) { // opt for single block
				// directly write single block
				indexes.setIndexes(1, 1);
				writer.append(indexes, src);
			}
			else {
				// general case
				// initialize blocks for reuse (at most 4 different blocks required)
				MatrixBlock[] blocks = createMatrixBlocksForReuse(rlen, clen, blen, sparse, src.getNonZeros());

				// create and write sub-blocks of matrix
				for(int blockRow = rl / blen; blockRow < (int) Math.ceil(ru / (double) blen); blockRow++) {
					for(int blockCol = 0; blockCol < (int) Math.ceil(clen / (double) blen); blockCol++) {
						int maxRow = (blockRow * blen + blen < rlen) ? blen : rlen - blockRow * blen;
						int maxCol = (blockCol * blen + blen < clen) ? blen : clen - blockCol * blen;

						int row_offset = blockRow * blen;
						int col_offset = blockCol * blen;

						// get reuse matrix block
						MatrixBlock block = getMatrixBlockForReuse(blocks, maxRow, maxCol, blen);

						// copy sub matrix to block
						src.slice(row_offset, row_offset + maxRow - 1, col_offset, col_offset + maxCol - 1, block);

						// append block to sequence file
						indexes.setIndexes(blockRow + 1, blockCol + 1);
						writer.append(indexes, block);

						// reset block for later reuse
						block.reset();
					}
				}
			}
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
	}

	protected final void writeDiagBinaryBlockMatrixToHDFS(Path path, JobConf job,  MatrixBlock src,
		long rlen, long clen, int blen) throws IOException {
		boolean sparse = src.isInSparseFormat();
		final int nRow = src.getNumRows();
		final int nCol = src.getNumColumns();
		final Writer writer = IOUtilFunctions.getSeqWriter(path, job, _replication);

		try {
			// 2) bound check for src block
			if(nRow > rlen || nCol > clen) {
				throw new IOException("Matrix block [1:" + nRow + ",1:" + nCol + "] "
					+ "out of overall matrix range [1:" + rlen + ",1:" + clen + "].");
			}

			// 3) reblock and write
			MatrixIndexes indexes = new MatrixIndexes();

			if(rlen <= blen && clen <= blen) { // opt for single block
				// directly write single block
				indexes.setIndexes(1, 1);
				writer.append(indexes, src);
			}
			else { // general case
				// initialize blocks for reuse (at most 4 different blocks required)
				MatrixBlock[] blocks = createMatrixBlocksForReuse(rlen, clen, blen, sparse, src.getNonZeros());
				MatrixBlock emptyBlock = new MatrixBlock();

				// create and write sub blocks of the matrix
				for(int blockRow = 0; blockRow < (int) Math.ceil(nRow / (double) blen); blockRow++) {

					for(int blockCol = 0; blockCol < (int) Math.ceil(nCol / (double) blen); blockCol++) {
						int maxRow = (blockRow * blen + blen < nRow) ? blen : nRow - blockRow * blen;
						int maxCol = (blockCol * blen + blen < nCol) ? blen : nCol - blockCol * blen;
						MatrixBlock block = null;

						if(blockRow == blockCol) { // block on diagonal
							int row_offset = blockRow * blen;
							int col_offset = blockCol * blen;

							// get reuse matrix block
							block = getMatrixBlockForReuse(blocks, maxRow, maxCol, blen);

							// copy sub matrix to block
							src.slice(row_offset, row_offset + maxRow - 1, col_offset, col_offset + maxCol - 1, block);
						}
						else { // empty block (not on diagonal)
							block = emptyBlock;
							block.reset(maxRow, maxCol);
						}

						// append block to sequence file
						indexes.setIndexes(blockRow + 1, blockCol + 1);
						writer.append(indexes, block);

						// reset block for later reuse
						if(blockRow != blockCol)
							block.reset();
					}
				}
			}
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
	}

	@Override
	public void writeMatrixFromStream(String fname, LocalTaskQueue<IndexedMatrixValue> stream, long rlen, long clen, int blen) throws IOException {
		DataOutputStream dostream_data = null;
		DataOutputStream dostream_header = null;

		String tempDataFname = fname + "._data";
		String tempHeaderFname = fname + "._header";
		Path dataPath = new Path(tempDataFname);
		Path headerPath = new Path(tempHeaderFname);
		Path finalPath = new Path(fname);

		FileSystem fs = null;

		try {
			// PASS 1: Stream to a temporary raw data file and count NNZ
			fs = IOUtilFunctions.getFileSystem(dataPath);
//			dostream = getHDFSDataOutputStream(fname, true);
			dostream_data = fs.create(dataPath, true);
//			dostream_data.writeLong(rlen);
//			dostream_data.writeLong(clen);

			long totalNnz = 0;
			IndexedMatrixValue i_val =  null;
			while((i_val = stream.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
				MatrixBlock mb = (MatrixBlock) i_val.getValue();
				totalNnz += mb.getNonZeros();

				double[] denseValues = mb.getDenseBlockValues();
				if (denseValues != null) {
					for (double v : denseValues) {
						dostream_data.writeDouble(v);
					}
				}
//				mb.write(dostream);
			}
			IOUtilFunctions.closeSilently(dostream_data);

			// PASS 2: Create a header file in RAM (very small)
			dostream_header = fs.create(headerPath, true);
			dostream_header.writeLong(rlen);
			dostream_header.writeLong(clen);
			dostream_header.writeInt(blen);
			dostream_header.writeBoolean(false); // isSparse
			dostream_header.writeLong(totalNnz);
			IOUtilFunctions.closeSilently(dostream_header);

			// MERGE STEP: Use HDFS concat for metadata-only merge
			fs.concat(finalPath, new Path[]{dataPath, headerPath});

		} catch (IOException ex) {
            throw new RuntimeException(ex);
        } catch (InterruptedException e) {
			throw new RuntimeException(e);
		} finally {
			// Cleanup incase of failure before concat
//			IOUtilFunctions.closeSilently(dostream_data);
//			IOUtilFunctions.closeSilently(dostream_header);
//
//			if (fs != null) {
//				if (fs.exists(dataPath)) fs.delete(dataPath, false);
//				if (fs.exists(headerPath)) fs.delete(headerPath, false);
//			}
		}

    };
}

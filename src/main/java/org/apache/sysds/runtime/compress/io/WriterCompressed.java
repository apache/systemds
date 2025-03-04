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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.lib.CLALibSeparator;
import org.apache.sysds.runtime.compress.lib.CLALibSeparator.SeparatedGroups;
import org.apache.sysds.runtime.compress.lib.CLALibSlice;
import org.apache.sysds.runtime.instructions.spark.CompressionSPInstruction.CompressionFunction;
import org.apache.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.apache.sysds.runtime.util.HDFSTool;

public final class WriterCompressed extends MatrixWriter {

	protected static final Log LOG = LogFactory.getLog(WriterCompressed.class.getName());

	protected static int jobUse = 0;
	protected static JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());

	private String fname;

	private FileSystem fs;
	private Future<Writer>[] writers;
	private Lock[] writerLocks;

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

	public static void writeRDDToHDFS(JavaPairRDD<MatrixIndexes, MatrixBlock> src, String path, int blen,
		DataCharacteristics mc) {
		final DataCharacteristics outC = new MatrixCharacteristics(mc).setBlocksize(blen);
		writeRDDToHDFS(RDDConverterUtils.binaryBlockToBinaryBlock(src, mc, outC), path);
	}

	public static void writeRDDToHDFS(JavaPairRDD<MatrixIndexes, MatrixBlock> src, String path) {
		src.mapValues(new CompressionFunction()) // Try to compress each block.
			.mapValues(new CompressWrap()) // Wrap in writable
			.saveAsHadoopFile(path, MatrixIndexes.class, CompressedWriteBlock.class, SequenceFileOutputFormat.class);
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

	private void write(MatrixBlock src, final String fname, final int blen) throws IOException {
		jobUse++;
		if(jobUse > 30) {
			job = new JobConf(ConfigurationManager.getCachedJobConf());
			jobUse = 0;
		}

		if(this.fname != fname) {
			this.fname = fname;
			this.writers = null;
		}

		fs = IOUtilFunctions.getFileSystem(new Path(fname), job);
	
		int k = OptimizerUtils.getParallelBinaryWriteParallelism();

		k =  Math.min(k, (int)(src.getInMemorySize() /  InfrastructureAnalyzer.getBlockSize(fs)));
		final int rlen = src.getNumRows();
		final int clen = src.getNumColumns();
		// Try to compress!
		if(!(src instanceof CompressedMatrixBlock))
			src = CompressedMatrixBlockFactory.compress(src, k).getLeft();

		if(rlen <= blen && clen <= blen)
			writeSingleBlock(src, k); // equivalent to single threaded.
		else if(!(src instanceof CompressedMatrixBlock))
			writeMultiBlockUncompressed(src, rlen, clen, blen, k);
		else
			writeMultiBlockCompressed(src, rlen, clen, blen, k);

	}

	private void writeSingleBlock(MatrixBlock b, int k) throws IOException {
		final Path path = new Path(fname);
		Writer w = generateWriter(job, path, fs);
		MatrixIndexes idx = new MatrixIndexes(1, 1);
		if(!(b instanceof CompressedMatrixBlock))
			b = CompressedMatrixBlockFactory.compress(b, k).getLeft();
		w.append(idx, new CompressedWriteBlock(b));
		IOUtilFunctions.closeSilently(w);
		cleanup(path);
	}

	private void writeMultiBlockUncompressed(MatrixBlock b, final int rlen, final int clen, final int blen, int k)
		throws IOException {
		final Path path = new Path(fname);
		Writer w = generateWriter(job, path, fs);
		final MatrixIndexes indexes = new MatrixIndexes();
		LOG.warn("Writing compressed format with non identical compression scheme");

		for(int bc = 0; bc * blen < clen; bc++) {
			final int sC = bc * blen;
			final int mC = Math.min(sC + blen, clen) - 1;
			for(int br = 0; br * blen < rlen; br++) {
				// Max Row and col in block
				final int sR = br * blen;
				final int mR = Math.min(sR + blen, rlen) - 1;
				MatrixBlock mb = b.slice(sR, mR, sC, mC);
				MatrixBlock mc = CompressedMatrixBlockFactory.compress(mb, k).getLeft();
				indexes.setIndexes(br + 1, bc + 1);
				w.append(indexes, new CompressedWriteBlock(mc));
			}
		}
		IOUtilFunctions.closeSilently(w);
		cleanup(path);
	}

	private void writeMultiBlockCompressed(MatrixBlock b, final int rlen, final int clen, final int blen, int k)
		throws IOException {
		if(k > 1)
			writeMultiBlockCompressedParallel(b, rlen, clen, blen, k);
		else
			writeMultiBlockCompressedSingleThread(b, rlen, clen, blen);

	}

	private void writeMultiBlockCompressedSingleThread(MatrixBlock mb, final int rlen, final int clen, final int blen)
		throws IOException {
		try {
			final CompressedMatrixBlock cmb = (CompressedMatrixBlock) mb;
			final Path path = new Path(fname);
			Writer w = generateWriter(job, path, fs);
			for(int bc = 0; bc * blen < clen; bc++) {// column blocks
				final int sC = bc * blen;
				final int mC = Math.min(sC + blen, clen) - 1;
				// slice out the current columns
				final CompressedMatrixBlock mc = CLALibSlice.sliceColumns(cmb, sC, mC);
				final SeparatedGroups s = CLALibSeparator.split(mc.getColGroups());
				final CompressedMatrixBlock rmc = new CompressedMatrixBlock(mc.getNumRows(), mc.getNumColumns(),
					mc.getNonZeros(), false, s.indexStructures);
				final int nBlocks = rlen / blen + (rlen % blen > 0 ? 1 : 0);
				write(w, rmc, bc + 1, 1, nBlocks + 1, blen);

				new DictWriteTask(fname, s.dicts, bc).call();

			}
			IOUtilFunctions.closeSilently(w);
			cleanup(path);
		}
		catch(Exception e) {
			throw new IOException(e);
		}

	}

	@SuppressWarnings("unchecked")
	private void writeMultiBlockCompressedParallel(MatrixBlock b, final int rlen, final int clen, final int blen, int k)
		throws IOException {

		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final ArrayList<Callable<Object>> tasks = new ArrayList<>();
			if(writers == null) {
				writers = new Future[k];
				writerLocks = new Lock[k];
			}
			for(int i = 0; i < k; i++) {
				final int j = i;
				if(writers[i] == null) {
					writers[i] = pool.submit(() -> {
						return generateWriter(job, getPath(j), fs);
					});
				}
				writerLocks[i] = new ReentrantLock();
			}

			final int colBlocks = (int) Math.ceil((double) clen / blen );
			final int nBlocks = (int) Math.ceil((double) rlen / blen);
			final int blocksPerThread = Math.max(1, nBlocks * colBlocks / k );
			HDFSTool.deleteFileIfExistOnHDFS(new Path(fname + ".dict"), job);
			
			int i = 0;
			for(int bc = 0; bc * blen < clen; bc++) {// column blocks
				final int sC = bc * blen;
				final int mC = Math.min(sC + blen, clen) - 1;
				final CompressedMatrixBlock mc = CLALibSlice.sliceColumns((CompressedMatrixBlock) b, sC, mC);
				final SeparatedGroups s = CLALibSeparator.split(mc.getColGroups());
				final CompressedMatrixBlock rmc = new CompressedMatrixBlock(mc.getNumRows(), mc.getNumColumns(),
					mc.getNonZeros(), false, s.indexStructures);
				for(int block = 0; block < nBlocks; block += blocksPerThread) {
					WriteTask we = new WriteTask(i++ % k, rmc, bc, block, Math.min(nBlocks, block + blocksPerThread),
						blen);
					tasks.add(we);
				}
				tasks.add(new DictWriteTask(fname, s.dicts, bc));
			}

			for(Future<Object> o : pool.invokeAll(tasks))
				o.get();

			for(int z = 0; z < writers.length; z++) {
				final int l = z;
				pool.submit(() -> {
					try {
						IOUtilFunctions.closeSilently(writers[l].get());
						cleanup(job, getPath(l), fs);
					}
					catch(Exception e) {
						throw new RuntimeException(e);
					}
				});
			}

		}
		catch(Exception e) {
			throw new IOException("Failed writing compressed multi block", e);
		}
		finally {
			pool.shutdown();
		}
	}

	private Path getPath(int id) {
		return new Path(fname, IOUtilFunctions.getPartFileName(id));
	}

	private static Writer generateWriter(JobConf job, Path path, FileSystem fs) throws IOException {

		return SequenceFile.createWriter(job, Writer.file(path), Writer.bufferSize(4096), //
			Writer.keyClass(MatrixIndexes.class), //
			Writer.valueClass(CompressedWriteBlock.class), //
			Writer.compression(IOUtilFunctions.getCompressionEncodingType(), IOUtilFunctions.getCompressionCodec()),
			Writer.replication((short) 1));
	}

	private void cleanup(Path p) throws IOException {
		cleanup(job, p, fs);
	}

	private static void cleanup(JobConf job, Path path, FileSystem fs) throws IOException {
		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	private static void write(Writer w, CompressedMatrixBlock rmc, int bc, int bl, int bu, int blen)
		throws IOException {
		final int nrow = rmc.getNumRows();
		final int nGroups = rmc.getColGroups().size();
		for(int b = bl; b < bu; b++) {
			MatrixIndexes index = new MatrixIndexes(b, bc);
			MatrixBlock cb = CLALibSlice.sliceRowsCompressed(rmc, (b - 1) * blen, Math.min(b * blen, nrow) - 1);
			if(cb instanceof CompressedMatrixBlock && ((CompressedMatrixBlock)cb).getColGroups().size() != nGroups)
				throw new RuntimeException("invalid writing of different number of column groups");
			CompressedWriteBlock blk = new CompressedWriteBlock(cb);
			w.append(index, blk);
		}
	}

	private class WriteTask implements Callable<Object> {
		final int id;
		final CompressedMatrixBlock rmc;
		final int bc;
		final int bl;
		final int bu;
		final int blen;

		private WriteTask(int id, CompressedMatrixBlock rmc, int bc, int bl, int bu, int blen) {
			this.id = id;
			this.rmc = rmc;
			// +1 for one indexed
			this.bl = bl + 1;
			this.bu = bu + 1;
			this.bc = bc + 1;
			this.blen = blen;
		}

		@Override
		public Object call() throws Exception {
			writerLocks[id].lock();
			try {
				Writer w = writers[id].get();
				write(w, rmc, bc, bl, bu, blen);
				return null;
			}
			finally {
				writerLocks[id].unlock();
			}
		}
	}

	private class DictWriteTask implements Callable<Object> {

		final String fname;
		final List<IDictionary> dicts;
		final Integer id;

		protected DictWriteTask(String fname, List<IDictionary> dicts, int id) {
			this.fname = fname;
			this.dicts = dicts;
			this.id = id;
		}

		@Override
		public Object call() throws Exception {

			Path p = new Path(fname + ".dict", IOUtilFunctions.getPartFileName(id));
			HDFSTool.deleteFileIfExistOnHDFS(p, job);
			try(Writer w = SequenceFile.createWriter(job, Writer.file(p), //
				Writer.bufferSize(4096), //
				Writer.keyClass(DictWritable.K.class), //
				Writer.valueClass(DictWritable.class), //
				Writer.compression(//
					IOUtilFunctions.getCompressionEncodingType(), //
					IOUtilFunctions.getCompressionCodec()), //
				Writer.replication((short) 1))) {
				w.append(new DictWritable.K(id), new DictWritable(dicts));
			}
			cleanup(job, p, fs);
			return null;

		}

	}

}

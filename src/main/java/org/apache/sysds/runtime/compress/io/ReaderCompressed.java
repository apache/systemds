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
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.lib.CLALibStack;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.util.CommonThreadPool;

public final class ReaderCompressed extends MatrixReader {

	private final int k;

	public ReaderCompressed() {
		this.k = OptimizerUtils.getParallelBinaryReadParallelism();
	}

	public ReaderCompressed(int k) {
		this.k = k;
	}

	public static ReaderCompressed create() {
		int numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
		return new ReaderCompressed(numThreads);
	}

	public static MatrixBlock readCompressedMatrixFromHDFS(String fname, long rlen, long clen, int blen) throws IOException {
		return create().readMatrixFromHDFS(fname, rlen, clen, blen, 100);
	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		return readCompressedMatrix(fname, job, fs, (int) rlen, (int) clen, blen);
	}

	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {
		throw new NotImplementedException("Not implemented reading compressedMatrix from input stream");
	}

	private MatrixBlock readCompressedMatrix(String fname, JobConf job, FileSystem fs, int rlen, int clen, int blen)
		throws IOException {
		if(k > 1) 
			return readCompressedMatrixParallel(fname, job, fs, rlen, clen, blen);
		else
			return readCompressedMatrixSingleThread(fname, job, fs, rlen, clen, blen);
	}

	private MatrixBlock readCompressedMatrixParallel(String fname, final JobConf job, FileSystem fs, int rlen, int clen,
		int blen) throws IOException {

		final Map<MatrixIndexes, MatrixBlock> data = new HashMap<>();
		Map<Integer, List<IDictionary>> dicts = null;
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			List<Future<Map<MatrixIndexes, MatrixBlock>>> rt = new ArrayList<>();
			List<Future<Map<Integer, List<IDictionary>>>> dt = new ArrayList<>();
			for(Path subPath : IOUtilFunctions.getSequenceFilePaths(fs, new Path(fname))) {
				final Path sp = subPath;
				rt.add(pool.submit(() -> readColumnGroups(sp, job)));
			}

			final Path dictPath = new Path(fname + ".dict");
			final boolean dictExists = fs.exists(dictPath);
			if(dictExists) {
				dicts = new HashMap<>();
				for(Path subPath : IOUtilFunctions.getSequenceFilePaths(fs, dictPath)) {
					final Path sp = subPath;
					dt.add(pool.submit(() -> readDictionaries(sp, job)));
				}
			}

			for(Future<Map<MatrixIndexes, MatrixBlock>> e : rt)
				data.putAll(e.get());

			if(dictExists && dicts != null)
				for(Future<Map<Integer, List<IDictionary>>> e : dt)
					dicts.putAll(e.get());

			return CLALibStack.combine(data, dicts, rlen, clen, blen, k);
		}
		catch(Exception e) {
			throw new IOException("failed parallel reading ", e);
		}
		finally {
			pool.shutdown();
		}
	}

	private MatrixBlock readCompressedMatrixSingleThread(String fname, JobConf job, FileSystem fs, int rlen, int clen,
		int blen) throws IOException {

		final Map<MatrixIndexes, MatrixBlock> data = new HashMap<>();

		for(Path subPath : IOUtilFunctions.getSequenceFilePaths(fs, new Path(fname))) {
			data.putAll(readColumnGroups(subPath, job));
		}

		final Path dictPath = new Path(fname + ".dict");
		Map<Integer, List<IDictionary>> dicts = null;
		if(fs.exists(dictPath)) {
			dicts = new HashMap<>();
			for(Path subPath : IOUtilFunctions.getSequenceFilePaths(fs, dictPath)) {
				dicts.putAll(readDictionaries(subPath, job));
			}
		}

		if(data.containsValue(null))
			throw new DMLCompressionException("Invalid read data contains null:");

		return CLALibStack.combine(data, dicts, k);
	}

	private static Map<MatrixIndexes, MatrixBlock> readColumnGroups(Path path, JobConf job) throws IOException {
		Map<MatrixIndexes, MatrixBlock> data = new HashMap<>();
		final Reader reader = new SequenceFile.Reader(job, SequenceFile.Reader.file(path));
		try {
			// Materialize all sub blocks.

			// Use write and read interface to read and write this object.
			MatrixIndexes key = new MatrixIndexes();
			CompressedWriteBlock value = new CompressedWriteBlock();

			while(reader.next(key, value)) {
				final MatrixBlock g = value.get();
				if(g instanceof CompressedMatrixBlock)
					data.put(key, g);
				else if(g.isEmpty())
					data.put(key, CompressedMatrixBlockFactory.createConstant(g.getNumRows(), g.getNumColumns(), 0.0));
				else
					data.put(key, g);
				key = new MatrixIndexes();
				value = new CompressedWriteBlock();
			}
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
		return data;
	}

	private static Map<Integer, List<IDictionary>> readDictionaries(Path path, JobConf job) throws IOException {
		Map<Integer, List<IDictionary>> data = new HashMap<>();
		final Reader reader = new SequenceFile.Reader(job, SequenceFile.Reader.file(path));

		try {
			// Materialize all sub blocks.
			// Use write and read interface to read and write this object.
			DictWritable.K key = new DictWritable.K(0);
			DictWritable value = new DictWritable(null);
			while(reader.next(key, value))
				data.put(key.id, value.dicts);
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
		return data;
	}
}

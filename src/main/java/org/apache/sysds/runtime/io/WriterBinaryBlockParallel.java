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

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.stream.SplittingOOCStream;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class WriterBinaryBlockParallel extends WriterBinaryBlock
{
	public WriterBinaryBlockParallel( int replication ) {
		super(replication);
	}
	
	@Override
	protected void writeBinaryBlockMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int blen )
		throws IOException, DMLRuntimeException
	{
		//estimate output size and number of output blocks (min 1)
		int numPartFiles = numPartsFiles(path.getFileSystem(job), rlen, clen, blen , src.getNonZeros());

		//determine degree of parallelism
		int numThreads = OptimizerUtils.getParallelBinaryWriteParallelism();
		numThreads = Math.min(numThreads, numPartFiles);
		
		//fall back to sequential write if dop is 1 (e.g., <128MB) in order to create single file
		if( numThreads <= 1 ) {
			super.writeBinaryBlockMatrixToHDFS(path, job,  src, rlen, clen, blen);
			return;
		}

		//create directory for concurrent tasks
		// HDFSTool.createDirIfNotExistOnHDFS(path, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		
		//create and execute write tasks
		final ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			ArrayList<WriteFileTask> tasks = new ArrayList<>();
			int blklen = (int)Math.ceil((double)rlen / blen / numThreads) * blen;
			for(int i=0; i<numThreads & i*blklen<rlen; i++) {
				Path newPath = new Path(path, IOUtilFunctions.getPartFileName(i));
				tasks.add(new WriteFileTask(newPath, job,  src, i*blklen, Math.min((i+1)*blklen, rlen), blen));
			}

			//check for exceptions 
			for( Future<Object> task : pool.invokeAll(tasks) )
				task.get();
			
		} 
		catch (Exception e) {
			throw new IOException("Failed parallel write of binary block input.", e);
		}
		finally{
			pool.shutdown();
		}
	}

	@Override
	public long writeMatrixFromStream(String fname, OOCStream<IndexedMatrixValue> stream, long rlen, long clen, int blen)
		throws IOException {
		Path path = new Path(fname);
		long nnz = -1;
		DataCharacteristics dc = stream.getDataCharacteristics();
		if(dc != null)
			nnz = dc.getNonZeros();
		if(nnz < 0 && rlen > 0 && clen > 0) {
			if(rlen > Long.MAX_VALUE / clen)
				nnz = Long.MAX_VALUE - 1;
			else
				nnz = rlen * clen;
		}
		else if(nnz < 0)
			nnz = 0;

		int numPartFiles = numPartsFiles(path.getFileSystem(job), rlen, clen, blen, nnz);
		int numThreads = OptimizerUtils.getParallelBinaryWriteParallelism();
		numThreads = Math.min(numThreads, numPartFiles);

		// fall back to sequential write if dop is 1 in order to create a single file
		if(numThreads <= 1)
			return super.writeMatrixFromStream(fname, stream, rlen, clen, blen);

		// Match CP parallel writer partitioning by contiguous row ranges.
		final int parallelism = numThreads;
		final int blklen = (int) Math.ceil((double) rlen / blen / parallelism) * blen;
		SplittingOOCStream<IndexedMatrixValue> split = new SplittingOOCStream<>(stream, iVal -> {
			int partition = (int) (((iVal.getIndexes().getRowIndex() - 1) * blen) / (long) blklen);
			return Math.max(0, Math.min(partition, parallelism - 1));
		}, parallelism);

		final ExecutorService pool = Executors.newFixedThreadPool(parallelism);
		try {
			ArrayList<WriteStreamTask> tasks = new ArrayList<>();
			for(int i = 0; i < parallelism && i * (long) blklen < rlen; i++) {
				Path newPath = new Path(path, IOUtilFunctions.getPartFileName(i));
				tasks.add(new WriteStreamTask(newPath, job, split.getSubStream(i)));
			}

			long totalNnz = 0;
			for(Future<Long> task : pool.invokeAll(tasks))
				totalNnz += task.get();
			return totalNnz;
		}
		catch(Exception e) {
			DMLRuntimeException ex = DMLRuntimeException.of(e);
			split.propagateFailure(ex);
			throw ex;
		}
		finally {
			pool.shutdown();
		}
	}

	public static int numPartsFiles(FileSystem fs, long rlen, long clen, long blen, long nZeros) {
		int numPartFiles = (int) (OptimizerUtils.estimatePartitionedSizeExactSparsity(rlen, clen, blen, nZeros) /
			InfrastructureAnalyzer.getBlockSize(fs));
		numPartFiles = Math.max(numPartFiles, 1);
		numPartFiles = Math.min(numPartFiles,
			(int) (Math.ceil((double) rlen / blen) * Math.ceil((double) clen / blen)));
		return numPartFiles;
	}

	private class WriteFileTask implements Callable<Object> 
	{
		private Path _path = null;
		private JobConf _job = null;
		private MatrixBlock _src = null;
		private long _rl = -1;
		private long _ru = -1;
		private int _blen = -1;
		
		public WriteFileTask(Path path, JobConf job, MatrixBlock src, long rl, long ru, int blen) {
			_path = path;
			_job = job;
			_src = src;
			_rl = rl;
			_ru = ru;
			_blen = blen;
		}
	
		@Override
		public Object call() throws Exception {
			writeBinaryBlockMatrixToSequenceFile(_path, _job,  _src, _blen, (int) _rl, (int) _ru);
			IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(_job, _path);
			return null;
		}
	}

	private class WriteStreamTask implements Callable<Long> {
		private final Path _path;
		private final JobConf _job;
		private final OOCStream<IndexedMatrixValue> _stream;

		public WriteStreamTask(Path path, JobConf job, OOCStream<IndexedMatrixValue> stream) {
			_path = path;
			_job = job;
			_stream = stream;
		}

		@Override
		public Long call() throws Exception {
			SequenceFile.Writer writer = null;
			long totalNnz = 0;
			try {
				writer = IOUtilFunctions.getSeqWriter(_path, _job, _replication);
				IndexedMatrixValue i_val;
				while((i_val = _stream.dequeue()) != LocalTaskQueue.NO_MORE_TASKS) {
					MatrixBlock mb = (MatrixBlock) i_val.getValue();
					MatrixIndexes ix = i_val.getIndexes();
					writer.append(ix, mb);
					totalNnz += mb.getNonZeros();
				}
			}
			finally {
				IOUtilFunctions.closeSilently(writer);
			}
			IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(_job, _path);
			return totalNnz;
		}
	}
}

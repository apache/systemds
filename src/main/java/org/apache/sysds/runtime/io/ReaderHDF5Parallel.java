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
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.hdf5.H5Constants;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class ReaderHDF5Parallel extends ReaderHDF5 {

	final private int _numThreads;
	protected JobConf _job;

	public ReaderHDF5Parallel(FileFormatPropertiesHDF5 props) {
		super(props);
		_numThreads = OptimizerUtils.getParallelTextReadParallelism();
	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {

		// prepare file access
		_job = new JobConf(ConfigurationManager.getCachedJobConf());

		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, _job);

		FileInputFormat.addInputPath(_job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(_job);

		// check existence and non-empty file
		checkValidInputFile(fs, path);

		// allocate output matrix block
		ArrayList<Path> files = new ArrayList<>();
		files.add(path);
		MatrixBlock src = computeHDF5Size(files, fs, _props.getDatasetName());

		//create and execute tasks
		try {
			ExecutorService pool = CommonThreadPool.get(_numThreads);
			int bufferSize = (src.getNumColumns() * src.getNumRows()) * 8 + H5Constants.STATIC_HEADER_SIZE;
			ArrayList<ReadHDF5Task> tasks = new ArrayList<>();
			rlen = src.getNumRows();
			int blklen = (int) Math.ceil((double) rlen / _numThreads);
			for(int i = 0; i < _numThreads & i * blklen < rlen; i++) {
				int rl = i * blklen;
				int ru = (int) Math.min((i + 1) * blklen, rlen);
				BufferedInputStream bis = new BufferedInputStream(fs.open(path), bufferSize);

				//BufferedInputStream bis, String datasetName, MatrixBlock src, MutableInt rl, int ru
				tasks.add(new ReadHDF5Task(bis, _props.getDatasetName(), src, rl, ru));
			}

			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);
			pool.shutdown();

			//check for exceptions
			for(Future<Object> task : rt)
				task.get();
		}
		catch(Exception e) {
			throw new IOException("Failed parallel read of HDF5 input.", e);
		}
		return src;
	}

	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {

		return new ReaderHDF5(_props).readMatrixFromInputStream(is, rlen, clen, blen, estnnz);
	}

	private static class ReadHDF5Task implements Callable<Object> {

		private final BufferedInputStream _bis;
		private final String _datasetName;
		private final MatrixBlock _src;
		private final int _rl;
		private final int _ru;

		public ReadHDF5Task(BufferedInputStream bis, String datasetName, MatrixBlock src, int rl, int ru) {
			_bis = bis;
			_datasetName = datasetName;
			_src = src;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Object call() throws IOException {
			readMatrixFromHDF5(_bis, _datasetName, _src, _rl, _ru, 0, 0);
			return null;
		}
	}
}

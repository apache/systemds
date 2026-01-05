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

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64DEDUP;
import org.apache.sysds.runtime.data.DenseBlockLFP64DEDUP;
import org.apache.sysds.runtime.io.hdf5.H5ByteReader;
import org.apache.sysds.runtime.io.hdf5.H5ContiguousDataset;
import org.apache.sysds.runtime.io.hdf5.H5RootObject;
import org.apache.sysds.runtime.io.hdf5.H5;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.UtilFunctions;

public class ReaderHDF5Parallel extends ReaderHDF5 {

	final private int _numThreads;
	protected JobConf _job;

	public ReaderHDF5Parallel(FileFormatPropertiesHDF5 props) {
		super(props);
		_numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
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
		MatrixBlock src = computeHDF5Size(files, fs, _props.getDatasetName(), estnnz);
		if(ReaderHDF5.isLocalFileSystem(fs) && !fs.getFileStatus(path).isDirectory()) {
			Long nnz = readMatrixFromHDF5ParallelLocal(path, fs, src, 0, src.getNumRows(),
				src.getNumColumns(), blen, _props.getDatasetName());
			if(nnz != null) {
				src.setNonZeros(nnz);
				return src;
			}
		}
		int numParts = Math.min(files.size(), _numThreads);
		
		//create and execute tasks
		ExecutorService pool = CommonThreadPool.get(_numThreads);
		try {
			ArrayList<ReadHDF5Task> tasks = new ArrayList<>();
			rlen = src.getNumRows();
			int blklen = (int) Math.ceil((double) rlen / numParts);
			for(int i = 0; i < _numThreads & i * blklen < rlen; i++) {
				int rl = i * blklen;
				int ru = (int) Math.min((i + 1) * blklen, rlen);
				Path newPath = HDFSTool.isDirectory(fs, path) ? 
					new Path(path, IOUtilFunctions.getPartFileName(i)) : path;
				tasks.add(new ReadHDF5Task(fs, newPath, _props.getDatasetName(), src, rl, ru, clen, blklen));
			}

			long nnz = 0;
			for(Future<Long> task : pool.invokeAll(tasks))
				nnz += task.get();
			src.setNonZeros(nnz);
			
			return src;
		}
		catch(Exception e) {
			throw new IOException("Failed parallel read of HDF5 input.", e);
		}
		finally{
			pool.shutdown();
		}
	}

	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException
	{
		return new ReaderHDF5(_props).readMatrixFromInputStream(is, rlen, clen, blen, estnnz);
	}

	private static Long readMatrixFromHDF5ParallelLocal(Path path, FileSystem fs, MatrixBlock dest,
		int rl, long ru, long clen, int blen, String datasetName) throws IOException
	{
		H5RootObject rootObject = null;
		long dataAddress;
		long elemSize;
		long rows;
		long cols;
		try {
			H5ByteReader metaReader = createByteReader(path, fs);
			rootObject = H5.H5Fopen(metaReader);
			H5ContiguousDataset dataset = H5.H5Dopen(rootObject, datasetName);
			if(dataset.isRankGt2() && !dataset.isRowContiguous()) {
				rootObject.close();
				return null;
			}
			elemSize = dataset.getElementSize();
			if(elemSize != 8) {
				rootObject.close();
				return null;
			}
			dataAddress = dataset.getDataAddress();
			rows = rootObject.getRow();
			cols = rootObject.getCol();
			long rowByteSize = dataset.getRowByteSize();
			if(rowByteSize <= 0) {
				rootObject.close();
				return null;
			}
			rootObject.close();
			rootObject = null;
		}
		finally {
			if(rootObject != null)
				rootObject.close();
		}

		if(dest.isInSparseFormat()) {
			if(HDF5_FORCE_DENSE) {
				dest.allocateDenseBlock(true);
				if(HDF5_READ_TRACE)
					System.out.println("[HDF5] Forcing dense output for parallel mmap dataset=" + datasetName);
			}
			else {
				return null;
			}
		}
		DenseBlock denseBlock = dest.getDenseBlock();
		boolean fastDense = denseBlock.isNumeric(ValueType.FP64)
			&& !(denseBlock instanceof DenseBlockFP64DEDUP)
			&& !(denseBlock instanceof DenseBlockLFP64DEDUP);
		boolean contiguousDense = fastDense && denseBlock.isContiguous();
		if(!fastDense) {
			return null;
		}

		if(cols > Integer.MAX_VALUE || rows > Integer.MAX_VALUE) {
			return null;
		}
		int ncol = (int) cols;
		long rowBytesLong = elemSize * ncol;
		if(rowBytesLong <= 0 || rowBytesLong > Integer.MAX_VALUE) {
			return null;
		}
		long totalRowsLong = ru - rl;
		if(totalRowsLong <= 0 || totalRowsLong > Integer.MAX_VALUE) {
			return null;
		}
		long totalBytes = totalRowsLong * rowBytesLong;
		if(totalBytes < HDF5_READ_PARALLEL_MIN_BYTES || HDF5_READ_PARALLEL_THREADS <= 1) {
			return null;
		}

		int numThreads = Math.min(HDF5_READ_PARALLEL_THREADS, (int) totalRowsLong);
		int rowsPerTask = (int) Math.ceil((double) totalRowsLong / numThreads);
		double[] destAll = contiguousDense ? denseBlock.values(0) : null;
		int destBase = contiguousDense ? denseBlock.pos(rl) : 0;
		int rowBytes = (int) rowBytesLong;
		int windowBytes = HDF5_READ_MAP_BYTES;
		boolean skipNnz = HDF5_SKIP_NNZ;
		if(HDF5_READ_TRACE) {
			System.out.println("[HDF5] Parallel mmap read enabled dataset=" + datasetName + " rows=" + totalRowsLong
				+ " cols=" + cols + " threads=" + numThreads + " windowBytes=" + windowBytes + " skipNnz=" + skipNnz);
		}

		java.io.File localFile = getLocalFile(path);
		ExecutorService pool = CommonThreadPool.get(numThreads);
		ArrayList<Callable<Long>> tasks = new ArrayList<>();
		for(int rowOffset = 0; rowOffset < totalRowsLong; rowOffset += rowsPerTask) {
			int rowsToRead = (int) Math.min(rowsPerTask, totalRowsLong - rowOffset);
			int destOffset = contiguousDense ? destBase + rowOffset * ncol : 0;
			int startRow = rl + rowOffset;
			long fileOffset = dataAddress + ((long) (rl + rowOffset) * rowBytes);
			tasks.add(new H5ParallelReadTask(localFile, fileOffset, rowBytes, rowsToRead, ncol, destAll,
				destOffset, denseBlock, startRow, windowBytes, skipNnz));
		}

		long lnnz = 0;
		try {
			for(Future<Long> task : pool.invokeAll(tasks))
				lnnz += task.get();
		}
		catch(Exception e) {
			throw new IOException("Failed parallel read of HDF5 input.", e);
		}
		finally {
			pool.shutdown();
		}

		if(skipNnz) {
			lnnz = Math.multiplyExact(totalRowsLong, clen);
		}
		return lnnz;
	}

	private static final class H5ParallelReadTask implements Callable<Long> {
		private static final int ELEM_BYTES = 8;
		private final java.io.File file;
		private final long fileOffset;
		private final int rowBytes;
		private final int rows;
		private final int ncol;
		private final double[] dest;
		private final int destOffset;
		private final DenseBlock denseBlock;
		private final int startRow;
		private final int windowBytes;
		private final boolean skipNnz;

		H5ParallelReadTask(java.io.File file, long fileOffset, int rowBytes, int rows, int ncol, double[] dest,
			int destOffset, DenseBlock denseBlock, int startRow, int windowBytes, boolean skipNnz)
		{
			this.file = file;
			this.fileOffset = fileOffset;
			this.rowBytes = rowBytes;
			this.rows = rows;
			this.ncol = ncol;
			this.dest = dest;
			this.destOffset = destOffset;
			this.denseBlock = denseBlock;
			this.startRow = startRow;
			this.windowBytes = windowBytes;
			this.skipNnz = skipNnz;
		}

		@Override
		public Long call() throws IOException {
			long nnz = 0;
			long remaining = (long) rows * rowBytes;
			long offset = fileOffset;
			int destIndex = destOffset;
			int rowCursor = startRow;
			int window = Math.max(windowBytes, ELEM_BYTES);
			try(FileInputStream fis = new FileInputStream(file);
				FileChannel channel = fis.getChannel()) {
				while(remaining > 0) {
					int mapBytes;
					if(dest != null) {
						mapBytes = (int) Math.min(window, remaining);
						mapBytes -= mapBytes % ELEM_BYTES;
						if(mapBytes == 0)
							mapBytes = (int) Math.min(remaining, ELEM_BYTES);
					}
					else {
						int rowsInMap = (int) Math.min(remaining / rowBytes, window / rowBytes);
						if(rowsInMap <= 0)
							rowsInMap = 1;
						mapBytes = rowsInMap * rowBytes;
					}
					MappedByteBuffer map = channel.map(FileChannel.MapMode.READ_ONLY, offset, mapBytes);
					map.order(ByteOrder.LITTLE_ENDIAN);
					DoubleBuffer db = map.asDoubleBuffer();
					int doubles = mapBytes / ELEM_BYTES;
					if(dest != null) {
						db.get(dest, destIndex, doubles);
						if(!skipNnz)
							nnz += UtilFunctions.computeNnz(dest, destIndex, doubles);
						destIndex += doubles;
					}
					else {
						int rowsRead = mapBytes / rowBytes;
						for(int r = 0; r < rowsRead; r++) {
							double[] rowVals = denseBlock.values(rowCursor + r);
							int rowPos = denseBlock.pos(rowCursor + r);
							db.get(rowVals, rowPos, ncol);
							if(!skipNnz)
								nnz += UtilFunctions.computeNnz(rowVals, rowPos, ncol);
						}
						rowCursor += rowsRead;
					}
					offset += mapBytes;
					remaining -= mapBytes;
				}
			}
			return nnz;
		}
	}

	private static class ReadHDF5Task implements Callable<Long> {

		private final FileSystem _fs;
		private final Path _path;
		private final String _datasetName;
		private final MatrixBlock _src;
		private final int _rl;
		private final int _ru;
		private final long _clen;
		private final int _blen;

		public ReadHDF5Task(FileSystem fs, Path path, String datasetName, MatrixBlock src, 
			int rl, int ru, long clen, int blen)
		{
			_fs = fs;
			_path = path;
			_datasetName = datasetName;
			_src = src;
			_rl = rl;
			_ru = ru;
			_clen = clen;
			_blen = blen;
		}

		@Override
		public Long call() throws IOException {
			try(H5ByteReader byteReader = ReaderHDF5.createByteReader(_path, _fs)) {
				return readMatrixFromHDF5(byteReader, _datasetName, _src, _rl, _ru, _clen, _blen);
			}
		}
	}
}

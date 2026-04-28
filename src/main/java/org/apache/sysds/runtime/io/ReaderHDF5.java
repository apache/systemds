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

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RawLocalFileSystem;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64DEDUP;
import org.apache.sysds.runtime.data.DenseBlockLFP64DEDUP;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.io.hdf5.H5;
import org.apache.sysds.runtime.io.hdf5.H5ContiguousDataset;
import org.apache.sysds.runtime.io.hdf5.H5ByteReader;
import org.apache.sysds.runtime.io.hdf5.H5RootObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.hops.OptimizerUtils;

public class ReaderHDF5 extends MatrixReader {
	protected final FileFormatPropertiesHDF5 _props;
	private static final int DEFAULT_HDF5_READ_BLOCK_BYTES = 8 * 1024 * 1024; // Default contiguous read block size (8 MiB).
	private static final int DEFAULT_HDF5_READ_BUFFER_BYTES = 16 * 1024 * 1024; // Default readahead window (16 MiB).
	private static final int DEFAULT_HDF5_READ_MAP_BYTES = 256 * 1024 * 1024; // Default mmap window (256 MiB).
	private static final int DEFAULT_HDF5_READ_PARALLEL_MIN_BYTES = 64 * 1024 * 1024; // Minimum bytes before parallel read.
	private static final int HDF5_READ_BLOCK_BYTES =
		getHdf5ReadInt("sysds.hdf5.read.block.bytes", DEFAULT_HDF5_READ_BLOCK_BYTES);
	private static final int HDF5_READ_BUFFER_BYTES = Math.max(
		getHdf5ReadInt("sysds.hdf5.read.buffer.bytes", DEFAULT_HDF5_READ_BUFFER_BYTES),
		HDF5_READ_BLOCK_BYTES);
	protected static final int HDF5_READ_MAP_BYTES = Math.max(
		getHdf5ReadInt("sysds.hdf5.read.map.bytes", DEFAULT_HDF5_READ_MAP_BYTES),
		HDF5_READ_BUFFER_BYTES);
	private static final boolean HDF5_READ_USE_MMAP =
		getHdf5ReadBoolean("sysds.hdf5.read.mmap", true);
	protected static final boolean HDF5_SKIP_NNZ =
		getHdf5ReadBoolean("sysds.hdf5.read.skip.nnz", false);
	protected static final boolean HDF5_FORCE_DENSE =
		getHdf5ReadBoolean("sysds.hdf5.read.force.dense", false);
	protected static final boolean HDF5_READ_TRACE =
		getHdf5ReadBoolean("sysds.hdf5.read.trace", false);
	protected static final int HDF5_READ_PARALLEL_THREADS = Math.max(1,
		getHdf5ReadInt("sysds.hdf5.read.parallel.threads",
			OptimizerUtils.getParallelBinaryReadParallelism()));
	protected static final int HDF5_READ_PARALLEL_MIN_BYTES =
		getHdf5ReadInt("sysds.hdf5.read.parallel.min.bytes", DEFAULT_HDF5_READ_PARALLEL_MIN_BYTES);

	public ReaderHDF5(FileFormatPropertiesHDF5 props) {
		_props = props;
	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {
		//allocate output matrix block
		MatrixBlock ret = null;
		if(rlen >= 0 && clen >= 0) //otherwise allocated on read
			ret = createOutputMatrixBlock(rlen, clen, (int) rlen, estnnz, true, true);

		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		//check existence and non-empty file
		checkValidInputFile(fs, path);

		//core read
		ret = readHDF5MatrixFromHDFS(path, job, fs, ret, rlen, clen, blen, _props.getDatasetName());

		//finally check if change of sparse/dense block representation required
		//(nnz explicitly maintained during read)
		ret.examSparsity();

		return ret;
	}

	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, (int) rlen, estnnz, true, false);

		//core read
		String datasetName = _props.getDatasetName();

		H5ByteReader byteReader = createByteReader(is, "input-stream(" + datasetName + ")", -1);
		long lnnz = readMatrixFromHDF5(byteReader, datasetName, ret, 0, rlen, clen, blen);

		//finally check if change of sparse/dense block representation required
		ret.setNonZeros(lnnz);
		ret.examSparsity();

		return ret;
	}

	static H5ByteReader createByteReader(InputStream is, String sourceId) throws IOException {
		return createByteReader(is, sourceId, -1);
	}

	static H5ByteReader createByteReader(InputStream is, String sourceId, long lengthHint) throws IOException {
		long length = lengthHint;
		if(is instanceof FSDataInputStream) {
			LOG.trace("[HDF5] Using FSDataInputStream-backed reader for " + sourceId);
			H5ByteReader base = new FsDataInputStreamByteReader((FSDataInputStream) is);
			if(length > 0 && length <= Integer.MAX_VALUE) {
				return new BufferedH5ByteReader(base, length, HDF5_READ_BUFFER_BYTES);
			}
			return base;
		}
		else if(is instanceof FileInputStream) {
			FileChannel channel = ((FileInputStream) is).getChannel();
			length = channel.size();
			LOG.trace("[HDF5] Using FileChannel-backed reader for " + sourceId + " (size=" + length + ")");
			if(HDF5_READ_USE_MMAP && length > 0) {
				return new MappedH5ByteReader(channel, length, HDF5_READ_MAP_BYTES);
			}
			H5ByteReader base = new FileChannelByteReader(channel);
			if(length > 0 && length <= Integer.MAX_VALUE) {
				return new BufferedH5ByteReader(base, length, HDF5_READ_BUFFER_BYTES);
			}
			return base;
		}
		else {
			byte[] cached = drainToByteArray(is);
			LOG.trace("[HDF5] Cached " + cached.length + " bytes into memory for " + sourceId);
			return new BufferedH5ByteReader(new ByteArrayH5ByteReader(cached), cached.length, HDF5_READ_BUFFER_BYTES);
		}
	}

	private static byte[] drainToByteArray(InputStream is) throws IOException {
		try(InputStream input = is; ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
			byte[] buff = new byte[8192];
			int len;
			while((len = input.read(buff)) != -1)
				bos.write(buff, 0, len);
			return bos.toByteArray();
		}
	}

	private static MatrixBlock readHDF5MatrixFromHDFS(Path path, JobConf job,
		FileSystem fs, MatrixBlock dest, long rlen, long clen, int blen, String datasetName)
		throws IOException, DMLRuntimeException
	{
		//prepare file paths in alphanumeric order
		ArrayList<Path> files = new ArrayList<>();
		if(fs.getFileStatus(path).isDirectory()) {
			for(FileStatus stat : fs.listStatus(path, IOUtilFunctions.hiddenFileFilter))
				files.add(stat.getPath());
			Collections.sort(files);
		}
		else
			files.add(path);

		//determine matrix size via additional pass if required
		if(dest == null) {
			dest = computeHDF5Size(files, fs, datasetName, rlen*clen);
			clen = dest.getNumColumns();
			rlen = dest.getNumRows();
		}

		//actual read of individual files
		long lnnz = 0;
		for(int fileNo = 0; fileNo < files.size(); fileNo++) {
			H5ByteReader byteReader = createByteReader(files.get(fileNo), fs);
			lnnz += readMatrixFromHDF5(byteReader, datasetName, dest, 0, rlen, clen, blen);
		}
		//post processing
		dest.setNonZeros(lnnz);

		return dest;
	}

	public static long readMatrixFromHDF5(H5ByteReader byteReader, String datasetName, MatrixBlock dest,
		int rl, long ru, long clen, int blen)
	{
		long lnnz = 0;
		boolean skipNnz = HDF5_SKIP_NNZ && !dest.isInSparseFormat();
		if(HDF5_FORCE_DENSE && dest.isInSparseFormat()) {
			dest.allocateDenseBlock(true);
			skipNnz = HDF5_SKIP_NNZ;
			if(HDF5_READ_TRACE)
				LOG.trace("[HDF5] Forcing dense output for dataset=" + datasetName);
		}
		H5RootObject rootObject = H5.H5Fopen(byteReader);
		H5ContiguousDataset contiguousDataset = H5.H5Dopen(rootObject, datasetName);

		int ncol = (int) rootObject.getCol();
		LOG.trace("[HDF5] readMatrix dataset=" + datasetName + " dims=" + rootObject.getRow() + "x"
			+ rootObject.getCol() + " loop=[" + rl + "," + ru + ") dest=" + dest.getNumRows() + "x"
			+ dest.getNumColumns());

		try {
			double[] row = null;
			double[] blockBuffer = null;
			int[] ixBuffer = null;
			double[] valBuffer = null;
			long elemSize = contiguousDataset.getDataType().getDoubleDataType().getSize();
			long rowBytes = (long) ncol * elemSize;
			if(rowBytes > Integer.MAX_VALUE) {
				throw new DMLRuntimeException("HDF5 row size exceeds buffer capacity: " + rowBytes);
			}
			int blockRows = 1;
			if(!contiguousDataset.isRankGt2() && rowBytes > 0) {
				blockRows = (int) Math.max(1, HDF5_READ_BLOCK_BYTES / rowBytes);
			}
			if( dest.isInSparseFormat() ) {
				SparseBlock sb = dest.getSparseBlock();
				if(contiguousDataset.isRankGt2()) {
					row = new double[ncol];
					for(int i = rl; i < ru; i++) {
						contiguousDataset.readRowDoubles(i, row, 0);
						int lnnzi = UtilFunctions.computeNnz(row, 0, ncol);
						sb.allocate(i, lnnzi); //avoid row reallocations
						for(int j = 0; j < ncol; j++)
							sb.append(i, j, row[j]); //prunes zeros
						lnnz += lnnzi;
					}
				}
				else {
					ixBuffer = new int[ncol];
					valBuffer = new double[ncol];
					for(int i = rl; i < ru; ) {
						int rowsToRead = (int) Math.min(blockRows, ru - i);
						ByteBuffer buffer = contiguousDataset.getDataBuffer(i, rowsToRead);
						DoubleBuffer db = buffer.order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
						int blockSize = rowsToRead * ncol;
						if(blockBuffer == null || blockBuffer.length < blockSize) {
							blockBuffer = new double[blockSize];
						}
						db.get(blockBuffer, 0, blockSize);
						for(int r = 0; r < rowsToRead; r++, i++) {
							int base = r * ncol;
							int lnnzi = 0;
							for(int j = 0; j < ncol; j++) {
								double v = blockBuffer[base + j];
								if(v != 0) {
									ixBuffer[lnnzi] = j;
									valBuffer[lnnzi] = v;
									lnnzi++;
								}
							}
							sb.allocate(i, lnnzi); //avoid row reallocations
							for(int k = 0; k < lnnzi; k++) {
								sb.append(i, ixBuffer[k], valBuffer[k]);
							}
							lnnz += lnnzi;
						}
					}
				}
			}
			else {
				DenseBlock denseBlock = dest.getDenseBlock();
				boolean fastDense = denseBlock.isNumeric(ValueType.FP64)
					&& !(denseBlock instanceof DenseBlockFP64DEDUP)
					&& !(denseBlock instanceof DenseBlockLFP64DEDUP);
				if(contiguousDataset.isRankGt2()) {
					row = new double[ncol];
					for(int i = rl; i < ru; i++) {
						if(fastDense) {
							double[] destRow = denseBlock.values(i);
							int destPos = denseBlock.pos(i);
							contiguousDataset.readRowDoubles(i, destRow, destPos);
							if(!skipNnz)
								lnnz += UtilFunctions.computeNnz(destRow, destPos, ncol);
						}
						else {
							contiguousDataset.readRowDoubles(i, row, 0);
							denseBlock.set(i, row);
							if(!skipNnz)
								lnnz += UtilFunctions.computeNnz(row, 0, ncol);
						}
					}
				}
				else {
					boolean contiguousDense = fastDense && denseBlock.isContiguous();
					double[] destAll = contiguousDense ? denseBlock.values(0) : null;
					for(int i = rl; i < ru; ) {
						int rowsToRead = (int) Math.min(blockRows, ru - i);
						ByteBuffer buffer = contiguousDataset.getDataBuffer(i, rowsToRead);
						DoubleBuffer db = buffer.order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
						int blockSize = rowsToRead * ncol;
						if(contiguousDense) {
							int destPos = denseBlock.pos(i);
							db.get(destAll, destPos, blockSize);
							if(!skipNnz)
								lnnz += UtilFunctions.computeNnz(destAll, destPos, blockSize);
							i += rowsToRead;
							continue;
						}
						if(fastDense) {
							if(blockBuffer == null || blockBuffer.length < blockSize) {
								blockBuffer = new double[blockSize];
							}
							db.get(blockBuffer, 0, blockSize);
							for(int r = 0; r < rowsToRead; r++, i++) {
								double[] destRow = denseBlock.values(i);
								int destPos = denseBlock.pos(i);
								System.arraycopy(blockBuffer, r * ncol, destRow, destPos, ncol);
							}
							if(!skipNnz)
								lnnz += UtilFunctions.computeNnz(blockBuffer, 0, blockSize);
							continue;
						}
						for(int r = 0; r < rowsToRead; r++, i++) {
							if(row == null) {
								row = new double[ncol];
							}
							db.get(row, 0, ncol);
							denseBlock.set(i, row);
							if(!skipNnz)
								lnnz += UtilFunctions.computeNnz(row, 0, ncol);
						}
					}
				}
			}
		}
		finally {
			rootObject.close();
		}
		if(skipNnz) {
			lnnz = Math.multiplyExact(ru - rl, clen);
		}
		return lnnz;
	}

	public static MatrixBlock computeHDF5Size(List<Path> files, FileSystem fs, String datasetName, long estnnz)
		throws IOException, DMLRuntimeException
	{
		int nrow = 0;
		int ncol = 0;
		for(int fileNo = 0; fileNo < files.size(); fileNo++) {
			H5ByteReader byteReader = createByteReader(files.get(fileNo), fs);
			H5RootObject rootObject = H5.H5Fopen(byteReader);
			H5.H5Dopen(rootObject, datasetName);

			nrow += (int) rootObject.getRow();
			ncol += (int) rootObject.getCol();

			rootObject.close();
		}
		// allocate target matrix block based on given size;
		return createOutputMatrixBlock(nrow, ncol, nrow, estnnz, true, true);
	}

	private static int getHdf5ReadInt(String key, int defaultValue) {
		String value = System.getProperty(key);
		if(value == null)
			return defaultValue;
		try {
			long parsed = Long.parseLong(value.trim());
			if(parsed <= 0 || parsed > Integer.MAX_VALUE)
				return defaultValue;
			return (int) parsed;
		}
		catch(NumberFormatException ex) {
			return defaultValue;
		}
	}

	private static boolean getHdf5ReadBoolean(String key, boolean defaultValue) {
		String value = System.getProperty(key);
		if(value == null)
			return defaultValue;
		return Boolean.parseBoolean(value.trim());
	}

	static java.io.File getLocalFile(Path path) {
		try {
			return new java.io.File(path.toUri());
		}
		catch(IllegalArgumentException ex) {
			return new java.io.File(path.toString());
		}
	}

	private static ByteBuffer sliceBuffer(ByteBuffer source, int offset, int length) {
		ByteBuffer dup = source.duplicate();
		dup.position(offset);
		dup.limit(offset + length);
		return dup.slice();
	}

	static boolean isLocalFileSystem(FileSystem fs) {
		if(fs instanceof LocalFileSystem || fs instanceof RawLocalFileSystem)
			return true;
		String scheme = fs.getScheme();
		return scheme != null && scheme.equalsIgnoreCase("file");
	}

	static H5ByteReader createByteReader(Path path, FileSystem fs) throws IOException {
		long fileLength = fs.getFileStatus(path).getLen();
		String sourceId = path.toString();
		if(isLocalFileSystem(fs)) {
			FileInputStream fis = new FileInputStream(getLocalFile(path));
			FileChannel channel = fis.getChannel();
			long length = channel.size();
			LOG.trace("[HDF5] Using FileChannel-backed reader for " + sourceId + " (size=" + length + ")");
			if(HDF5_READ_USE_MMAP && length > 0) {
				return new MappedH5ByteReader(channel, length, HDF5_READ_MAP_BYTES);
			}
			H5ByteReader base = new FileChannelByteReader(channel);
			if(length > 0 && length <= Integer.MAX_VALUE) {
				return new BufferedH5ByteReader(base, length, HDF5_READ_BUFFER_BYTES);
			}
			return base;
		}
		FSDataInputStream fsin = fs.open(path);
		return createByteReader(fsin, sourceId, fileLength);
	}

	private static final class FsDataInputStreamByteReader implements H5ByteReader {
		private final FSDataInputStream input;

		FsDataInputStreamByteReader(FSDataInputStream input) {
			this.input = input;
		}

		@Override
		public ByteBuffer read(long offset, int length) throws IOException {
			byte[] buffer = new byte[length];
			input.readFully(offset, buffer, 0, length);
			return ByteBuffer.wrap(buffer);
		}

		@Override
		public ByteBuffer read(long offset, int length, ByteBuffer reuse) throws IOException {
			if(reuse == null || reuse.capacity() < length || !reuse.hasArray()) {
				return read(offset, length);
			}
			byte[] buffer = reuse.array();
			int baseOffset = reuse.arrayOffset();
			input.readFully(offset, buffer, baseOffset, length);
			reuse.position(baseOffset);
			reuse.limit(baseOffset + length);
			if(baseOffset == 0) {
				return reuse;
			}
			return reuse.slice();
		}

		@Override
		public void close() throws IOException {
			input.close();
		}
	}

	private static final class BufferedH5ByteReader implements H5ByteReader {
		private final H5ByteReader base;
		private final long length;
		private final int windowSize;
		private long windowStart = -1;
		private int windowLength;
		private ByteBuffer window;
		private ByteBuffer windowStorage;

		BufferedH5ByteReader(H5ByteReader base, long length, int windowSize) {
			this.base = base;
			this.length = length;
			this.windowSize = windowSize;
		}

		@Override
		public ByteBuffer read(long offset, int length) throws IOException {
			if(length <= 0 || length > windowSize) {
				return base.read(offset, length);
			}
			if(this.length > 0 && offset + length > this.length) {
				return base.read(offset, length);
			}
			if(window != null && offset >= windowStart && offset + length <= windowStart + windowLength) {
				return sliceBuffer(window, (int) (offset - windowStart), length);
			}
			int readSize = windowSize;
			if(this.length > 0) {
				long remaining = this.length - offset;
				if(remaining > 0)
					readSize = (int) Math.min(readSize, remaining);
			}
			if(readSize < length) {
				readSize = length;
			}
			if(windowStorage == null || windowStorage.capacity() < readSize) {
				windowStorage = ByteBuffer.allocate(windowSize);
			}
			window = base.read(offset, readSize, windowStorage);
			windowStart = offset;
			windowLength = window.remaining();
			return sliceBuffer(window, 0, length);
		}

		@Override
		public void close() throws IOException {
			base.close();
		}
	}

	private static final class FileChannelByteReader implements H5ByteReader {
		private final FileChannel channel;

		FileChannelByteReader(FileChannel channel) {
			this.channel = channel;
		}

		@Override
		public ByteBuffer read(long offset, int length) throws IOException {
			ByteBuffer buffer = ByteBuffer.allocate(length);
			long pos = offset;
			while(buffer.hasRemaining()) {
				int read = channel.read(buffer, pos);
				if(read < 0)
					throw new IOException("Unexpected EOF while reading HDF5 data at offset " + offset);
				pos += read;
			}
			buffer.flip();
			return buffer;
		}

		@Override
		public ByteBuffer read(long offset, int length, ByteBuffer reuse) throws IOException {
			if(reuse == null || reuse.capacity() < length) {
				return read(offset, length);
			}
			reuse.clear();
			reuse.limit(length);
			long pos = offset;
			while(reuse.hasRemaining()) {
				int read = channel.read(reuse, pos);
				if(read < 0)
					throw new IOException("Unexpected EOF while reading HDF5 data at offset " + offset);
				pos += read;
			}
			reuse.flip();
			return reuse;
		}

		@Override
		public void close() throws IOException {
			channel.close();
		}
	}

	private static final class MappedH5ByteReader implements H5ByteReader {
		private final FileChannel channel;
		private final long length;
		private final int windowSize;
		private long windowStart = -1;
		private int windowLength;
		private MappedByteBuffer window;

		MappedH5ByteReader(FileChannel channel, long length, int windowSize) {
			this.channel = channel;
			this.length = length;
			this.windowSize = windowSize;
		}

		@Override
		public ByteBuffer read(long offset, int length) throws IOException {
			if(length <= 0)
				return ByteBuffer.allocate(0);
			if(this.length > 0 && offset + length > this.length) {
				throw new IOException("Attempted to read past EOF at offset " + offset + " length " + length);
			}
			if(length > windowSize) {
				MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, offset, length);
				return mapped;
			}
			if(window != null && offset >= windowStart && offset + length <= windowStart + windowLength) {
				return sliceBuffer(window, (int) (offset - windowStart), length);
			}
			int readSize = windowSize;
			if(this.length > 0) {
				long remaining = this.length - offset;
				if(remaining > 0)
					readSize = (int) Math.min(readSize, remaining);
			}
			if(readSize < length) {
				readSize = length;
			}
			window = channel.map(FileChannel.MapMode.READ_ONLY, offset, readSize);
			windowStart = offset;
			windowLength = readSize;
			return sliceBuffer(window, 0, length);
		}

		@Override
		public void close() throws IOException {
			channel.close();
		}
	}

	private static final class ByteArrayH5ByteReader implements H5ByteReader {
		private final byte[] data;

		ByteArrayH5ByteReader(byte[] data) {
			this.data = data;
		}

		@Override
		public ByteBuffer read(long offset, int length) throws IOException {
			if(offset < 0 || offset + length > data.length) {
				throw new IOException("Attempted to read outside cached buffer (offset=" + offset + ", len=" + length
					+ ", size=" + data.length + ")");
			}
			if(offset > Integer.MAX_VALUE) {
				throw new IOException("Offset exceeds byte array capacity: " + offset);
			}
			return ByteBuffer.wrap(data, (int) offset, length).slice();
		}

		@Override
		public void close() {
			// nothing to close
		}
	}
}

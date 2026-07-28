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
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.io.hdf5.H5;
import org.apache.sysds.runtime.io.hdf5.H5RootObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.HDFSTool;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.util.Arrays;

public class WriterHDF5 extends MatrixWriter {

	private static final int DEFAULT_HDF5_WRITE_BATCH_ROWS = 1024;
	private static final int DEFAULT_HDF5_WRITE_BATCH_BYTES = 1024 * 1024;

	private static final int HDF5_WRITE_BATCH_ROWS =
		getHdf5WriteInt("sysds.hdf5.write.batch.rows", DEFAULT_HDF5_WRITE_BATCH_ROWS);

	private static final int HDF5_WRITE_BATCH_BYTES =
		getHdf5WriteInt("sysds.hdf5.write.batch.bytes", DEFAULT_HDF5_WRITE_BATCH_BYTES);

	private static final String HDF5_WRITE_SPARSE_LAYOUT =
		System.getProperty("sysds.hdf5.write.sparse.layout", "dense");

	private static int getHdf5WriteInt(String key, int defaultValue) {
		String value = System.getProperty(key);
		if(value == null)
			return defaultValue;

		try {
			int parsed = Integer.parseInt(value.trim());
			return parsed > 0 ? parsed : defaultValue;
		}
		catch(NumberFormatException ex) {
			return defaultValue;
		}
	}

	protected static FileFormatPropertiesHDF5 _props = null;

	public WriterHDF5(FileFormatPropertiesHDF5 _props) {
		WriterHDF5._props = _props;
	}

	private static boolean useSparseCOO(MatrixBlock src) {
		return src.isInSparseFormat()
			&& "coo".equalsIgnoreCase(HDF5_WRITE_SPARSE_LAYOUT);
	}

	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int blen, long nnz, boolean diag)
		throws IOException, DMLRuntimeException
	{
		//validity check matrix dimensions
		if(src.getNumRows() != rlen || src.getNumColumns() != clen)
			throw new IOException("Matrix dimensions mismatch with metadata: " + src.getNumRows() + "x" + src
				.getNumColumns() + " vs " + rlen + "x" + clen + ".");
		if(rlen == 0 || clen == 0)
			throw new IOException(
				"Write of matrices with zero rows or columns not supported (" + rlen + "x" + clen + ").");

		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		//if the file already exists on HDFS, remove it.
		HDFSTool.deleteFileIfExistOnHDFS(fname);

		if(useSparseCOO(src))
			writeSparseCOOMatrixToFile(path, fs, src, rlen, clen, nnz);
		else
			writeHDF5MatrixToHDFS(path, job, fs, src);

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	@Override
	public final void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen)
		throws IOException, DMLRuntimeException 
	{
		throw new DMLRuntimeException("writing empty HDF5 matrices not supported yet");
	}

	protected void writeHDF5MatrixToHDFS(Path path, JobConf job, FileSystem fs, MatrixBlock src) 
		throws IOException
	{
		writeHDF5MatrixToFile(path, job, fs, src, 0, src.getNumRows());
	}

	protected static void writeHDF5MatrixToFile(Path path, JobConf job, FileSystem fs, MatrixBlock src, int rl, int rlen) 
		throws IOException 
	{
		int clen = src.getNumColumns();
		String datasetName = _props.getDatasetName();
		
		try(BufferedOutputStream bos = new BufferedOutputStream(fs.create(path, true))) {
			H5RootObject rootObject = H5.H5Screate(bos, src.getNumRows(), src.getNumColumns());
			H5.H5Dcreate(rootObject, src.getNumRows(), src.getNumColumns(), datasetName);

			if(rl == 0)
				H5.H5WriteHeaders(rootObject);

			int batchRows = getWriteBatchRows(clen);
			if(src.isInSparseFormat())
				writeSparseBatched(rootObject, src, rl, rlen, clen, batchRows);
			else
				writeDenseBatched(rootObject, src, rl, rlen, clen, batchRows);
		}
	}

	private static int getWriteBatchRows(int clen) {
		long rowBytes = (long) clen * Double.BYTES;

		int rowsByBytes = rowBytes > 0 ? (int) Math.max(1, HDF5_WRITE_BATCH_BYTES / rowBytes) : 1;

		int rows = Math.max(1, Math.min(HDF5_WRITE_BATCH_ROWS, rowsByBytes));
		rows = roundDownPowerOfTwo(rows);
		long cells = (long) rows * clen;

		if(cells > Integer.MAX_VALUE)
			throw new DMLRuntimeException("HDF5 write batch too large: " + rows + " x " + clen);

		return rows;
	}

	private static int roundDownPowerOfTwo(int value) {
		int ret = 1;
		while(ret <= value / 2)
			ret *= 2;
		return ret;
	}

	private static void writeDenseBatched(H5RootObject rootObject, MatrixBlock src, int rl, int ru, int clen, int batchRows) {

		DenseBlock db = src.getDenseBlock();
		double[] batch = new double[batchRows * clen];

		for(int rowStart = rl; rowStart < ru; rowStart += batchRows) {
			int rows = Math.min(batchRows, ru - rowStart);

			for(int r = 0; r < rows; r++) {
				int srcRow = rowStart + r;
				int off = r * clen;

				for(int c = 0; c < clen; c++)
					batch[off + c] = db.get(srcRow, c);
			}

			if(rows == batchRows)
				H5.H5Dwrite(rootObject, batch);
			else
				H5.H5Dwrite(rootObject, Arrays.copyOf(batch, rows * clen));
		}
	}

	private static void writeSparseBatched(H5RootObject rootObject, MatrixBlock src, int rl, int ru, int clen, int batchRows) {
		SparseBlock sb = src.getSparseBlock();
		double[] batch = new double[batchRows * clen];

		for(int rowStart = rl; rowStart < ru; rowStart += batchRows) {
			int rows = Math.min(batchRows, ru - rowStart);
			Arrays.fill(batch, 0, rows * clen, 0.0);

			for(int r = 0; r < rows; r++) {
				int srcRow = rowStart + r;

				if(sb == null || sb.isEmpty(srcRow))
					continue;

				int apos = sb.pos(srcRow);
				int alen = sb.size(srcRow);
				int[] aix = sb.indexes(srcRow);
				double[] avals = sb.values(srcRow);

				int off = r * clen;
				for(int k = apos; k < apos + alen; k++)
					batch[off + aix[k]] = avals[k];
			}

			if(rows == batchRows)
				H5.H5Dwrite(rootObject, batch);
			else
				H5.H5Dwrite(rootObject, Arrays.copyOf(batch, rows * clen));
		}
	}

	private static void writeSparseCOOMatrixToFile(Path path, FileSystem fs, MatrixBlock src, long rlen, long clen, long nnz) throws IOException {
		String datasetName = _props.getDatasetName();

		long cooRows = nnz + 1;
		long cooCols = 3;

		try(BufferedOutputStream bos = new BufferedOutputStream(fs.create(path, true))) {
			H5RootObject rootObject = H5.H5Screate(bos, cooRows, cooCols);
			H5.H5Dcreate(rootObject, cooRows, cooCols, datasetName);
			H5.H5WriteHeaders(rootObject);

			H5.H5Dwrite(rootObject, new double[] {
				(double) rlen,
				(double) clen,
				(double) nnz
			});

			writeSparseCOOEntries(rootObject, src);
		}
	}

	private static void writeSparseCOOEntries(H5RootObject rootObject, MatrixBlock src) {
		SparseBlock sb = src.getSparseBlock();
		int batchRows = getWriteBatchRows(3);
		double[] batch = new double[batchRows * 3];

		int pos = 0;
		for(int i = 0; i < src.getNumRows(); i++) {
			if(sb == null || sb.isEmpty(i))
				continue;

			int apos = sb.pos(i);
			int alen = sb.size(i);
			int[] aix = sb.indexes(i);
			double[] avals = sb.values(i);

			for(int k = apos; k < apos + alen; k++) {
				batch[pos++] = i;
				batch[pos++] = aix[k];
				batch[pos++] = avals[k];

				if(pos == batch.length) {
					H5.H5Dwrite(rootObject, batch);
					pos = 0;
				}
			}
		}

		if(pos > 0)
			H5.H5Dwrite(rootObject, Arrays.copyOf(batch, pos));
	}

	@Override
	public long writeMatrixFromStream(String fname, OOCStream<IndexedMatrixValue> stream, long rlen, long clen, int blen) {
		throw new UnsupportedOperationException("Writing from an OOC stream is not supported for the HDF5 format.");
	};
}

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
import org.apache.sysds.runtime.io.hdf5.H5;
import org.apache.sysds.runtime.io.hdf5.H5RootObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.HDFSTool;

import java.io.BufferedOutputStream;
import java.io.IOException;

public class WriterHDF5 extends MatrixWriter {

	protected static FileFormatPropertiesHDF5 _props = null;

	public WriterHDF5(FileFormatPropertiesHDF5 _props) {
		WriterHDF5._props = _props;
	}

	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int blen, long nnz,
		boolean diag) throws IOException, DMLRuntimeException {

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

		//core write (sequential/parallel)
		writeHDF5MatrixToHDFS(path, job, fs, src);

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);

	}

	@Override
	public final void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen)
		throws IOException, DMLRuntimeException {

	}

	protected void writeHDF5MatrixToHDFS(Path path, JobConf job, FileSystem fs, MatrixBlock src) throws IOException {
		//sequential write HDF5 file
		writeHDF5MatrixToFile(path, job, fs, src, 0, src.getNumRows());
	}

	protected static void writeHDF5MatrixToFile(Path path, JobConf job, FileSystem fs, MatrixBlock src, int rl,
		int rlen) throws IOException {

		int clen = src.getNumColumns();
		BufferedOutputStream bos = new BufferedOutputStream(fs.create(path, true));
		String datasetName = _props.getDatasetName();
		H5RootObject rootObject = H5.H5Screate(bos, src.getNumRows(), src.getNumColumns());
		H5.H5Dcreate(rootObject, src.getNumRows(), src.getNumColumns(), datasetName);

		//write headers
		if(rl == 0) {
			H5.H5WriteHeaders(rootObject);
		}

		try {
			//TODO: HDF5 format don't support spars matrix
			// How to store spars matrix in HDF5 format?

			// Write the data to the datasets.
			double[] data = new double[clen];
			for(int i = rl; i < rlen; i++) {
				for(int j = 0; j < clen;j++) {
					double lvalue = src.getValueDenseUnsafe(i, j);
					data[j] = lvalue;
				}
				H5.H5Dwrite(rootObject, data);
			}
		}
		finally {
			IOUtilFunctions.closeSilently(bos);
		}

	}
}

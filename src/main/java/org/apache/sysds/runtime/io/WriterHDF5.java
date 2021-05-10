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

import hdf.hdf5lib.H5;
import hdf.hdf5lib.HDF5Constants;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.HDFSTool;
import java.io.IOException;

public class WriterHDF5 extends MatrixWriter {

	protected static FileFormatPropertiesHDF5 _props = null;
	public WriterHDF5(FileFormatPropertiesHDF5 _props) {
		WriterHDF5._props = _props;
	}

	public WriterHDF5() {
	}

	@Override public final void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int blen,
		long nnz, boolean diag) throws IOException, DMLRuntimeException {
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

	@Override public final void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen)
		throws IOException, DMLRuntimeException {

	}

	protected void writeHDF5MatrixToHDFS(Path path, JobConf job, FileSystem fs, MatrixBlock src) throws IOException {
		//sequential write libsvm file
		writeHDF5MatrixToFile(path, job, fs, src, 0, src.getNumRows());
	}

	protected static void writeHDF5MatrixToFile(Path path, JobConf job, FileSystem fs, MatrixBlock src, int rl,
		int rlen) throws IOException {
		boolean sparse = src.isInSparseFormat();
		int clen = src.getNumColumns();
		long[] dims = {rlen, clen};
		int rank = 2;

		// DATA SET NAME SHOULD READ FROM INPUT (file properties)
		String datasetName = "dst";

		long fileID = HDF5Constants.H5I_INVALID_HID;
		long fileSpaceID = HDF5Constants.H5I_INVALID_HID;
		long datasetID = HDF5Constants.H5I_INVALID_HID;
		long datasetPropertyID = HDF5Constants.H5I_INVALID_HID;

		try {
			// Create a new file using default properties.
			fileID = H5.H5Fcreate(path.getName(), HDF5Constants.H5F_ACC_TRUNC, HDF5Constants.H5P_DEFAULT,
				HDF5Constants.H5P_DEFAULT);

			// Create the data space for the dataset.
			fileSpaceID = H5.H5Screate_simple(rank, dims, dims);

			// Create the dataset creation property list, and set the chunk size.
			datasetPropertyID = H5.H5Pcreate(HDF5Constants.H5P_DATASET_CREATE);

			// Set the allocation time to "early". This way we can be sure
			// that reading from the dataset immediately after creation will
			// return the fill value.
			if(datasetPropertyID >= 0)
				H5.H5Pset_alloc_time(datasetPropertyID, HDF5Constants.H5D_ALLOC_TIME_EARLY);

			// Create the dataset using the dataset default creation property list.
			if((fileID >= 0) && (fileSpaceID >= 0))
				datasetID = H5.H5Dcreate(fileID, datasetName, HDF5Constants.H5T_NATIVE_DOUBLE, fileSpaceID,
					HDF5Constants.H5P_DEFAULT, HDF5Constants.H5P_DEFAULT, HDF5Constants.H5P_DEFAULT);

			// Write the data to the datasets.
			if(datasetID >= 0)
				H5.H5Dwrite(datasetID, HDF5Constants.H5T_NATIVE_DOUBLE, HDF5Constants.H5S_ALL, HDF5Constants.H5S_ALL,
					HDF5Constants.H5P_DEFAULT, datasetName);
		}
		finally {
			if(datasetPropertyID >= 0)
				H5.H5Pclose(datasetPropertyID);

			if(datasetID >= 0)
				H5.H5Dclose(datasetID);

			if(fileSpaceID >= 0)
				H5.H5Sclose(fileSpaceID);

			if(fileID >= 0)
				H5.H5Fclose(fileID);
		}
	}
}

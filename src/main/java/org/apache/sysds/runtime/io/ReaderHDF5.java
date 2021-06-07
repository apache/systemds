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

import org.apache.commons.lang.mutable.MutableInt;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.io.hdf5.H5;
import org.apache.sysds.runtime.io.hdf5.H5ContiguousDataset;
import org.apache.sysds.runtime.io.hdf5.H5RootObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ReaderHDF5 extends MatrixReader {
	protected final FileFormatPropertiesHDF5 _props;

	public ReaderHDF5(FileFormatPropertiesHDF5 props) {
		_props = props;
	}

	@Override public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {
		//allocate output matrix block
		MatrixBlock ret = null;
		if(rlen >= 0 && clen >= 0) //otherwise allocated on read
			ret = createOutputMatrixBlock(rlen, clen, (int) rlen, estnnz, true, false);

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

	@Override public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, (int) rlen, estnnz, true, false);

		//core read
		String datasetName = _props.getDatasetName();

		BufferedInputStream bis = new BufferedInputStream(is);

		long lnnz = readMatrixFromHDF5(bis, datasetName, ret, new MutableInt(0), rlen, clen, blen);

		//finally check if change of sparse/dense block representation required
		ret.setNonZeros(lnnz);
		ret.examSparsity();

		return ret;
	}

	@SuppressWarnings("unchecked") private static MatrixBlock readHDF5MatrixFromHDFS(Path path, JobConf job,
		FileSystem fs, MatrixBlock dest, long rlen, long clen, int blen, String datasetName)
		throws IOException, DMLRuntimeException {
		//prepare file paths in alphanumeric order
		ArrayList<Path> files = new ArrayList<>();
		if(fs.isDirectory(path)) {
			for(FileStatus stat : fs.listStatus(path, IOUtilFunctions.hiddenFileFilter))
				files.add(stat.getPath());
			Collections.sort(files);
		}
		else
			files.add(path);

		//determine matrix size via additional pass if required
		if(dest == null) {
			dest = computeHDF5Size(files, fs, datasetName);
			clen = dest.getNumColumns();
			rlen = dest.getNumRows();
		}

		//actual read of individual files
		long lnnz = 0;
		MutableInt row = new MutableInt(0);
		for(int fileNo = 0; fileNo < files.size(); fileNo++) {
			BufferedInputStream bis = new BufferedInputStream(fs.open(files.get(fileNo)));
			lnnz += readMatrixFromHDF5(bis, datasetName, dest, row, rlen, clen, blen);
		}
		//post processing
		dest.setNonZeros(lnnz);

		return dest;
	}

	public static long readMatrixFromHDF5(BufferedInputStream bis, String datasetName, MatrixBlock dest,
		MutableInt rowPos, long rlen, long clen, int blen) {
		int row = rowPos.intValue();
		long lnnz = 0;
		H5RootObject rootObject = H5.H5Fopen(bis);
		H5ContiguousDataset contiguousDataset = H5.H5Dopen(rootObject, datasetName);

		int[] dims = rootObject.getDimensions();
		int ncol = dims[1];

		DenseBlock denseBlock = dest.getDenseBlock();
		for(int i = row; i < rlen; i++) {
			double[] rowData = H5.H5Dread(rootObject, contiguousDataset, i);
			for(int j = 0; j < ncol; j++) {
				if(rowData[j] != 0) {
					denseBlock.set(i, j, rowData[j]);
					lnnz++;
				}
			}
			row++;
		}

		rowPos.setValue(row);
		return lnnz;
	}

	public static MatrixBlock computeHDF5Size(List<Path> files, FileSystem fs, String datasetName)
		throws IOException, DMLRuntimeException {
		int nrow = 0;
		int ncol = 0;

		for(int fileNo = 0; fileNo < files.size(); fileNo++) {

			BufferedInputStream bis = new BufferedInputStream(fs.open(files.get(fileNo)));
			H5RootObject rootObject = H5.H5Fopen(bis);
			H5.H5Dopen(rootObject, datasetName);

			int[] dims = rootObject.getDimensions();
			nrow += dims[0];
			ncol += dims[1];
		}
		// allocate target matrix block based on given size;
		return createOutputMatrixBlock(nrow, ncol, nrow, (long) nrow * ncol, true, false);
	}
}

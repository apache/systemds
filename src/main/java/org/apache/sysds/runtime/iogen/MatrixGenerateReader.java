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

package org.apache.sysds.runtime.iogen;

import org.apache.commons.lang.mutable.MutableInt;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class MatrixGenerateReader extends MatrixReader {

	protected static CustomProperties _props;
	protected final FastStringTokenizer fastStringTokenizerDelim;

	public MatrixGenerateReader(CustomProperties _props) {
		MatrixGenerateReader._props = _props;
		fastStringTokenizerDelim = new FastStringTokenizer(_props.getDelim());
	}

	protected MatrixBlock computeSize(List<Path> files, FileSystem fs, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		// allocate target matrix block based on given size;
		return new MatrixBlock(getNumRows(files, fs), (int) clen, rlen * clen);
	}

	private static int getNumRows(List<Path> files, FileSystem fs) throws IOException, DMLRuntimeException {
		int rows = 0;
		String value;
		for(int fileNo = 0; fileNo < files.size(); fileNo++) {
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));
			try {
				// Row Regular
				if(_props.getRowPattern().equals(CustomProperties.GRPattern.Regular)) {
					// TODO: check the file has header?
					while(br.readLine() != null)
						rows++;
				}
				// Row Irregular
				else {
					FastStringTokenizer st = new FastStringTokenizer(_props.getDelim());
					while((value = br.readLine()) != null) {
						st.reset(value);
						int row = st.nextInt();
						rows = Math.max(rows, row);
					}
					rows++;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(br);
			}
		}
		return rows;
	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {

		MatrixBlock ret = null;
		if(rlen >= 0 && clen >= 0) //otherwise allocated on read
			ret = createOutputMatrixBlock(rlen, clen, (int) rlen, estnnz, true, false);

		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		//core read
		ret = readMatrixFromHDFS(path, job, fs, ret, rlen, clen, blen);

		return ret;
	}

	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {

		MatrixBlock ret = null;
		if(rlen >= 0 && clen >= 0) //otherwise allocated on read
			ret = createOutputMatrixBlock(rlen, clen, (int) rlen, estnnz, true, false);

		return ret;
	}

	@SuppressWarnings("unchecked")
	private MatrixBlock readMatrixFromHDFS(Path path, JobConf job, FileSystem fs,
		MatrixBlock dest, long rlen, long clen, int blen) throws IOException, DMLRuntimeException {
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
			dest = computeSize(files, fs, rlen, clen);
			rlen = dest.getNumRows();
			//clen = dest.getNumColumns();
		}

		//actual read of individual files
		long lnnz = 0;
		MutableInt row = new MutableInt(0);
		for(int fileNo = 0; fileNo < files.size(); fileNo++) {
			lnnz += readMatrixFromInputStream(fs.open(files.get(fileNo)), path.toString(), dest, row, rlen, clen, blen);
		}

		//post processing
		dest.setNonZeros(lnnz);

		return dest;
	}

	protected abstract long readMatrixFromInputStream(InputStream is, String srcInfo, MatrixBlock dest,
		MutableInt rowPos, long rlen, long clen, int blen) throws IOException;

	public static class MatrixReaderRowRegularColRegular extends MatrixGenerateReader {

		public MatrixReaderRowRegularColRegular(CustomProperties _props) {
			super(_props);
		}

		@Override
		protected long readMatrixFromInputStream(InputStream is, String srcInfo, MatrixBlock dest,
			MutableInt rowPos, long rlen, long clen, int blen) throws IOException {

			String value = null;
			int row = rowPos.intValue();
			double cellValue = 0;
			int col = 0;
			long lnnz = 0;
			fastStringTokenizerDelim.setNaStrings(_props.getNaStrings());

			BufferedReader br = new BufferedReader(new InputStreamReader(is));

			//TODO: separate implementation for Sparse and Dens Matrix Blocks

			// Read the data
			try {
				while((value = br.readLine()) != null) //foreach line
				{
					fastStringTokenizerDelim.reset(value);
					while(col != -1) {
						cellValue = fastStringTokenizerDelim.nextDouble();
						col = fastStringTokenizerDelim.getIndex();
						if(cellValue != 0) {
							dest.appendValue(row, col, cellValue);
							lnnz++;
						}
					}
					row++;
					col = 0;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(br);
			}

			rowPos.setValue(row);
			return lnnz;
		}
	}

	public static class MatrixReaderRowRegularColIrregular extends MatrixGenerateReader {

		public MatrixReaderRowRegularColIrregular(CustomProperties _props) {
			super(_props);
		}

		@Override
		protected long readMatrixFromInputStream(InputStream is, String srcInfo, MatrixBlock dest,
			MutableInt rowPos, long rlen, long clen, int blen) throws IOException {

			String value = null;
			int row = rowPos.intValue();
			double cellValue = 0;
			int col = 0;
			long lnnz = 0;

			final FastStringTokenizer fastStringTokenizerIndexDelim = new FastStringTokenizer(_props.getIndexDelim());
			BufferedReader br = new BufferedReader(new InputStreamReader(is));

			//TODO: separate implementation for Sparse and Dens Matrix Blocks

			// Read the data
			try {
				while((value = br.readLine()) != null) //foreach line
				{
					fastStringTokenizerDelim.reset(value);
					String cellValueString = fastStringTokenizerDelim.nextToken();
					cellValue = UtilFunctions.parseToDouble(cellValueString, null);
					dest.appendValue(row, (int) clen - _props.getFirstColIndex() - 1, cellValue);

					while(col != -1) {
						String nt = fastStringTokenizerDelim.nextToken();
						if(fastStringTokenizerDelim.getIndex() == -1)
							break;
						fastStringTokenizerIndexDelim.reset(nt);
						col = fastStringTokenizerIndexDelim.nextInt();
						cellValue = fastStringTokenizerIndexDelim.nextDouble();
						if(cellValue != 0) {
							dest.appendValue(row, col - _props.getFirstColIndex(), cellValue);
							lnnz++;
						}
					}
					row++;
					col = 0;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(br);
			}

			rowPos.setValue(row);
			return lnnz;
		}
	}

	public static class MatrixReaderRowIrregular extends MatrixGenerateReader {

		public MatrixReaderRowIrregular(CustomProperties _props) {
			super(_props);
		}

		@Override protected long readMatrixFromInputStream(InputStream is, String srcInfo, MatrixBlock dest,
			MutableInt rowPos, long rlen, long clen, int blen) throws IOException {
			String value = null;
			int row = rowPos.intValue();
			double cellValue = 0;
			int col = 0;
			long lnnz = 0;

			BufferedReader br = new BufferedReader(new InputStreamReader(is));

			//TODO: separate implementation for Sparse and Dens Matrix Blocks

			// Read the data
			try {
				while((value = br.readLine()) != null) //foreach line
				{
					fastStringTokenizerDelim.reset(value);
					int ri = fastStringTokenizerDelim.nextInt();
					col = fastStringTokenizerDelim.nextInt();
					cellValue = fastStringTokenizerDelim.nextDouble();

					if(cellValue != 0) {
						dest.appendValue(ri - _props.getFirstColIndex(), col - _props.getFirstColIndex(), cellValue);
						lnnz++;
					}
					row = Math.max(row, ri);
				}
			}
			finally {
				IOUtilFunctions.closeSilently(br);
			}
			rowPos.setValue(row);
			return lnnz;
		}
	}

	public static class MatrixReaderJSON extends MatrixGenerateReader {

		public MatrixReaderJSON(CustomProperties _props) {
			super(_props);
		}

		@Override
		protected long readMatrixFromInputStream(InputStream is, String srcInfo, MatrixBlock dest,
			MutableInt rowPos, long rlen, long clen, int blen) throws IOException {
			String value;
			int row = rowPos.intValue();
			double cellValue;
			long lnnz = 0;

			BufferedReader br = new BufferedReader(new InputStreamReader(is));
			String[] colKeys = _props.getColKeys();
			// Read the data
			try {
				while((value = br.readLine()) != null) {
					FastJSONIndex fastJSONIndex = new FastJSONIndex(value);
					for(int c = 0; c < clen; c++) {
						cellValue = fastJSONIndex.getDoubleValue(colKeys[c]);
						if(cellValue != 0) {
							dest.appendValue(row, c, cellValue);
							lnnz++;
						}
					}
					row++;
				}
			}
			catch(Exception e) {
				throw new RuntimeException(e);
			}
			finally {
				IOUtilFunctions.closeSilently(br);
			}
			rowPos.setValue(row);
			return lnnz;
		}
	}
}

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
import java.util.HashSet;
import java.io.Serializable;

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

	private int getNumRows(List<Path> files, FileSystem fs) throws IOException, DMLRuntimeException {
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

	private
	MatrixBlock readMatrixFromHDFS(Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen,
		long clen, int blen) throws IOException, DMLRuntimeException {
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

	protected static class FastStringTokenizer implements Serializable {
		private static final long serialVersionUID = -4698672725609750097L;
		private String _string = null;
		private String _del = "";
		private int _pos = -1;
		private int _index = 0;
		private HashSet<String> naStrings = null;

		public FastStringTokenizer(String delimiter) {
			_del = delimiter;
			reset(null);
		}

		public void reset(String string) {
			_string = string;
			_pos = 0;
			_index = 0;
		}

		public String nextToken() {
			int len = _string.length();
			int start = _pos;

			if(_pos == -1) {
				_index = -1;
				return "0";
			}
			//find start (skip over leading delimiters)
			while(start < len && _del.equals(_string.substring(start, start + _del.length()))) {
				start += _del.length();
				_index++;
			}

			//find end (next delimiter) and return
			if(start < len) {
				_pos = _string.indexOf(_del, start);
				if(start < _pos && _pos < len)
					return _string.substring(start, _pos);
				else
					return _string.substring(start);
			}
			//no next token
			_index = -1;
			return null;
		}

		public int nextInt() {
			return Integer.parseInt(nextToken());
		}

		public long nextLong() {
			return Long.parseLong(nextToken());
		}

		public double nextDouble() {
			String nt = nextToken();
			if((naStrings != null && naStrings.contains(nt)) || nt == null)
				return 0;
			else
				return Double.parseDouble(nt);
		}

		public int getIndex() {
			return _index;
		}

		public void setNaStrings(HashSet<String> naStrings) {
			this.naStrings = naStrings;
		}
	}

	public static class MatrixReaderRowRegularColRegular extends MatrixGenerateReader {

		public MatrixReaderRowRegularColRegular(CustomProperties _props) {
			super(_props);
		}

		@Override
		protected long readMatrixFromInputStream(InputStream is, String srcInfo, MatrixBlock dest,
			MutableInt rowPos, long rlen, long clen, int blen) throws IOException {

			boolean sparse = dest.isInSparseFormat();
			String value = null;
			int row = rowPos.intValue();
			double cellValue = 0;
			int col = 0;
			long lnnz = 0;
			fastStringTokenizerDelim.naStrings = _props.getNaStrings();

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

			boolean sparse = dest.isInSparseFormat();
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
					dest.appendValue(row, (int) clen, cellValue);

					while(col != -1) {
						String nt = fastStringTokenizerDelim.nextToken();
						if(fastStringTokenizerDelim.getIndex() == -1)
							break;
						fastStringTokenizerIndexDelim.reset(nt);
						col = fastStringTokenizerIndexDelim.nextInt();
						cellValue = fastStringTokenizerIndexDelim.nextDouble();
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

	public static class MatrixReaderRowIrregular extends MatrixGenerateReader {

		public MatrixReaderRowIrregular(CustomProperties _props) {
			super(_props);
		}

		@Override
		protected long readMatrixFromInputStream(InputStream is, String srcInfo, MatrixBlock dest,
			MutableInt rowPos, long rlen, long clen, int blen) throws IOException {
			boolean sparse = dest.isInSparseFormat();
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
						dest.appendValue(ri, col, cellValue);
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
}



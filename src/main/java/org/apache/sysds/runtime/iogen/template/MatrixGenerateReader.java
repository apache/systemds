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

package org.apache.sysds.runtime.iogen.template;

import org.apache.commons.lang.mutable.MutableInt;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class MatrixGenerateReader extends MatrixReader {

	protected static CustomProperties _props;

	public MatrixGenerateReader(CustomProperties _props) {
		MatrixGenerateReader._props = _props;
	}

	protected MatrixBlock computeSize(List<Path> files, FileSystem fs, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		// allocate target matrix block based on given size;
		return new MatrixBlock(getNumRows(files, fs), (int) clen, rlen * clen);
	}

	private static int getNumRows(List<Path> files, FileSystem fs) throws IOException, DMLRuntimeException {
		int rows = 0;
		for(int fileNo = 0; fileNo < files.size(); fileNo++) {
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));
			try {
				// Row Identify
				if(_props.getRowIndex().equals(CustomProperties.IndexProperties.IDENTIFY)) {
					while(br.readLine() != null)
						rows++;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(br);
			}
		}
		return rows;
	}

	@Override public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)
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

	@Override public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException {

		MatrixBlock ret = null;
		if(rlen >= 0 && clen >= 0) //otherwise allocated on read
			ret = createOutputMatrixBlock(rlen, clen, (int) rlen, estnnz, true, false);

		return ret;
	}

	@SuppressWarnings("unchecked") private MatrixBlock readMatrixFromHDFS(Path path, JobConf job, FileSystem fs,
		MatrixBlock dest, long rlen, long clen, int blen) throws IOException, DMLRuntimeException {

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
			dest = computeSize(files, fs, rlen, clen);
			rlen = dest.getNumRows();
			//clen = _props.getColumnIdentifyProperties().length;
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

	protected void saveCode(String fileName, String code) {
		try(Writer writer = new BufferedWriter(
			new OutputStreamWriter(new FileOutputStream(fileName, false), "utf-8"))) {
			writer.write(code);
		}
		catch(Exception ex) {
		}
	}

//	protected double getCellValue(String str, ArrayList<String> keys, HashSet<String> endDelim) {
//		int currPos = 0;
//		for(String k : keys) {
//			int index = str.indexOf(k, currPos);
//			if(index != -1)
//				currPos = index + k.length();
//			else
//				return 0;
//		}
//		int endPos = -1;
//		for(String d : endDelim) {
//			endPos = d.length()> 0 ? str.indexOf(d, currPos): str.length();
//			if(endPos != -1)
//				break;
//		}
//		//------------------------
//		if(endDelim.contains("")){}
//		return UtilFunctions.getDouble(str.substring(currPos, endPos));
//	}
}

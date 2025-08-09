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

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.HDFSTool;

public class WriterTextLIBSVM extends MatrixWriter {

	protected static FileFormatPropertiesLIBSVM _props = null;

	public WriterTextLIBSVM(FileFormatPropertiesLIBSVM _props) {
		WriterTextLIBSVM._props = _props;
	}

	@Override
	public final void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int blen, long nnz, boolean diag)
		throws IOException, DMLRuntimeException
	{
		//validity check matrix dimensions
		if( src.getNumRows() != rlen || src.getNumColumns() != clen )
			throw new IOException("Matrix dimensions mismatch with metadata: "+src.getNumRows()+"x"+src.getNumColumns()+" vs "+rlen+"x"+clen+".");
		if( rlen == 0 || clen == 0 )
			throw new IOException("Write of matrices with zero rows or columns not supported ("+rlen+"x"+clen+").");

		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		//if the file already exists on HDFS, remove it.
		HDFSTool.deleteFileIfExistOnHDFS( fname );

		//core write (sequential/parallel)
		writeLIBSVMMatrixToHDFS(path, job, fs, src);

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	@Override
	public final void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen)
		throws IOException, DMLRuntimeException
	{

	}

	protected void writeLIBSVMMatrixToHDFS(Path path, JobConf job, FileSystem fs, MatrixBlock src)
		throws IOException
	{
		//sequential write libsvm file
		writeLIBSVMMatrixToFile(path, job, fs, src, 0, src.getNumRows());
	}

	protected static void writeLIBSVMMatrixToFile( Path path, JobConf job, FileSystem fs, MatrixBlock src, int rl, int rlen )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		int clen = src.getNumColumns();

		//create buffered writer
		BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));

		try
		{
			StringBuilder sb = new StringBuilder();
			_props = _props == null ? new FileFormatPropertiesLIBSVM() : _props;

			// Write data lines
			if( sparse ) //SPARSE
			{
				SparseBlock sblock = src.getSparseBlock();
				for(int i=rl; i < rlen; i++) {
					// append the class label as the 1st column
					double label = (sblock!=null) ?
						sblock.get(i, clen-1) : 0;
					sb.append(label);

					if( sblock!=null && i<sblock.numRows() && !sblock.isEmpty(i) ) {
						int pos = sblock.pos(i);
						int alen = sblock.size(i);
						int[] aix = sblock.indexes(i);
						double[] avals = sblock.values(i);
						// append sparse row
						for( int k=pos; k<pos+alen; k++ ) {
							if( aix[k]!=clen-1 ) {
								sb.append(_props.getDelim());
								appendIndexValLibsvm(sb, aix[k], avals[k]);
							}
						}
					}
					// write the string row
					sb.append('\n');
					br.write( sb.toString() );
					sb.setLength(0);
				}
			}
			else //DENSE
			{
				DenseBlock d = src.getDenseBlock();
				for( int i=rl; i<rlen; i++ ) {
					// append the class label as the 1st column
					double label = d!=null ? d.get(i, clen-1) : 0;
					sb.append(label);

					// append dense row
					for( int j=0; j<clen-1; j++ ) {
						double val = d!=null ? d.get(i, j) : 0;
						if( val != 0 ) {
							sb.append(_props.getDelim());
							appendIndexValLibsvm(sb, j, val);
						}
					}
					// write the string row
					sb.append('\n');
					br.write( sb.toString() );
					sb.setLength(0);
				}
			}
		}
		finally {
			IOUtilFunctions.closeSilently(br);
		}
	}

	// Return string in libsvm format (<index#>:<value#>)
	protected static void appendIndexValLibsvm(StringBuilder sb, int index, double value) {
		sb.append(index+1);  // convert 0 based matrix index to 1 base libsvm index
		sb.append(_props.getIndexDelim());
		sb.append(value);
	}

	@Override
	public long writeMatrixFromStream(String fname, LocalTaskQueue<IndexedMatrixValue> stream, long rlen, long clen, int blen) {
		throw new UnsupportedOperationException("Writing from an OOC stream is not supported for the LIBSVM format.");
	};
}

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
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.HDFSTool;

public class WriterMatrixMarket extends MatrixWriter
{
	@Override
	public final void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int blen, long nnz, boolean diag) 
		throws IOException, DMLRuntimeException 
	{
		//validity check matrix dimensions
		if( src.getNumRows() != rlen || src.getNumColumns() != clen ) {
			throw new IOException("Matrix dimensions mismatch with metadata: "+src.getNumRows()+"x"+src.getNumColumns()+" vs "+rlen+"x"+clen+".");
		}
				
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		//if the file already exists on HDFS, remove it.
		HDFSTool.deleteFileIfExistOnHDFS( fname );
			
		//core write
		writeMatrixMarketMatrixToHDFS(path, job, fs, src);

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	@Override
	public final void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen) 
		throws IOException, DMLRuntimeException 
	{
		Path path = new Path( fname );
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		
		FSDataOutputStream writer = null;
		try {
			writer = fs.create(path);
			writer.writeBytes(IOUtilFunctions.EMPTY_TEXT_LINE);
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	protected void writeMatrixMarketMatrixToHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock src )
		throws IOException
	{
		//sequential write
		writeMatrixMarketMatrixToFile(path, job, fs, src, 0, src.getNumRows());
	}

	protected static void writeMatrixMarketMatrixToFile( Path path, JobConf job, FileSystem fs, MatrixBlock src, int rl, int ru )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		int rlen = src.getNumRows();
		int clen = src.getNumColumns();
		long nnz = src.getNonZeros();
		
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		

		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			if( rl == 0 ) {
				// First output MM header
				sb.append ("%%MatrixMarket matrix coordinate real general\n");
			
				// output number of rows, number of columns and number of nnz
				sb.append (rlen + " " + clen + " " + nnz + "\n");
				br.write( sb.toString());
				sb.setLength(0);
			}
			 
			// output matrix cell
			if( sparse ) //SPARSE
			{			   
				Iterator<IJV> iter = src.getSparseBlockIterator(rl, ru);
				while( iter.hasNext() )
				{
					IJV cell = iter.next();

					sb.append(cell.getI()+1);
					sb.append(' ');
					sb.append(cell.getJ()+1);
					sb.append(' ');
					sb.append(cell.getV());
					sb.append('\n');
					br.write( sb.toString() ); //same as append
					sb.setLength(0); 
				}
			}
			else //DENSE
			{
				if( !src.isEmpty() ) {
					DenseBlock d = src.getDenseBlock();
					for( int i=rl; i<ru; i++ ) {
						String rowIndex = Integer.toString(i+1);
						for( int j=0; j<clen; j++ ) {
							double lvalue = d.get(i, j);
							if( lvalue != 0 ) //for nnz
							{
								sb.append(rowIndex);
								sb.append(' ');
								sb.append(j+1);
								sb.append(' ');
								sb.append(lvalue);
								sb.append('\n');
								br.write( sb.toString() ); //same as append
								sb.setLength(0); 
							}
						}
					}
				}
			}
	
			//handle empty result
			if ( src.isEmptyBlock(false) && rl==0 )
				br.write(IOUtilFunctions.EMPTY_TEXT_LINE);
		}
		finally {
			IOUtilFunctions.closeSilently(br);
		}
	}

	public static void mergeTextcellToMatrixMarket( String srcFileName, String fileName, long rlen, long clen, long nnz )
		throws IOException
	{
		Configuration conf = new Configuration(ConfigurationManager.getCachedJobConf());
		
		Path src = new Path (srcFileName);
		Path merge = new Path (fileName);
		FileSystem fs = IOUtilFunctions.getFileSystem(src, conf);

		if (fs.exists (merge)) {
			fs.delete(merge, true);
		}

		OutputStream out = fs.create(merge, true);

		// write out the header first 
		StringBuilder  sb = new StringBuilder();
		sb.append ("%%MatrixMarket matrix coordinate real general\n");

		// output number of rows, number of columns and number of nnz
		sb.append (rlen + " " + clen + " " + nnz + "\n");
		out.write (sb.toString().getBytes());

		// if the source is a directory
		if (fs.getFileStatus(src).isDirectory()) {
			try {
				FileStatus[] contents = fs.listStatus(src);
				for (int i = 0; i < contents.length; i++) {
					if (!contents[i].isDirectory()) {
						InputStream in = fs.open (contents[i].getPath());
						try {
							IOUtils.copyBytes (in, out, conf, false);
						}  finally {
							IOUtilFunctions.closeSilently(in);
						}
					}
				}
			} finally {
			 IOUtilFunctions.closeSilently(out);
			}
		} else if (fs.getFileStatus(src).isDirectory()) {
			InputStream in = null;
			try {
				in = fs.open (src);
				IOUtils.copyBytes (in, out, conf, true);
			}
			finally {
				IOUtilFunctions.closeSilently(in);
				IOUtilFunctions.closeSilently(out);
			}
		} else {
			throw new IOException(src.toString() + ": No such file or directory");
		}
	}

	@Override
	public void writeMatrixFromStream(String fname, LocalTaskQueue<IndexedMatrixValue> stream, long rlen, long clen, int blen) {
		throw new UnsupportedOperationException("Writing from an OOC stream is not supported for the HDF5 format.");
	};
}

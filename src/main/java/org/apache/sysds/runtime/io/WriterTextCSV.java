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
import java.util.ArrayList;
import java.util.Collections;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.HDFSTool;

public class WriterTextCSV extends MatrixWriter
{
	//blocksize for string concatenation in order to prevent write OOM 
	//(can be set to very large value to disable blocking)
	public static final int BLOCKSIZE_J = 32; //32 cells (typically ~512B, should be less than write buffer of 1KB)
	
	protected FileFormatPropertiesCSV _props = null;
	
	public WriterTextCSV( FileFormatPropertiesCSV props ) {
		_props = props;
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
		writeCSVMatrixToHDFS(path, job, fs, src, _props);

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	@Override
	public final void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen) 
		throws IOException, DMLRuntimeException 
	{
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		MatrixBlock src = new MatrixBlock((int)rlen, 1, true);
		writeCSVMatrixToHDFS(path, job, fs, src, _props);

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	protected void writeCSVMatrixToHDFS(Path path, JobConf job, FileSystem fs, MatrixBlock src, FileFormatPropertiesCSV csvprops) 
		throws IOException 
	{
		//sequential write csv file
		writeCSVMatrixToFile(path, job, fs, src, 0, src.getNumRows(), csvprops);
	}

	protected static void writeCSVMatrixToFile( Path path, JobConf job, FileSystem fs, MatrixBlock src, int rl, int ru, FileFormatPropertiesCSV props )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		int clen = src.getNumColumns();
		
		//create buffered writer
		BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
		
		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			props = (props==null)? new FileFormatPropertiesCSV() : props;
			String delim = props.getDelim();
			boolean csvsparse = props.isSparse();
			
			// Write header line, if needed
			if( props.hasHeader() && rl==0 ) 
			{
				//write row chunk-wise to prevent OOM on large number of columns
				for( int bj=0; bj<clen; bj+=BLOCKSIZE_J )
				{
					for( int j=bj; j < Math.min(clen,bj+BLOCKSIZE_J); j++) 
					{
						sb.append("C"+ (j+1));
						if ( j < clen-1 )
							sb.append(delim);
					}
					br.write( sb.toString() );
		            sb.setLength(0);	
				}
				sb.append('\n');
				br.write( sb.toString() );
	            sb.setLength(0);
			}
			
			// Write data lines
			if( sparse ) //SPARSE
			{	
				SparseBlock sblock = src.getSparseBlock();
				for(int i=rl; i < ru; i++) 
	            {
					//write row chunk-wise to prevent OOM on large number of columns
					int prev_jix = -1;
					if(    sblock!=null && i<sblock.numRows() 
						&& !sblock.isEmpty(i) )
					{
						int pos = sblock.pos(i);
						int alen = sblock.size(i);
						int[] aix = sblock.indexes(i);
						double[] avals = sblock.values(i);
						
						for(int j=pos; j<pos+alen; j++) 
						{
							int jix = aix[j];
							
							// output empty fields, if needed
							for( int j2=prev_jix; j2<jix-1; j2++ ) {
								if( !csvsparse )
									sb.append('0');
								sb.append(delim);
							
								//flush buffered string
					            if( j2%BLOCKSIZE_J==0 ){
									br.write( sb.toString() );
						            sb.setLength(0);
					            }
							}
							
							// output the value (non-zero)
							sb.append( avals[j] );
							if( jix < clen-1)
								sb.append(delim);
							br.write( sb.toString() );
				            sb.setLength(0);
				            
				            //flush buffered string
				            if( jix%BLOCKSIZE_J==0 ){
								br.write( sb.toString() );
					            sb.setLength(0);
				            }
				            
							prev_jix = jix;
						}
					}
					
					// Output empty fields at the end of the row.
					// In case of an empty row, output (clen-1) empty fields
					for( int bj=prev_jix+1; bj<clen; bj+=BLOCKSIZE_J )
					{
						for( int j = bj; j < Math.min(clen,bj+BLOCKSIZE_J); j++) {
							if( !csvsparse )
								sb.append('0');
							if( j < clen-1 )
								sb.append(delim);
						}
						br.write( sb.toString() );
			            sb.setLength(0);	
					}

					sb.append('\n');
					br.write( sb.toString() ); 
					sb.setLength(0); 
				}
			}
			else //DENSE
			{
				DenseBlock d = src.getDenseBlock();
				for( int i=rl; i<ru; i++ ) 
				{
					//write row chunk-wise to prevent OOM on large number of columns
					for( int bj=0; bj<clen; bj+=BLOCKSIZE_J )
					{
						for( int j=bj; j<Math.min(clen,bj+BLOCKSIZE_J); j++ )
						{
							double lvalue = d!=null ? d.get(i, j) : 0;
							if( lvalue != 0 ) //for nnz
								sb.append(lvalue);
							else if( !csvsparse ) 
								sb.append('0');
							
							if( j != clen-1 )
								sb.append(delim);
						}
						br.write( sb.toString() );
						sb.setLength(0);
					}
					
					sb.append('\n');
					br.write( sb.toString() ); //same as append
					sb.setLength(0); 
				}
			}
		}
		finally {
			IOUtilFunctions.closeSilently(br);
		}
	}

	public final void addHeaderToCSV(String srcFileName, String destFileName, long rlen, long clen) 
		throws IOException 
	{
		Configuration conf = new Configuration(ConfigurationManager.getCachedJobConf());

		Path srcFilePath = new Path(srcFileName);
		Path destFilePath = new Path(destFileName);
		FileSystem fs = IOUtilFunctions.getFileSystem(srcFilePath, conf);
		
		if ( !_props.hasHeader() ) {
			// simply move srcFile to destFile
			
			/*
			 * TODO: Remove this roundabout way! 
			 * For example: destFilePath = /user/biadmin/csv/temp/out/file.csv 
			 *              & the only path that exists already on HDFS is /user/biadmin/csv/.
			 * In this case: the directory structure /user/biadmin/csv/temp/out must be created. 
			 * Simple hdfs.rename() does not seem to create this directory structure.
			 */
			
			// delete the destination file, if exists already
			fs.delete(destFilePath, true);
			
			// Create /user/biadmin/csv/temp/out/file.csv so that ..../temp/out/ is created.
			fs.createNewFile(destFilePath);
			
			// delete the file "file.csv" but preserve the directory structure /user/biadmin/csv/temp/out/
			fs.delete(destFilePath, true);
			
			// finally, move the data to destFilePath = /user/biadmin/csv/temp/out/file.csv
			fs.rename(srcFilePath, destFilePath);

			return;
		}
	
		// construct the header line
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < clen; i++) {
			sb.append("C" + (i + 1));
			if (i < clen - 1)
				sb.append(_props.getDelim());
		}
		sb.append('\n');

		if (fs.getFileStatus(srcFilePath).isDirectory()) {

			// compute sorted order among part files
			ArrayList<Path> files=new ArrayList<>();
			for(FileStatus stat: fs.listStatus(srcFilePath, IOUtilFunctions.hiddenFileFilter))
				files.add(stat.getPath());
			Collections.sort(files);
		
			// first part file path
			Path firstpart = files.get(0);
			
			// create a temp file, and add header and contents of first part
			Path tmp = new Path(firstpart.toString() + ".tmp");
			OutputStream out = fs.create(tmp, true);
			out.write(sb.toString().getBytes());
			sb.setLength(0);
			
			// copy rest of the data from firstpart
			InputStream in = null;
			try {
				in = fs.open(firstpart);
				IOUtils.copyBytes(in, out, conf, true);
			} finally {
				IOUtilFunctions.closeSilently(in);
				IOUtilFunctions.closeSilently(out);
			}
			
			// rename tmp to firstpart
			fs.delete(firstpart, true);
			fs.rename(tmp, firstpart);
			
			// rename srcfile to destFile
			fs.delete(destFilePath, true);
			fs.createNewFile(destFilePath); // force the creation of directory structure
			fs.delete(destFilePath, true);  // delete the file, but preserve the directory structure
			fs.rename(srcFilePath, destFilePath); // move the data 
		
		} else if (fs.getFileStatus(srcFilePath).isFile()) {
			// create destination file
			OutputStream out = fs.create(destFilePath, true);
			
			// write header
			out.write(sb.toString().getBytes());
			sb.setLength(0);
			
			// copy the data from srcFile
			InputStream in = null;
			try {
				in = fs.open(srcFilePath);
				IOUtils.copyBytes(in, out, conf, true);
			} 
			finally {
				IOUtilFunctions.closeSilently(in);
				IOUtilFunctions.closeSilently(out);
			}
		} else {
			throw new IOException(srcFilePath.toString() + ": No such file or directory");
		}
	}

	@Override
	public long writeMatrixFromStream(String fname, LocalTaskQueue<IndexedMatrixValue> stream, long rlen, long clen, int blen) {
		throw new UnsupportedOperationException("Writing from an OOC stream is not supported for the TextCSV format.");
	};
}

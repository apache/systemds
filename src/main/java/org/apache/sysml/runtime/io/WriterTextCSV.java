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

package org.apache.sysml.runtime.io;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.mapred.JobConf;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.matrix.CSVReblockMR;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseRow;
import org.apache.sysml.runtime.util.MapReduceTool;

/**
 * 
 */
public class WriterTextCSV extends MatrixWriter
{
	//blocksize for string concatenation in order to prevent write OOM 
	//(can be set to very large value to disable blocking)
	public static final int BLOCKSIZE_J = 32; //32 cells (typically ~512B, should be less than write buffer of 1KB)
	
	protected CSVFileFormatProperties _props = null;
	
	public WriterTextCSV( CSVFileFormatProperties props ) {
		_props = props;
	}
	
	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int brlen, int bclen, long nnz) 
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//validity check matrix dimensions
		if( src.getNumRows() != rlen || src.getNumColumns() != clen ) {
			throw new IOException("Matrix dimensions mismatch with metadata: "+src.getNumRows()+"x"+src.getNumColumns()+" vs "+rlen+"x"+clen+".");
		}
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );
			
		//core write
		writeCSVMatrixToHDFS(path, job, src, rlen, clen, nnz, _props);
	}

	@Override
	public void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int brlen, int bclen) 
		throws IOException, DMLRuntimeException 
	{
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );

		MatrixBlock src = new MatrixBlock((int)rlen, 1, true);
		writeCSVMatrixToHDFS(path, job, src, brlen, clen, 0, _props);
	}
	
	/**
	 * 
	 * @param fileName
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @throws IOException
	 */
	protected void writeCSVMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, long nnz, CSVFileFormatProperties props )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		FileSystem fs = FileSystem.get(job);
        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
		
		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			props = (props==null)? new CSVFileFormatProperties() : props;
			String delim = props.getDelim();
			boolean csvsparse = props.isSparse();
			
			// Write header line, if needed
			if( props.hasHeader() ) 
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
				SparseRow[] sparseRows = src.getSparseBlock();
				for(int i=0; i < rlen; i++) 
	            {
					//write row chunk-wise to prevent OOM on large number of columns
					int prev_jix = -1;
					if(    sparseRows!=null && i<sparseRows.length 
						&& sparseRows[i]!=null && !sparseRows[i].isEmpty() )
					{
						SparseRow arow = sparseRows[i];
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();
						
						for(int j=0; j < alen; j++) 
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
				for( int i=0; i<rlen; i++ ) 
				{
					//write row chunk-wise to prevent OOM on large number of columns
					for( int bj=0; bj<clen; bj+=BLOCKSIZE_J )
					{
						for( int j=bj; j<Math.min(clen,bj+BLOCKSIZE_J); j++ )
						{
							double lvalue = src.getValueDenseUnsafe(i, j);
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
		finally
		{
			IOUtilFunctions.closeSilently(br);
		}
	}


	
	/**
	 * Method to merge multiple CSV part files on HDFS into a single CSV file on HDFS. 
	 * The part files are created by CSV_WRITE MR job. 
	 * 
	 * This method is invoked from CP-write instruction.
	 * 
	 * @param srcFileName
	 * @param destFileName
	 * @param csvprop
	 * @param rlen
	 * @param clen
	 * @throws IOException
	 */
	public void mergeCSVPartFiles(String srcFileName, String destFileName, CSVFileFormatProperties csvprop, long rlen, long clen) 
		throws IOException 
	{	
		Configuration conf = new Configuration(ConfigurationManager.getCachedJobConf());

		Path srcFilePath = new Path(srcFileName);
		Path mergedFilePath = new Path(destFileName);
		FileSystem hdfs = FileSystem.get(conf);

		if (hdfs.exists(mergedFilePath)) {
			hdfs.delete(mergedFilePath, true);
		}
		OutputStream out = hdfs.create(mergedFilePath, true);

		// write out the header, if needed
		if (csvprop.hasHeader()) {
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < clen; i++) {
				sb.append("C" + (i + 1));
				if (i < clen - 1)
					sb.append(csvprop.getDelim());
			}
			sb.append('\n');
			out.write(sb.toString().getBytes());
			sb.setLength(0);
		}

		// if the source is a directory
		if (hdfs.isDirectory(srcFilePath)) {
			try {
				FileStatus[] contents = hdfs.listStatus(srcFilePath);
				Path[] partPaths = new Path[contents.length];
				int numPartFiles = 0;
				for (int i = 0; i < contents.length; i++) {
					if (!contents[i].isDirectory()) {
						partPaths[i] = contents[i].getPath();
						numPartFiles++;
					}
				}
				Arrays.sort(partPaths);

				for (int i = 0; i < numPartFiles; i++) {
					InputStream in = hdfs.open(partPaths[i]);
					try {
						IOUtils.copyBytes(in, out, conf, false);
						if(i<numPartFiles-1)
							out.write('\n');
					} 
					finally {
						IOUtilFunctions.closeSilently(in);
					}
				}
			} finally {
				IOUtilFunctions.closeSilently(out);
			}
		} else if (hdfs.isFile(srcFilePath)) {
			InputStream in = null;
			try {
				in = hdfs.open(srcFilePath);
				IOUtils.copyBytes(in, out, conf, true);
			} finally {
				IOUtilFunctions.closeSilently(in);
				IOUtilFunctions.closeSilently(out);
			}
		} else {
			throw new IOException(srcFilePath.toString()
					+ ": No such file or directory");
		}
	}
		
	/**
	 * 
	 * @param srcFileName
	 * @param destFileName
	 * @param csvprop
	 * @param rlen
	 * @param clen
	 * @throws IOException
	 */
	@SuppressWarnings("unchecked")
	public void addHeaderToCSV(String srcFileName, String destFileName, long rlen, long clen) 
		throws IOException 
	{
		Configuration conf = new Configuration(ConfigurationManager.getCachedJobConf());

		Path srcFilePath = new Path(srcFileName);
		Path destFilePath = new Path(destFileName);
		FileSystem hdfs = FileSystem.get(conf);
		
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
			//boolean ret1 = 
			hdfs.delete(destFilePath, true);
			
			// Create /user/biadmin/csv/temp/out/file.csv so that ..../temp/out/ is created.
			//boolean ret2 = 
			hdfs.createNewFile(destFilePath);
			
			// delete the file "file.csv" but preserve the directory structure /user/biadmin/csv/temp/out/
			//boolean ret3 = 
			hdfs.delete(destFilePath, true);
			
			// finally, move the data to destFilePath = /user/biadmin/csv/temp/out/file.csv
			//boolean ret4 = 
			hdfs.rename(srcFilePath, destFilePath);

			//System.out.println("Return values = del:" + ret1 + ", createNew:" + ret2 + ", del:" + ret3 + ", rename:" + ret4);
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

		if (hdfs.isDirectory(srcFilePath)) {

			// compute sorted order among part files
			ArrayList<Path> files=new ArrayList<Path>();
			for(FileStatus stat: hdfs.listStatus(srcFilePath, CSVReblockMR.hiddenFileFilter))
				files.add(stat.getPath());
			Collections.sort(files);
		
			// first part file path
			Path firstpart = files.get(0);
			
			// create a temp file, and add header and contents of first part
			Path tmp = new Path(firstpart.toString() + ".tmp");
			OutputStream out = hdfs.create(tmp, true);
			out.write(sb.toString().getBytes());
			sb.setLength(0);
			
			// copy rest of the data from firstpart
			InputStream in = null;
			try {
				in = hdfs.open(firstpart);
				IOUtils.copyBytes(in, out, conf, true);
			} finally {
				IOUtilFunctions.closeSilently(in);
				IOUtilFunctions.closeSilently(out);
			}
			
			// rename tmp to firstpart
			hdfs.delete(firstpart, true);
			hdfs.rename(tmp, firstpart);
			
			// rename srcfile to destFile
			hdfs.delete(destFilePath, true);
			hdfs.createNewFile(destFilePath); // force the creation of directory structure
			hdfs.delete(destFilePath, true);  // delete the file, but preserve the directory structure
			hdfs.rename(srcFilePath, destFilePath); // move the data 
		
		} else if (hdfs.isFile(srcFilePath)) {
			// create destination file
			OutputStream out = hdfs.create(destFilePath, true);
			
			// write header
			out.write(sb.toString().getBytes());
			sb.setLength(0);
			
			// copy the data from srcFile
			InputStream in = null;
			try {
				in = hdfs.open(srcFilePath);
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
}

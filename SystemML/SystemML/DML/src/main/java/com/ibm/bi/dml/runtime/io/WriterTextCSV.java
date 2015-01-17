/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

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

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.FileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

public class WriterTextCSV extends MatrixWriter
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private CSVFileFormatProperties _props = null;
	
	public WriterTextCSV( CSVFileFormatProperties props )
	{
		_props = props;
	}
	
	@Override
	public void writeMatrixFromHDFS(MatrixBlock src, String fname, long rlen, long clen, int brlen, int bclen, long nnz) 
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//prepare file access
		JobConf job = new JobConf();
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );
			
		//core write
		writeCSVMatrixToHDFS(path, job, src, rlen, clen, nnz, _props);
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
	public void writeCSVMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, long nnz, FileFormatProperties formatProperties )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		FileSystem fs = FileSystem.get(job);
        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
        
    	int rows = src.getNumRows();
		int cols = src.getNumColumns();

		//blocksize for string concatenation in order to prevent write OOM 
		//(can be set to very large value to disable blocking)
		final int blockSizeJ = 32; //32 cells (typically ~512B, should be less than write buffer of 1KB)
		
		//bound check per block
		if( rows > rlen || cols > clen ) {
			throw new IOException("Matrix block [1:"+rows+",1:"+cols+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		//file format property check
		if( formatProperties!=null &&  !(formatProperties instanceof CSVFileFormatProperties) ) {
			throw new IOException("Wrong type of file format properties for CSV writer.");
		}
		
		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			CSVFileFormatProperties csvProperties = (CSVFileFormatProperties)formatProperties;
			csvProperties = (csvProperties==null)? new CSVFileFormatProperties() : csvProperties;
			String delim = csvProperties.getDelim(); //Pattern.quote(csvProperties.getDelim());
			boolean csvsparse = csvProperties.isSparse();
			
			// Write header line, if needed
			if( csvProperties.hasHeader() ) 
			{
				//write row chunk-wise to prevent OOM on large number of columns
				for( int bj=0; bj<clen; bj+=blockSizeJ )
				{
					for( int j=bj; j < Math.min(clen,bj+blockSizeJ); j++) 
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
				SparseRow[] sparseRows = src.getSparseRows();
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
					            if( j2%blockSizeJ==0 ){
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
				            if( jix%blockSizeJ==0 ){
								br.write( sb.toString() );
					            sb.setLength(0);
				            }
				            
							prev_jix = jix;
						}
					}
					
					// Output empty fields at the end of the row.
					// In case of an empty row, output (clen-1) empty fields
					for( int bj=prev_jix+1; bj<clen; bj+=blockSizeJ )
					{
						for( int j = bj; j < Math.min(clen,bj+blockSizeJ); j++) {
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
					for( int bj=0; bj<clen; bj+=blockSizeJ )
					{
						for( int j=bj; j<Math.min(clen,bj+blockSizeJ); j++ )
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
			if( br != null )
				br.close();
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
	public void mergeCSVPartFiles(String srcFileName,
			String destFileName, CSVFileFormatProperties csvprop, long rlen, long clen) 
			throws IOException 
	{	
		Configuration conf = new Configuration();

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
					} finally {
						in.close();
					}
				}
			} finally {
				out.close();
			}
		} else if (hdfs.isFile(srcFilePath)) {
			InputStream in = null;
			try {
				in = hdfs.open(srcFilePath);
				IOUtils.copyBytes(in, out, conf, true);
			} finally {
				in.close();
				out.close();
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
	public void addHeaderToCSV(String srcFileName, String destFileName, CSVFileFormatProperties csvprop, long rlen, long clen) 
			throws IOException 
	{
		
		Configuration conf = new Configuration();

		Path srcFilePath = new Path(srcFileName);
		Path destFilePath = new Path(destFileName);
		FileSystem hdfs = FileSystem.get(conf);
		
		if ( !csvprop.hasHeader() ) {
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
				sb.append(csvprop.getDelim());
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
				in.close();
				out.close();
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
			} finally {
				in.close();
				out.close();
			}
		} else {
			throw new IOException(srcFilePath.toString()
					+ ": No such file or directory");
		}
	}
}

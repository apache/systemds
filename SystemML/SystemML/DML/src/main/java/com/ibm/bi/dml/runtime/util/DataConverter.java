/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.OutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;


import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.io.FileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.io.IJV;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.SparseRow;
import com.ibm.bi.dml.runtime.matrix.io.SparseRowsIterator;
import com.ibm.bi.dml.runtime.matrix.mapred.DistributedCacheInput;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;


/**
 * This class provides methods to read and write matrix blocks from to HDFS using different data formats.
 * Those functionalities are used especially for CP read/write and exporting in-memory matrices to HDFS
 * (before executing MR jobs).
 * 
 */
public class DataConverter 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	
	//////////////
	// READING and WRITING of matrix blocks to/from HDFS
	// (textcell, binarycell, binaryblock)
	///////
	
	private class ReadProperties {
		// Properties common to all file formats 
		String path;
		long rlen, clen;
		int brlen, bclen;
		double expectedSparsity;
		InputInfo inputInfo;
		boolean localFS;
		
		// Properties specific to CSV files
		FileFormatProperties formatProperties;
		
		public ReadProperties() {
			rlen = clen = -1;
			brlen = bclen = -1;
			expectedSparsity = 0.1d;
			inputInfo = null;
			localFS = false;
		}
	}
	
	/**
	 * 
	 * @param mat
	 * @param dir
	 * @param outputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 */
	public static void writeMatrixToHDFS(MatrixBlock mat, String dir, OutputInfo outputinfo, 
			                             MatrixCharacteristics mc )
		throws IOException
	{
		writeMatrixToHDFS(mat, dir, outputinfo, mc, -1, null);
	}
	
	/**
	 * 
	 * @param mat
	 * @param dir
	 * @param outputinfo
	 * @param mc
	 * @param replication
	 * @param formatProperties
	 * @throws IOException
	 */
	public static void writeMatrixToHDFS(MatrixBlock mat, String dir, OutputInfo outputinfo, MatrixCharacteristics mc, int replication, FileFormatProperties formatProperties)
			throws IOException
	{
		JobConf job = new JobConf();
		Path path = new Path(dir);

		//System.out.println("write matrix (sparse="+mat.isInSparseFormat()+") to HDFS: "+dir);
		
		try
		{
			// If the file already exists on HDFS, remove it.
			MapReduceTool.deleteFileIfExistOnHDFS(dir);

			// core matrix writing
			if ( outputinfo == OutputInfo.TextCellOutputInfo ) 
			{	
				writeTextCellMatrixToHDFS(path, job, mat, mc.get_rows(), mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block());
			}
			else if ( outputinfo == OutputInfo.BinaryCellOutputInfo ) 
			{
				writeBinaryCellMatrixToHDFS(path, job, mat, mc.get_rows(), mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block());
			}
			else if( outputinfo == OutputInfo.BinaryBlockOutputInfo )
			{
				writeBinaryBlockMatrixToHDFS(path, job, mat, mc.get_rows(), mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block(), replication);
			}
			else if ( outputinfo == OutputInfo.MatrixMarketOutputInfo ) 
			{
				writeMatrixMarketToHDFS( path, job, mat, mc.get_rows(), mc.get_cols(), mc.getNonZeros());
			}
			else if ( outputinfo == OutputInfo.CSVOutputInfo ) {
				writeCSVToHDFS(path, job, mat, mc.get_rows(), mc.get_cols(), mc.getNonZeros(), formatProperties);
			}
			else {
				throw new RuntimeException("Unknown format (" + OutputInfo.outputInfoToString(outputinfo));
			}
		}
		catch(Exception e)
		{
			throw new IOException(e);
		}
	}
	
	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen, boolean localFS) 
		throws IOException
	{	
		DataConverter dc = new DataConverter();
		ReadProperties prop = dc.new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.localFS = localFS;
		
		//expected matrix is sparse (default SystemML usecase)
		return readMatrixFromHDFS(prop);
	}
	
	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen) 
		throws IOException
	{	
		DataConverter dc = new DataConverter();
		ReadProperties prop = dc.new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		
		//expected matrix is sparse (default SystemML usecase)
		return readMatrixFromHDFS(prop);
	}

	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen, double expectedSparsity) 
		throws IOException
	{	
		DataConverter dc = new DataConverter();
		ReadProperties prop = dc.new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.expectedSparsity = expectedSparsity;
		
		return readMatrixFromHDFS(prop);
	}

	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @param localFS
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int brlen, int bclen, double expectedSparsity, boolean localFS) 
	throws IOException
	{
		DataConverter dc = new DataConverter();
		ReadProperties prop = dc.new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.expectedSparsity = expectedSparsity;
		prop.localFS = localFS;
		
		return readMatrixFromHDFS(prop);
	}
	
	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @param localFS
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int brlen, int bclen, double expectedSparsity, FileFormatProperties formatProperties) 
	throws IOException
	{
		DataConverter dc = new DataConverter();
		ReadProperties prop = dc.new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.expectedSparsity = expectedSparsity;
		prop.formatProperties = formatProperties;
		
		//prop.printMe();
		return readMatrixFromHDFS(prop);
	}
	
	/**
	 * Core method for reading matrices in format textcell, matrixmarket, binarycell, or binaryblock 
	 * from HDFS into main memory. For expected dense matrices we directly copy value- or block-at-a-time 
	 * into the target matrix. In contrast, for sparse matrices, we append (column-value)-pairs and do a 
	 * final sort if required in order to prevent large reorg overheads and increased memory consumption 
	 * in case of unordered inputs.  
	 * 
	 * DENSE MxN input:
	 *  * best/average/worst: O(M*N)
	 * SPARSE MxN input
	 *  * best (ordered, or binary block w/ clen<=bclen): O(M*N)
	 *  * average (unordered): O(M*N*log(N))
	 *  * worst (descending order per row): O(M * N^2)
	 * 
	 * NOTE: providing an exact estimate of 'expected sparsity' can prevent a full copy of the result
	 * matrix block (required for changing sparse->dense, or vice versa)
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(ReadProperties prop) 
		throws IOException
	{	
		//Timing time = new Timing(true);
		
		long rlen = prop.rlen;
		long clen = prop.clen;
		InputInfo inputinfo = prop.inputInfo;
		
		MatrixBlock ret = null;
		boolean sparse = false;
		
		if ( inputinfo == InputInfo.CSVInputInfo && (rlen==-1 || clen==-1) ) {
			// CP-side CSV reblock based on file size for matrix w/ unknown dimensions
			ret = null;
		}
		else {
			//determine target representation (sparse/dense)
			long estnnz = (long)(prop.expectedSparsity*rlen*clen);
			sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, estnnz); 
			
			//prepare result matrix block
			ret = new MatrixBlock((int)rlen, (int)clen, sparse, (int)estnnz);
			if( !sparse && inputinfo != InputInfo.BinaryBlockInputInfo )
				ret.allocateDenseBlockUnsafe((int)rlen, (int)clen);
			else if( sparse )
				ret.adjustSparseRows((int)rlen-1);
		}
		
		//prepare file access
		JobConf job = new JobConf();	
		FileSystem fs = (prop.localFS) ? FileSystem.getLocal(job) : FileSystem.get(job);
		Path path = new Path( ((prop.localFS) ? "file:///" : "") + prop.path); 
		if( !fs.exists(path) )	
			throw new IOException("File "+prop.path+" does not exist on HDFS/LFS.");
		
		try 
		{
			//check for empty file
			if( MapReduceTool.isFileEmpty( fs, path.toString() ) )
				throw new EOFException("Empty input file "+ prop.path +".");
			
			boolean isMMFile = (inputinfo == InputInfo.MatrixMarketInputInfo);
			
			//core matrix reading 
			if( inputinfo == InputInfo.TextCellInputInfo )
			{			
				if( fs.isDirectory(path) )
					readTextCellMatrixFromHDFS(path, job, ret, rlen, clen, prop.brlen, prop.bclen);
				else
					readRawTextCellMatrixFromHDFS(path, job, fs, ret, rlen, clen, prop.brlen, prop.bclen, isMMFile);
			}
			else if( inputinfo == InputInfo.BinaryCellInputInfo )
			{
				readBinaryCellMatrixFromHDFS( path, job, fs, ret, rlen, clen, prop.brlen, prop.bclen );
			}
			else if( inputinfo == InputInfo.BinaryBlockInputInfo )
			{
				readBinaryBlockMatrixFromHDFS( path, job, fs, ret, rlen, clen, prop.brlen, prop.bclen );
			}
			else if ( inputinfo == InputInfo.MatrixMarketInputInfo ) {
				readRawTextCellMatrixFromHDFS(path, job, fs, ret, rlen, clen, prop.brlen, prop.bclen, isMMFile);
			} 
			else if ( inputinfo == InputInfo.CSVInputInfo ) {
				CSVFileFormatProperties csvprop = (CSVFileFormatProperties) prop.formatProperties;
				ret = readCSVMatrixFromHDFS(path, job, fs, ret, rlen, clen, prop.brlen, prop.bclen, csvprop.isHasHeader(), csvprop.getDelim(), csvprop.isFill(), csvprop.getFillValue());
			}
			else {
				throw new IOException("Can not read files with format = " + InputInfo.inputInfoToString(inputinfo));
			}
			
			//finally check if change of sparse/dense block representation required
			if( !sparse || inputinfo==InputInfo.BinaryBlockInputInfo )
				ret.recomputeNonZeros();
			ret.examSparsity();	
		} 
		catch (Exception e) 
		{
			throw new IOException(e);
		}

		//System.out.println("read matrix ("+rlen+","+clen+","+ret.getNonZeros()+") in "+time.stop());
		
		return ret;
	}

	/**
	 * 	
	 * @param is
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromInputStream( InputStream is, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen, double expectedSparsity ) 
		throws IOException
	{
		//check valid input infos
		if( !(inputinfo == InputInfo.TextCellInputInfo || inputinfo== InputInfo.MatrixMarketInputInfo) )
			throw new IOException("Unsupported inputinfo for read from input stream: "+inputinfo);
		
		long estnnz = (long)(expectedSparsity*rlen*clen);
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, estnnz); 

		//prepare result matrix block
		MatrixBlock ret = new MatrixBlock((int)rlen, (int)clen, sparse, (int)estnnz);
		if( !sparse && inputinfo != InputInfo.BinaryBlockInputInfo )
			ret.allocateDenseBlockUnsafe((int)rlen, (int)clen);
		else if( sparse )
			ret.adjustSparseRows((int)rlen-1);

		try
		{
			//core read (consistent use of internal readers)
			readRawTextCellMatrixFromInputStream(is, ret, rlen, clen, brlen, bclen, (inputinfo==InputInfo.MatrixMarketInputInfo));
		
			//finally check if change of sparse/dense block representation required
			if( !sparse || inputinfo==InputInfo.BinaryBlockInputInfo )
				ret.recomputeNonZeros();
			ret.examSparsity();	
		}
		catch(Exception ex)
		{
			throw new IOException(ex);
		}
		
		return ret;
	}

	/**
	 * 
	 * @param dir
	 * @param inputInfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param localFS
	 * @return
	 * @throws IOException
	 */
	public static ArrayList<IndexedMatrixValue> readMatrixBlocksFromHDFS(String dir, InputInfo inputInfo, long rlen, long clen, int brlen, int bclen, boolean localFS) 
		throws IOException
	{	
		//Timing time = new Timing(true);
		
		ArrayList<IndexedMatrixValue> ret = new ArrayList<IndexedMatrixValue>();
		
		//check input format
		if( inputInfo != InputInfo.BinaryBlockInputInfo )
			throw new IOException("Read matrix blocks from hdfs is only supported for binary blocked format.");
		
		//prepare file access
		JobConf job = ConfigurationManager.getCachedJobConf();
		FileSystem fs = (localFS) ? FileSystem.getLocal(job) : FileSystem.get(job);
		Path path = new Path( ((localFS) ? "file:///" : "") + dir); 
		if( !fs.exists(path) )	
			throw new IOException("File "+path+" does not exist on HDFS/LFS.");
		
		try 
		{
			//check for empty file
			if( MapReduceTool.isFileEmpty( fs, path.toString() ) )
				throw new EOFException("Empty input file "+ path +".");
			
			readBinaryBlockMatrixBlocksFromHDFS( path, job, fs, ret, rlen, clen, brlen, bclen );
			
		} 
		catch (Exception e) 
		{
			throw new IOException(e);
		}

		//System.out.println("read matrix ("+rlen+","+clen+","+ret.getNonZeros()+") in "+time.stop());
		
		return ret;
	}

	
	/**
	 * Reads a partition that contains the given block index (rowBlockIndex, colBlockIndex)
	 * from a file on HDFS/LocalFS as specified by <code>dir</code>, which is partitioned using 
	 * <code>pformat</code> and <code>psize</code>.
	 * 
	 * @param dir
	 * @param localFS
	 * @param rows
	 * @param cols
	 * @param pformat
	 * @param psize
	 * @param rowBlockIndex
	 * @param colBlockIndex
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock readPartitionFromDistCache(String dir, boolean localFS, 
			long rows, long cols, int rowBlockSize, int colBlockSize,
			int partID, int partSize) throws DMLRuntimeException {
		
		dir = dir + "/" + partID;
		
		try {
			MatrixBlock partition = readMatrixFromHDFS(dir, InputInfo.BinaryBlockInputInfo, partSize, 1, rowBlockSize, colBlockSize, localFS);
			//System.out.println("Reading complete...");
			return partition;
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}
	
	/**
	 * 
	 * @param srcfileName
	 * @param fileName
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @throws IOException
	 */
	public static void mergeTextcellToMatrixMarket( String srcFileName, String fileName, long rlen, long clen, long nnz )
		throws IOException
	{
		  Configuration conf = new Configuration();
		
		  Path src = new Path (srcFileName);
	      Path merge = new Path (fileName);
	      FileSystem hdfs = FileSystem.get(conf);
	    
	      if (hdfs.exists (merge)) {
	    	hdfs.delete(merge, true);
	      }
        
	      OutputStream out = hdfs.create(merge, true);

	      // write out the header first 
	      StringBuilder  sb = new StringBuilder();
	      sb.append ("%%MatrixMarket matrix coordinate real general\n");
	    
	      // output number of rows, number of columns and number of nnz
	 	  sb.append (rlen + " " + clen + " " + nnz + "\n");
	      out.write (sb.toString().getBytes());

	      // if the source is a directory
	      if (hdfs.getFileStatus(src).isDir()) {
	        try {
	          FileStatus contents[] = hdfs.listStatus(src);
	          for (int i = 0; i < contents.length; i++) {
	            if (!contents[i].isDir()) {
	               InputStream in = hdfs.open (contents[i].getPath());
	               try {
	                 IOUtils.copyBytes (in, out, conf, false);
	               }  finally {
	                 in.close();
	               }
	             }
	           }
	         } finally {
	            out.close();
	         }
	      } else if (hdfs.isFile(src))  {
	        InputStream in = null;
	        try {
   	          in = hdfs.open (src);
	          IOUtils.copyBytes (in, out, conf, true);
	        } finally {
	          in.close();
	          out.close();
	        }
	      } else {
	        throw new IOException(src.toString() + ": No such file or directory");
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
	public static void mergeCSVPartFiles(String srcFileName,
			String destFileName, CSVFileFormatProperties csvprop, long rlen, long clen) 
			throws IOException {
		
		Configuration conf = new Configuration();

		Path srcFilePath = new Path(srcFileName);
		Path mergedFilePath = new Path(destFileName);
		FileSystem hdfs = FileSystem.get(conf);

		if (hdfs.exists(mergedFilePath)) {
			hdfs.delete(mergedFilePath, true);
		}
		OutputStream out = hdfs.create(mergedFilePath, true);

		// write out the header, if needed
		if (csvprop.isHasHeader()) {
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
				FileStatus contents[] = hdfs.listStatus(srcFilePath);
				Path[] partPaths = new Path[contents.length];
				int numPartFiles = 0;
				for (int i = 0; i < contents.length; i++) {
					if (!contents[i].isDir()) {
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
	 * @param fileName
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @throws IOException
	 */
	public static void writeCSVToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, long nnz, FileFormatProperties formatProperties )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		FileSystem fs = FileSystem.get(job);
        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
        
    	int rows = src.getNumRows();
		int cols = src.getNumColumns();

		//bound check per block
		if( rows > rlen || cols > clen )
		{
			throw new IOException("Matrix block [1:"+rows+",1:"+cols+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		
		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			CSVFileFormatProperties csvProperties = (CSVFileFormatProperties)formatProperties;
			String delim = csvProperties.getDelim(); //Pattern.quote(csvProperties.getDelim());
			
			// Write header line, if needed
			if (csvProperties.isHasHeader()) {
				for(int i=0; i < clen; i++) {
					sb.append("C"+ (i+1));
					if ( i < clen-1 )
						sb.append(delim);
				}
				sb.append('\n');
				br.write( sb.toString());
	            sb.setLength(0);
			}
			
			// Write data lines
			if( sparse ) //SPARSE
			{			   
	            int c, prev_c;
	            double v;
				for(int r=0; r < rows; r++) {
					SparseRow spRow = src.getSparseRows()[r];
					prev_c = -1;
					if ( spRow != null) {
						for(int ind=0; ind < spRow.size(); ind++) {
							c = spRow.getIndexContainer()[ind];
							v = spRow.getValueContainer()[ind];
							
							// output empty fields, if needed
							while(prev_c < c-1) {
								if (!csvProperties.isSparse()) {
									sb.append(0.0);
								}
								sb.append(delim);
								prev_c++;
							}
							
							// output the value
							sb.append(v);
							if ( c < clen-1)
								sb.append(delim);
							prev_c = c;
						}
					}
					// Output empty fields at the end of the row.
					// In case of an empty row, output (clen-1) empty fields
					while (prev_c < clen - 1) {
						if (!csvProperties.isSparse()) {
							sb.append(0.0);
						}
						prev_c++;
						if (prev_c < clen-1)
							sb.append(delim);
					}
					sb.append('\n');
					/*if ( sb.toString().split(Pattern.quote(delim)).length != clen) {
						throw new RuntimeException("row = " + r + ", prev_c = " + prev_c + ", filedcount=" + sb.toString().split(Pattern.quote(delim)).length + ": " + sb.toString());
					}*/
					br.write( sb.toString() ); 
					sb.setLength(0); 
				}
			}
			else //DENSE
			{
				for( int i=0; i<rows; i++ ) {
					for( int j=0; j<cols; j++ )
					{
						double lvalue = src.getValueDenseUnsafe(i, j);
						if( lvalue != 0 ) //for nnz
						{
							sb.append(lvalue);
						}
						else {
							if (!csvProperties.isSparse()) {
								// write in dense format
								sb.append(0.0);
							}
						}
						if ( j != cols-1 )
							sb.append(delim);
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
	 * 
	 * @param path
	 * @param job
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 */
	private static void writeTextCellMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		boolean entriesWritten = false;
		FileSystem fs = FileSystem.get(job);
        BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
		
    	int rows = src.getNumRows();
		int cols = src.getNumColumns();

		//bound check per block
		if( rows > rlen || cols > clen )
		{
			throw new IOException("Matrix block [1:"+rows+",1:"+cols+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		
		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			if( sparse ) //SPARSE
			{			   
				SparseRowsIterator iter = src.getSparseRowsIterator();
				while( iter.hasNext() )
				{
					IJV cell = iter.next();

					sb.append(cell.i+1);
					sb.append(' ');
					sb.append(cell.j+1);
					sb.append(' ');
					sb.append(cell.v);
					sb.append('\n');
					br.write( sb.toString() ); //same as append
					sb.setLength(0); 
					entriesWritten = true;					
				}
			}
			else //DENSE
			{
				for( int i=0; i<rows; i++ )
				{
					String rowIndex = Integer.toString(i+1);					
					for( int j=0; j<cols; j++ )
					{
						double lvalue = src.getValueDenseUnsafe(i, j);
						if( lvalue != 0 ) //for nnz
						{
							sb.append(rowIndex);
							sb.append(' ');
							sb.append( j+1 );
							sb.append(' ');
							sb.append( lvalue );
							sb.append('\n');
							br.write( sb.toString() ); //same as append
							sb.setLength(0); 
							entriesWritten = true;
						}
						
					}
				}
			}
	
			//handle empty result
			if ( !entriesWritten ) {
				br.write("1 1 0\n");
			}
		}
		finally
		{
			if( br != null )
				br.close();
		}
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
	private static void writeMatrixMarketToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, long nnz )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		boolean entriesWritten = false;
		FileSystem fs = FileSystem.get(job);
        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
        
    	int rows = src.getNumRows();
		int cols = src.getNumColumns();

		//bound check per block
		if( rows > rlen || cols > clen )
		{
			throw new IOException("Matrix block [1:"+rows+",1:"+cols+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		
		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			// First output MM header
			sb.append ("%%MatrixMarket matrix coordinate real general\n");
		
			// output number of rows, number of columns and number of nnz
			sb.append (rlen + " " + clen + " " + nnz + "\n");
            br.write( sb.toString());
            sb.setLength(0);
            
            // output matrix cell
			if( sparse ) //SPARSE
			{			   
				SparseRowsIterator iter = src.getSparseRowsIterator();
				while( iter.hasNext() )
				{
					IJV cell = iter.next();

					sb.append(cell.i+1);
					sb.append(' ');
					sb.append(cell.j+1);
					sb.append(' ');
					sb.append(cell.v);
					sb.append('\n');
					br.write( sb.toString() ); //same as append
					sb.setLength(0); 
					entriesWritten = true;					
				}
			}
			else //DENSE
			{
				for( int i=0; i<rows; i++ )
				{
					String rowIndex = Integer.toString(i+1);					
					for( int j=0; j<cols; j++ )
					{
						double lvalue = src.getValueDenseUnsafe(i, j);
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
							entriesWritten = true;
						}
					}
				}
			}
	
			//handle empty result
			if ( !entriesWritten ) {
				br.write("1 1 0\n");
			}
		}
		finally
		{
			if( br != null )
				br.close();
		}
	}
	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 */
	private static void writeBinaryCellMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		boolean entriesWritten = false;
		FileSystem fs = FileSystem.get(job);
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixCell.class);
		
		MatrixIndexes indexes = new MatrixIndexes();
		MatrixCell cell = new MatrixCell();

		int rows = src.getNumRows(); 
		int cols = src.getNumColumns();
        
		//bound check per block
		if( rows > rlen || cols > clen )
		{
			throw new IOException("Matrix block [1:"+rows+",1:"+cols+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		
		try
		{
			if( sparse ) //SPARSE
			{
				
				SparseRowsIterator iter = src.getSparseRowsIterator();
				while( iter.hasNext() )
				{
					IJV lcell = iter.next();
					indexes.setIndexes(lcell.i+1, lcell.j+1);
					cell.setValue(lcell.v);
					writer.append(indexes, cell);
					entriesWritten = true;
				}
			}
			else //DENSE
			{
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
					{
						double lvalue  = src.getValueDenseUnsafe(i, j);
						if( lvalue != 0 ) //for nnz
						{
							indexes.setIndexes(i+1, j+1);
							cell.setValue(lvalue);
							writer.append(indexes, cell);
							entriesWritten = true;
						}
					}
			}
	
			//handle empty result
			if ( !entriesWritten ) {
				writer.append(new MatrixIndexes(1, 1), new MatrixCell(0));
			}
		}
		finally
		{
			if( writer != null )
				writer.close();
		}
	}
	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 */
	private static void writeBinaryBlockMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int brlen, int bclen, int replication )
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException
	{
		boolean sparse = src.isInSparseFormat();
		FileSystem fs = FileSystem.get(job);
		
		// 1) create sequence file writer, with right replication factor 
		// (config via 'dfs.replication' not possible since sequence file internally calls fs.getDefaultReplication())
		SequenceFile.Writer writer = null;
		if( replication > 0 ) //if replication specified (otherwise default)
		{
			//copy of SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class), except for replication
			writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class, job.getInt("io.file.buffer.size", 4096),  
					                         (short)replication, fs.getDefaultBlockSize(), null, new SequenceFile.Metadata());
		}
		else	
		{
			writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class);
		}
		
		// 2) bound check for src block
		if( src.getNumRows() > rlen || src.getNumColumns() > clen )
		{
			throw new IOException("Matrix block [1:"+src.getNumRows()+",1:"+src.getNumColumns()+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		
		//3) reblock and write
		try
		{
			MatrixIndexes indexes = new MatrixIndexes();

			if( rlen <= brlen && clen <= bclen ) //opt for single block
			{
				//directly write single block
				indexes.setIndexes(1, 1);
				writer.append(indexes, src);
			}
			else //general case
			{
				//initialize blocks for reuse (at most 4 different blocks required)
				MatrixBlock[] blocks = createMatrixBlocksForReuse(rlen, clen, brlen, bclen, sparse, src.getNonZeros());  
				
				//create and write subblocks of matrix
				for(int blockRow = 0; blockRow < (int)Math.ceil(src.getNumRows()/(double)brlen); blockRow++)
					for(int blockCol = 0; blockCol < (int)Math.ceil(src.getNumColumns()/(double)bclen); blockCol++)
					{
						int maxRow = (blockRow*brlen + brlen < src.getNumRows()) ? brlen : src.getNumRows() - blockRow*brlen;
						int maxCol = (blockCol*bclen + bclen < src.getNumColumns()) ? bclen : src.getNumColumns() - blockCol*bclen;
				
						int row_offset = blockRow*brlen;
						int col_offset = blockCol*bclen;
						
						//get reuse matrix block
						MatrixBlock block = getMatrixBlockForReuse(blocks, maxRow, maxCol, brlen, bclen);
	
						//copy submatrix to block
						src.sliceOperations( row_offset+1, row_offset+maxRow, 
								             col_offset+1, col_offset+maxCol, 
								             block );
						
						//append block to sequence file
						indexes.setIndexes(blockRow+1, blockCol+1);
						writer.append(indexes, block);
						
						//reset block for later reuse
						block.reset();
					}
			}
		}
		finally
		{
			if( writer != null )
				writer.close();
		}
	}
	
	
	public static void writePartitionedBinaryBlockMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int brlen, int bclen, PDataPartitionFormat pformat )
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException
	{
		boolean sparse = src.isInSparseFormat();
		FileSystem fs = FileSystem.get(job);
		
		//initialize blocks for reuse (at most 4 different blocks required)
		MatrixBlock[] blocks = createMatrixBlocksForReuse(rlen, clen, brlen, bclen, sparse, src.getNonZeros());  
		
		switch( pformat )
		{
			case ROW_BLOCK_WISE_N:
			{
				long numBlocks = ((rlen-1)/brlen)+1;
				long numPartBlocks = (long)Math.ceil(((double)DistributedCacheInput.PARTITION_SIZE)/clen/brlen);
						
				int count = 0;		
				for( int k = 0; k<numBlocks; k+=numPartBlocks )
				{
					// 1) create sequence file writer, with right replication factor 
					// (config via 'dfs.replication' not possible since sequence file internally calls fs.getDefaultReplication())
					Path path2 = new Path(path.toString()+File.separator+(++count));
					SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path2, MatrixIndexes.class, MatrixBlock.class);
					
					//3) reblock and write
					try
					{
						MatrixIndexes indexes = new MatrixIndexes();
			
						//create and write subblocks of matrix
						for(int blockRow = k; blockRow < Math.min((int)Math.ceil(src.getNumRows()/(double)brlen),k+numPartBlocks); blockRow++)
							for(int blockCol = 0; blockCol < (int)Math.ceil(src.getNumColumns()/(double)bclen); blockCol++)
							{
								int maxRow = (blockRow*brlen + brlen < src.getNumRows()) ? brlen : src.getNumRows() - blockRow*brlen;
								int maxCol = (blockCol*bclen + bclen < src.getNumColumns()) ? bclen : src.getNumColumns() - blockCol*bclen;
						
								int row_offset = blockRow*brlen;
								int col_offset = blockCol*bclen;
								
								//get reuse matrix block
								MatrixBlock block = getMatrixBlockForReuse(blocks, maxRow, maxCol, brlen, bclen);
			
								//copy submatrix to block
								src.sliceOperations( row_offset+1, row_offset+maxRow, 
										             col_offset+1, col_offset+maxCol, 
										             block );
								
								//append block to sequence file
								indexes.setIndexes(blockRow+1, blockCol+1);
								writer.append(indexes, block);
								
								//reset block for later reuse
								block.reset();
							}
					}
					finally
					{
						if( writer != null )
							writer.close();
					}
				}
				break;
			}
			case COLUMN_BLOCK_WISE_N:
			{
				long numBlocks = ((clen-1)/bclen)+1;
				long numPartBlocks = (long)Math.ceil(((double)DistributedCacheInput.PARTITION_SIZE)/rlen/bclen);
				
				int count = 0;		
				for( int k = 0; k<numBlocks; k+=numPartBlocks )
				{
					// 1) create sequence file writer, with right replication factor 
					// (config via 'dfs.replication' not possible since sequence file internally calls fs.getDefaultReplication())
					Path path2 = new Path(path.toString()+File.separator+(++count));
					SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path2, MatrixIndexes.class, MatrixBlock.class);
					
					//3) reblock and write
					try
					{
						MatrixIndexes indexes = new MatrixIndexes();
			
						//create and write subblocks of matrix
						for(int blockRow = 0; blockRow < (int)Math.ceil(src.getNumRows()/(double)brlen); blockRow++)
							for(int blockCol = k; blockCol < Math.min((int)Math.ceil(src.getNumColumns()/(double)bclen),k+numPartBlocks); blockCol++)
							{
								int maxRow = (blockRow*brlen + brlen < src.getNumRows()) ? brlen : src.getNumRows() - blockRow*brlen;
								int maxCol = (blockCol*bclen + bclen < src.getNumColumns()) ? bclen : src.getNumColumns() - blockCol*bclen;
						
								int row_offset = blockRow*brlen;
								int col_offset = blockCol*bclen;
								
								//get reuse matrix block
								MatrixBlock block = getMatrixBlockForReuse(blocks, maxRow, maxCol, brlen, bclen);
			
								//copy submatrix to block
								src.sliceOperations( row_offset+1, row_offset+maxRow, 
										             col_offset+1, col_offset+maxCol, 
										             block );
								
								//append block to sequence file
								indexes.setIndexes(blockRow+1, blockCol+1);
								writer.append(indexes, block);
								
								//reset block for later reuse
								block.reset();
							}
					}
					finally
					{
						if( writer != null )
							writer.close();
					}
				}
				break;
			}
				
			default:
				throw new DMLRuntimeException("Unsupported partition format for distributed cache input: "+pformat);
		}
	}
	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static void readTextCellMatrixFromHDFS( Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, IllegalAccessException, InstantiationException
	{
		boolean sparse = dest.isInSparseFormat();
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		
		LongWritable key = new LongWritable();
		Text value = new Text();
		int row = -1;
		int col = -1;
		
		try
		{
			FastStringTokenizer st = new FastStringTokenizer(' ');
			
			for(InputSplit split: splits)
			{
				RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
			
				try
				{
					if( sparse ) //SPARSE<-value
					{
						while( reader.next(key, value) )
						{
							st.reset( value.toString() ); //reinit tokenizer
							row = st.nextInt() - 1;
							col = st.nextInt() - 1;
							double lvalue = st.nextDouble();
							dest.appendValue(row, col, lvalue);
						}
						
						dest.sortSparseRows();
					} 
					else //DENSE<-value
					{
						while( reader.next(key, value) )
						{
							st.reset( value.toString() ); //reinit tokenizer
							row = st.nextInt()-1;
							col = st.nextInt()-1;
							double lvalue = st.nextDouble();
							dest.setValueDenseUnsafe( row, col, lvalue );
						}
					}
				}
				finally
				{
					if( reader != null )
						reader.close();
				}
			}
		}
		catch(Exception ex)
		{
			//post-mortem error handling and bounds checking
			if( row < 0 || row + 1 > rlen || col < 0 || col + 1 > clen )
			{
				throw new IOException("Matrix cell ["+(row+1)+","+(col+1)+"] " +
									  "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
			else
			{
				throw new IOException( "Unable to read matrix in text cell format.", ex );
			}
		}
	}

	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static void readRawTextCellMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen, boolean matrixMarket )
		throws IOException, IllegalAccessException, InstantiationException
	{
		//create input stream for path
		InputStream inputStream = fs.open(path);
		
		//actual read
		readRawTextCellMatrixFromInputStream(inputStream, dest, rlen, clen, brlen, bclen, matrixMarket);
	}
	
	/**
	 * 
	 * @param is
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param matrixMarket
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static void readRawTextCellMatrixFromInputStream( InputStream is, MatrixBlock dest, long rlen, long clen, int brlen, int bclen, boolean matrixMarket )
			throws IOException, IllegalAccessException, InstantiationException
	{
		BufferedReader br = new BufferedReader(new InputStreamReader( is ));	
		
		boolean sparse = dest.isInSparseFormat();
		String value = null;
		int row = -1;
		int col = -1;
		
		// Read the header lines, if reading from a matrixMarket file
		if ( matrixMarket ) {
			value = br.readLine(); // header line
			if ( !value.startsWith("%%") ) {
				throw new IOException("Error while reading file in MatrixMarket format. Expecting a header line, but encountered, \"" + value +"\".");
			}
			
			// skip until end-of-comments
			do {
				value = br.readLine();
			} while(value.charAt(0) == '%');
			
			// the first line after comments is the one w/ matrix dimensions
			
			//value = br.readLine(); // line with matrix dimensions
			
			// validate (rlen clen nnz)
			String[] fields = value.trim().split("\\s+"); 
			long mm_rlen = Long.parseLong(fields[0]);
			long mm_clen = Long.parseLong(fields[1]);
			if ( rlen != mm_rlen || clen != mm_clen ) {
				throw new IOException("Unexpected matrix dimensions while reading file in MatrixMarket format. Expecting dimensions [" + rlen + " rows, " + clen + " cols] but encountered [" + mm_rlen + " rows, " + mm_clen + "cols].");
			}
		}
		
		try
		{			
			FastStringTokenizer st = new FastStringTokenizer(' ');
			
			if( sparse ) //SPARSE<-value
			{
				while( (value=br.readLine())!=null )
				{
					st.reset( value ); //reinit tokenizer
					row = st.nextInt()-1;
					col = st.nextInt()-1;
					double lvalue = st.nextDouble();
					dest.appendValue(row, col, lvalue);
				}
				
				dest.sortSparseRows();
			} 
			else //DENSE<-value
			{
				while( (value=br.readLine())!=null )
				{
					st.reset( value ); //reinit tokenizer
					row = st.nextInt()-1;
					col = st.nextInt()-1;	
					double lvalue = st.nextDouble();
					dest.setValueDenseUnsafe( row, col, lvalue );
				}
			}
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			
			//post-mortem error handling and bounds checking
			if( row < 0 || row + 1 > rlen || col < 0 || col + 1 > clen ) 
			{
				throw new IOException("Matrix cell ["+(row+1)+","+(col+1)+"] " +
									  "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
			else
			{
				throw new IOException( "Unable to read matrix in raw text cell format.", ex );
			}
		}
		finally
		{
			if( br != null )
				br.close();
		}
	}
	
	private static MatrixBlock computeCSVSize ( List<Path> files, JobConf job, FileSystem fs, boolean hasHeader, String delim, 
			boolean fill, double fillValue) throws IOException {
		
		int nrow = -1;
		int ncol = -1;
		String value = null;
		
		String escapedDelim = Pattern.quote(delim);
		MatrixBlock dest = new MatrixBlock();
		String headerLine = null, cellStr = null;
		for(int fileNo=0; fileNo<files.size(); fileNo++)
		{
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));	
			try
			{
				// Read the header line, if there is one.
				if(fileNo==0)
				{
					if ( hasHeader ) 
						headerLine = br.readLine();
					value = br.readLine();
					cellStr = value.toString().trim();
					ncol = cellStr.split(escapedDelim,-1).length;
					nrow = 1;
				}
				
				while ( (value = br.readLine()) != null ) {
					nrow++;
				}
			}
			finally
			{
				if( br != null )
					br.close();
			}
		}
		dest = new MatrixBlock(nrow, ncol, true);
		return dest;
	}
	
	/*private static MatrixBlock computeCSVSize ( Path path, JobConf job, FileSystem fs, boolean hasHeader, String delim, 
			boolean fill, double fillValue) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));	

		int nrow = -1;
		int ncol = -1;
		String value = null;
		
		String escapedDelim = Pattern.quote(delim);
		MatrixBlock dest = new MatrixBlock();
		try
		{
			// Read the header line, if there is one.
			String headerLine = null, cellStr = null;
			if ( hasHeader ) 
				headerLine = br.readLine();
			
			value = br.readLine();
			cellStr = value.toString().trim();
			ncol = cellStr.split(escapedDelim,-1).length;
			nrow = 1;
			
			while ( (value = br.readLine()) != null ) {
				nrow++;
			}
			
			dest = new MatrixBlock(nrow, ncol, true);
		}
		finally
		{
			if( br != null )
				br.close();
		}
		return dest;
	}*/
	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static MatrixBlock readCSVMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, 
			long rlen, long clen, int brlen, int bclen, boolean hasHeader, String delim, boolean fill, double fillValue )
		throws IOException, IllegalAccessException, InstantiationException
	{
		ArrayList<Path> files=new ArrayList<Path>();
		if(fs.isDirectory(path))
		{
			for(FileStatus stat: fs.listStatus(path, CSVReblockMR.hiddenFileFilter))
				files.add(stat.getPath());
			Collections.sort(files);
		}else
			files.add(path);
		
		if ( dest == null ) {
			dest = computeCSVSize(files, job, fs, hasHeader, delim, fill, fillValue);
			rlen = dest.getNumRows();
			clen = dest.getNumColumns();
		}
		
		boolean sparse = dest.isInSparseFormat();
		
		/////////////////////////////////////////
		String value = null;
		int row = 0;
		int col = -1;
		double cellValue = 0;
		
		String escapedDelim = Pattern.quote(delim);
		String headerLine = null, cellStr = null;
		
		for(int fileNo=0; fileNo<files.size(); fileNo++)
		{
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));
			if(fileNo==0 && hasHeader ) 
					headerLine = br.readLine();
			
			// Read the data
			boolean emptyValuesFound = false;
			try{
				if( sparse ) //SPARSE<-value
				{
					while( (value=br.readLine())!=null )
					{
						col = 0;
						cellStr = value.toString().trim();
						emptyValuesFound = false;
						for(String part : cellStr.split(escapedDelim, -1)) {
							part = part.trim();
							if ( part.isEmpty() ) {
								emptyValuesFound = true;
								cellValue = fillValue;
							}
							else {
								cellValue = Double.parseDouble(part);
							}
							if ( Double.compare(cellValue, 0.0) != 0 )
								dest.appendValue(row, col, cellValue);
							col++;
						}
						if ( !fill && emptyValuesFound) {
							throw new IOException("Empty fields found in delimited file (" + path.toString() + "). Use \"fill\" option to read delimited files with empty fields." + cellStr);
						}
						if ( col != clen ) {
							throw new IOException("Invalid number of columns (" + col + ") found in delimited file (" + path.toString() + "). Expecting (" + clen + "): " + value);
						}
						row++;
					}
				} 
				else //DENSE<-value
				{
					while( (value=br.readLine())!=null )
					{
						cellStr = value.toString().trim();
						col = 0;
						for(String part : cellStr.split(escapedDelim, -1)) {
							part = part.trim();
							if ( part.isEmpty() ) {
								if ( !fill ) {
									throw new IOException("Empty fields found in delimited file (" + path.toString() + "). Use \"fill\" option to read delimited files with empty fields.");
								}
								else {
									cellValue = fillValue;
								}
							}
							else {
								cellValue = Double.parseDouble(part);
							}
							dest.setValueDenseUnsafe(row, col, cellValue);
							col++;
						}
						if ( col != clen ) {
							throw new IOException("Invalid number of columns (" + col + ") found in delimited file (" + path.toString() + "). Expecting (" + clen + "): " + value);
						}
						row++;
					}
				}
			}
			finally
			{
				if( br != null )
					br.close();
			}
		}
		
		dest.recomputeNonZeros();
		return dest;
		/*
		//FileSystem fs = FileSystem.get(job);
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));	
		
		String value = null;
		int row = -1;
		int col = -1;
		double cellValue = 0;
		
		String escapedDelim = Pattern.quote(delim);
		
		try
		{
			// Read the header line, if there is one.
			String headerLine = null, cellStr = null;
			if ( hasHeader ) 
				headerLine = br.readLine();
			
			// Read the data
			boolean emptyValuesFound = false;
			if( sparse ) //SPARSE<-value
			{
				row = 0;
				while( (value=br.readLine())!=null )
				{
					col = 0;
					cellStr = value.toString().trim();
					emptyValuesFound = false;
					for(String part : cellStr.split(escapedDelim, -1)) {
						part = part.trim();
						if ( part.isEmpty() ) {
							emptyValuesFound = true;
							cellValue = fillValue;
						}
						else {
							cellValue = Double.parseDouble(part);
						}
						if ( Double.compare(cellValue, 0.0) != 0 )
							dest.appendValue(row, col, cellValue);
						col++;
					}
					if ( !fill && emptyValuesFound) {
						throw new IOException("Empty fields found in delimited file (" + path.toString() + "). Use \"fill\" option to read delimited files with empty fields." + cellStr);
					}
					if ( col != clen ) {
						throw new IOException("Invalid number of columns (" + col + ") found in delimited file (" + path.toString() + "). Expecting (" + clen + "): " + value);
					}
					row++;
				}
				dest.recomputeNonZeros();
			} 
			else //DENSE<-value
			{
				row = 0;
				while( (value=br.readLine())!=null )
				{
					cellStr = value.toString().trim();
					col = 0;
					for(String part : cellStr.split(escapedDelim, -1)) {
						part = part.trim();
						if ( part.isEmpty() ) {
							if ( !fill ) {
								throw new IOException("Empty fields found in delimited file (" + path.toString() + "). Use \"fill\" option to read delimited files with empty fields.");
							}
							else {
								cellValue = fillValue;
							}
						}
						else {
							cellValue = Double.parseDouble(part);
						}
						dest.setValueDenseUnsafe(row, col, cellValue);
						col++;
					}
					if ( col != clen ) {
						throw new IOException("Invalid number of columns (" + col + ") found in delimited file (" + path.toString() + "). Expecting (" + clen + "): " + value);
					}
					row++;
				}
				dest.recomputeNonZeros();
			}
		}
		finally
		{
			if( br != null )
				br.close();
		}
		*/
	}

	
	/**
	 * Note: see readBinaryBlockMatrixFromHDFS for why we directly use SequenceFile.Reader.
	 * 
	 * @param path
	 * @param job
	 * @param fs 
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static void readBinaryCellMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, IllegalAccessException, InstantiationException
	{
		boolean sparse = dest.isInSparseFormat();		
		MatrixIndexes key = new MatrixIndexes();
		MatrixCell value = new MatrixCell();
		int row = -1;
		int col = -1;
		
		try
		{
			for( Path lpath : getSequenceFilePaths(fs,path) ) //1..N files 
			{
				//directly read from sequence files (individual partfiles)
				SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
				
				try
				{
					if( sparse )
					{
						while(reader.next(key, value))
						{
							row = (int)key.getRowIndex()-1;
							col = (int)key.getColumnIndex()-1;
							double lvalue = value.getValue();
							//dest.quickSetValue( row, col, lvalue );
							dest.appendValue(row, col, lvalue);
						}
					}
					else
					{
						while(reader.next(key, value))
						{
							row = (int)key.getRowIndex()-1;
							col = (int)key.getColumnIndex()-1;
							double lvalue = value.getValue();
							dest.setValueDenseUnsafe( row, col, lvalue );
						}
					}
				}
				finally
				{
					if( reader != null )
						reader.close();
				}
			}
			
			if( sparse )
				dest.sortSparseRows();
		}
		catch(Exception ex)
		{
			//post-mortem error handling and bounds checking
			if( row < 0 || row + 1 > rlen || col < 0 || col + 1 > clen )
			{
				throw new IOException("Matrix cell ["+(row+1)+","+(col+1)+"] " +
									  "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
			else
			{
				throw new IOException( "Unable to read matrix in binary cell format.", ex );
			}
		}
	}
	
	/**
	 * Note: For efficiency, we directly use SequenceFile.Reader instead of SequenceFileInputFormat-
	 * InputSplits-RecordReader (SequenceFileRecordReader). First, this has no drawbacks since the
	 * SequenceFileRecordReader internally uses SequenceFile.Reader as well. Second, it is 
	 * advantageous if the actual sequence files are larger than the file splits created by   
	 * informat.getSplits (which is usually aligned to the HDFS block size) because then there is 
	 * overhead for finding the actual split between our 1k-1k blocks. This case happens
	 * if the read matrix was create by CP or when jobs directly write to large output files 
	 * (e.g., parfor matrix partitioning).
	 * 
	 * @param path
	 * @param job
	 * @param fs 
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static void readBinaryBlockMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, IllegalAccessException, InstantiationException
	{
		boolean sparse = dest.isInSparseFormat();
		MatrixIndexes key = new MatrixIndexes(); 
		MatrixBlock value = new MatrixBlock();
		
		for( Path lpath : getSequenceFilePaths(fs, path) ) //1..N files 
		{
			//directly read from sequence files (individual partfiles)
			SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
			
			try
			{
				while( reader.next(key, value) )
				{	
					//empty block filter (skip entire block)
					if( value.isEmptyBlock(false) )
						continue;
					
					int row_offset = (int)(key.getRowIndex()-1)*brlen;
					int col_offset = (int)(key.getColumnIndex()-1)*bclen;
					
					int rows = value.getNumRows();
					int cols = value.getNumColumns();
					
					//bound check per block
					if( row_offset + rows < 0 || row_offset + rows > rlen || col_offset + cols<0 || col_offset + cols > clen )
					{
						throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
								              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
					}
			
					//copy block to result
					if( sparse )
					{
						dest.appendToSparse(value, row_offset, col_offset);
						//note: append requires final sort
					} 
					else
					{
						dest.copy( row_offset, row_offset+rows-1, 
								   col_offset, col_offset+cols-1,
								   value, false );
					}
				}
			}
			finally
			{
				if( reader != null )
					reader.close();
			}
		}
		
		if( sparse && clen>bclen ){
			//no need to sort if 1 column block since always sorted
			dest.sortSparseRows();
		}
	}
	
	
	private static void readBinaryBlockMatrixBlocksFromHDFS( Path path, JobConf job, FileSystem fs, Collection<IndexedMatrixValue> dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, IllegalAccessException, InstantiationException
	{
		MatrixIndexes key = new MatrixIndexes(); 
		MatrixBlock value = new MatrixBlock();
			
		for( Path lpath : getSequenceFilePaths(fs, path) ) //1..N files 
		{
			//directly read from sequence files (individual partfiles)
			SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
			
			try
			{
				while( reader.next(key, value) )
				{	
					int row_offset = (int)(key.getRowIndex()-1)*brlen;
					int col_offset = (int)(key.getColumnIndex()-1)*bclen;
					int rows = value.getNumRows();
					int cols = value.getNumColumns();
					
					//bound check per block
					if( row_offset + rows < 0 || row_offset + rows > rlen || col_offset + cols<0 || col_offset + cols > clen )
					{
						throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
								              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
					}
			
					//copy block to result
					dest.add(new IndexedMatrixValue(new MatrixIndexes(key), new MatrixBlock(value)));
				}
			}
			finally
			{
				if( reader != null )
					reader.close();
			}
		}
	}
	
	
	//////////////
	// Utils for CREATING and COPYING matrix blocks 
	///////
	
	/**
	 * Creates a two-dimensional double matrix of the input matrix block. 
	 * 
	 * @param mb
	 * @return
	 */
	public static double[][] convertToDoubleMatrix( MatrixBlock mb )
	{
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		double[][] ret = new double[rows][cols];
		
		if( mb.isInSparseFormat() )
		{
			SparseRowsIterator iter = mb.getSparseRowsIterator();
			while( iter.hasNext() )
			{
				IJV cell = iter.next();
				ret[cell.i][cell.j] = cell.v;
			}
		}
		else
		{
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
					ret[i][j] = mb.getValueDenseUnsafe(i, j);
		}
				
		return ret;
	}
	
	/**
	 * Creates a dense Matrix Block and copies the given double matrix into it.
	 * 
	 * @param data
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock convertToMatrixBlock( double[][] data ) 
		throws DMLRuntimeException
	{
		int rows = data.length;
		int cols = (rows > 0)? data[0].length : 0;
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		try
		{ 
			//copy data to mb (can be used because we create a dense matrix)
			mb.init( data, rows, cols );
		} 
		catch (Exception e){} //can never happen
		
		//check and convert internal representation
		mb.examSparsity();
		
		return mb;
	}

	/**
	 * Creates a dense Matrix Block and copies the given double vector into it.
	 * 
	 * @param data
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock convertToMatrixBlock( double[] data, boolean columnVector ) 
		throws DMLRuntimeException
	{
		int rows = columnVector ? data.length : 1;
		int cols = columnVector ? 1 : data.length;
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		
		try
		{ 
			//copy data to mb (can be used because we create a dense matrix)
			mb.init( data, rows, cols );
		} 
		catch (Exception e){} //can never happen
		
		//check and convert internal representation
		mb.examSparsity();
		
		return mb;
	}

	/**
	 * 
	 * @param map
	 * @return
	 */
	public static MatrixBlock convertToMatrixBlock( HashMap<MatrixIndexes,Double> map )
	{
		// compute dimensions from the map
		long nrows=0, ncols=0;
		for (MatrixIndexes index : map.keySet()) {
			nrows = Math.max( nrows, index.getRowIndex() );
			ncols = Math.max( ncols, index.getColumnIndex() );
		}
		
		int rlen = (int)nrows;
		int clen = (int)ncols;
		int nnz = map.size();
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz); 		
		MatrixBlock mb = new MatrixBlock(rlen, clen, sparse, nnz);
		
		// copy map values into new block
		for (MatrixIndexes index : map.keySet()) {
			double value  = map.get(index).doubleValue();
			if ( value != 0 )
			{
				mb.quickSetValue( (int)index.getRowIndex()-1, 
						          (int)index.getColumnIndex()-1, 
						          value );
			}
		}
		
		return mb;
	}
	
	/////////////////////////////////////////////
	// Helper methods for the specific formats //
	/////////////////////////////////////////////

	/**
	 * 
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param sparse
	 * @return
	 */
	public static MatrixBlock[] createMatrixBlocksForReuse( long rlen, long clen, int brlen, int bclen, boolean sparse, long nonZeros )
	{
		MatrixBlock[] blocks = new MatrixBlock[4];
		double sparsity = ((double)nonZeros)/(rlen*clen);
		long estNNZ = -1;
		
		//full block 
		if( rlen >= brlen && clen >= bclen )
		{
			estNNZ = (long) (brlen*bclen*sparsity);
			blocks[0] = new MatrixBlock( brlen, bclen, sparse, (int)estNNZ );
		}
		//partial col block
		if( rlen >= brlen && clen%bclen!=0 )
		{
			estNNZ = (long) (brlen*(clen%bclen)*sparsity);
			blocks[1] = new MatrixBlock( brlen, (int)(clen%bclen), sparse, (int)estNNZ );
		}
		//partial row block
		if( rlen%brlen!=0 && clen>=bclen )
		{
			estNNZ = (long) ((rlen%brlen)*bclen*sparsity);
			blocks[2] = new MatrixBlock( (int)(rlen%brlen), bclen, sparse, (int)estNNZ );
		}
		//partial row/col block
		if( rlen%brlen!=0 && clen%bclen!=0 )
		{
			estNNZ = (long) ((rlen%brlen)*(clen%bclen)*sparsity);
			blocks[3] = new MatrixBlock( (int)(rlen%brlen), (int)(clen%bclen), sparse, (int)estNNZ );
		}
		
		//space allocation
		for( MatrixBlock b : blocks )
			if( b != null )
				if( !sparse )
					b.allocateDenseBlockUnsafe(b.getNumRows(), b.getNumColumns());		
		//NOTE: no preallocation for sparse (preallocate sparserows with estnnz) in order to reduce memory footprint
		
		return blocks;
	}
	
	/**
	 * 
	 * @param blocks
	 * @param rows
	 * @param cols
	 * @param brlen
	 * @param bclen
	 * @return
	 */
	public static MatrixBlock getMatrixBlockForReuse( MatrixBlock[] blocks, int rows, int cols, int brlen, int bclen )
	{
		int index = -1;
		
		if( rows==brlen && cols==bclen )
			index = 0;
		else if( rows==brlen && cols<bclen )
			index = 1;
		else if( rows<brlen && cols==bclen )
			index = 2;
		else //if( rows<brlen && cols<bclen )
			index = 3;

		return blocks[ index ];
	}

	/**
	 * 
	 * @param file
	 * @return
	 * @throws IOException
	 */
	public static Path[] getSequenceFilePaths( FileSystem fs, Path file ) 
		throws IOException
	{
		Path[] ret = null;
		
		if( fs.isDirectory(file) )
		{
			LinkedList<Path> tmp = new LinkedList<Path>();
			FileStatus[] dStatus = fs.listStatus(file);
			for( FileStatus fdStatus : dStatus )
				if( !fdStatus.getPath().getName().startsWith("_") ) //skip internal files
					tmp.add(fdStatus.getPath());
			ret = tmp.toArray(new Path[0]);
		}
		else
		{
			ret = new Path[]{ file };
		}
		
		return ret;
	}
}

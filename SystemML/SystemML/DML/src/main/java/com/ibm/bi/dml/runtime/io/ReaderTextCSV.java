/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;

public class ReaderTextCSV extends MatrixReader
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	CSVFileFormatProperties _props = null;
	
	public ReaderTextCSV(CSVFileFormatProperties props)
	{
		_props = props;
	}
	

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = null;
		if( rlen>0 && clen>0 ) //otherwise CSV reblock based on file size for matrix w/ unknown dimensions
			createOutputMatrixBlock(rlen, clen, estnnz, true);
		
		//prepare file access
		JobConf job = new JobConf();	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname );
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		ret = readCSVMatrixFromHDFS(path, job, fs, ret, rlen, clen, brlen, bclen, 
				   _props.hasHeader(), _props.getDelim(), _props.isFill(), _props.getFillValue() );
		
		//finally check if change of sparse/dense block representation required
		if( !ret.isInSparseFormat() )
			ret.recomputeNonZeros();
		ret.examSparsity();
		
		return ret;
	}

	/**
	 * 
	 * @param path
	 * @param job
	 * @param fs
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @param fillValue
	 * @return
	 * @throws IOException
	 */
	@SuppressWarnings("unchecked")
	private MatrixBlock readCSVMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, 
			long rlen, long clen, int brlen, int bclen, boolean hasHeader, String delim, boolean fill, double fillValue )
		throws IOException
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
			clen = dest.getNumColumns();
		}
		
		boolean sparse = dest.isInSparseFormat();
		
		/////////////////////////////////////////
		String value = null;
		int row = 0;
		int col = -1;
		double cellValue = 0;
		
		String escapedDelim = Pattern.quote(delim);
		String cellStr = null;
		
		for(int fileNo=0; fileNo<files.size(); fileNo++)
		{
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));
			if(fileNo==0 && hasHeader ) 
				br.readLine(); //ignore header
			
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
	}
	
	/**
	 * 
	 * @param files
	 * @param job
	 * @param fs
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @param fillValue
	 * @return
	 * @throws IOException
	 */
	private MatrixBlock computeCSVSize ( List<Path> files, JobConf job, FileSystem fs, boolean hasHeader, String delim, boolean fill, double fillValue) 
		throws IOException 
	{		
		int nrow = -1;
		int ncol = -1;
		String value = null;
		
		String escapedDelim = Pattern.quote(delim);
		String cellStr = null;
		for(int fileNo=0; fileNo<files.size(); fileNo++)
		{
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));	
			try
			{
				// Read the header line, if there is one.
				if(fileNo==0)
				{
					if ( hasHeader ) 
						br.readLine(); //ignore header
					if( (value = br.readLine()) != null ) {
						cellStr = value.toString().trim();
						ncol = cellStr.split(escapedDelim,-1).length;
						nrow = 1;
					}
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
		
		return new MatrixBlock(nrow, ncol, true);
	}
}

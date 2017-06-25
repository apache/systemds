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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.mutable.MutableInt;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.CSVReblockMR;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.UtilFunctions;

public class ReaderTextCSV extends MatrixReader
{
	private CSVFileFormatProperties _props = null;
	
	public ReaderTextCSV(CSVFileFormatProperties props) {
		_props = props;
	}
	
	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = null;
		if( rlen>0 && clen>0 ) //otherwise allocated on read
			ret = createOutputMatrixBlock(rlen, clen, (int)rlen, (int)clen, estnnz, true, false);
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		Path path = new Path( fname );
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		ret = readCSVMatrixFromHDFS(path, job, fs, ret, rlen, clen, brlen, bclen, 
				   _props.hasHeader(), _props.getDelim(), _props.isFill(), _props.getFillValue() );
		
		//finally check if change of sparse/dense block representation required
		//(nnz explicitly maintained during read)
		ret.examSparsity();
		
		return ret;
	}
	
	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, (int)rlen, (int)clen, estnnz, true, false);
		
		//core read 
		long lnnz = readCSVMatrixFromInputStream(is, "external inputstream", ret, new MutableInt(0), rlen, clen, 
			brlen, bclen, _props.hasHeader(), _props.getDelim(), _props.isFill(), _props.getFillValue(), true);
				
		//finally check if change of sparse/dense block representation required
		ret.setNonZeros( lnnz );
		ret.examSparsity();
		
		return ret;
	}
	
	@SuppressWarnings("unchecked")
	private MatrixBlock readCSVMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, 
			long rlen, long clen, int brlen, int bclen, boolean hasHeader, String delim, boolean fill, double fillValue )
		throws IOException, DMLRuntimeException
	{
		//prepare file paths in alphanumeric order
		ArrayList<Path> files=new ArrayList<Path>();
		if(fs.isDirectory(path)) {
			for(FileStatus stat: fs.listStatus(path, CSVReblockMR.hiddenFileFilter))
				files.add(stat.getPath());
			Collections.sort(files);
		}
		else
			files.add(path);
		
		//determine matrix size via additional pass if required
		if ( dest == null ) {
			dest = computeCSVSize(files, job, fs, hasHeader, delim, fill, fillValue);
			clen = dest.getNumColumns();
		}
		
		//actual read of individual files
		long lnnz = 0;
		MutableInt row = new MutableInt(0);
		for(int fileNo=0; fileNo<files.size(); fileNo++) {
			lnnz += readCSVMatrixFromInputStream(fs.open(files.get(fileNo)), path.toString(), dest, 
				row, rlen, clen, brlen, bclen, hasHeader, delim, fill, fillValue, fileNo==0);
		}
		
		//post processing
		dest.setNonZeros( lnnz );
		
		return dest;
	}
	
	private long readCSVMatrixFromInputStream( InputStream is, String srcInfo, MatrixBlock dest, MutableInt rowPos, 
			long rlen, long clen, int brlen, int bclen, boolean hasHeader, String delim, boolean fill, double fillValue, boolean first )
		throws IOException
	{
		boolean sparse = dest.isInSparseFormat();
		String value = null;
		int row = rowPos.intValue();
		double cellValue = 0;
		long lnnz = 0;
		
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		if(first && hasHeader ) 
			br.readLine(); //ignore header
		
		// Read the data
		boolean emptyValuesFound = false;
		try
		{
			if( sparse ) //SPARSE<-value
			{
				while( (value=br.readLine())!=null ) //foreach line
				{
					String cellStr = value.toString().trim();
					emptyValuesFound = false;
					String[] parts = IOUtilFunctions.split(cellStr, delim);
					int col = 0;
					
					for(String part : parts) //foreach cell
					{
						part = part.trim();
						if ( part.isEmpty() ) {
							emptyValuesFound = true;
							cellValue = fillValue;
						}
						else {
							cellValue = UtilFunctions.parseToDouble(part);
						}
						if ( cellValue != 0 ) {
							dest.appendValue(row, col, cellValue);
							lnnz++;
						}
						col++;
					}
					
					//sanity checks for empty values and number of columns
					IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(cellStr, fill, emptyValuesFound);
					IOUtilFunctions.checkAndRaiseErrorCSVNumColumns(srcInfo, cellStr, parts, clen);
					row++;
				}
			} 
			else //DENSE<-value
			{
				while( (value=br.readLine())!=null ) //foreach line
				{
					String cellStr = value.toString().trim();
					emptyValuesFound = false;
					String[] parts = IOUtilFunctions.split(cellStr, delim);
					int col = 0;
					
					for( String part : parts ) //foreach cell
					{
						part = part.trim();
						if ( part.isEmpty() ) {
							emptyValuesFound = true;
							cellValue = fillValue;
						}
						else {
							cellValue = UtilFunctions.parseToDouble(part);
						}
						if ( cellValue != 0 ) {
							dest.setValueDenseUnsafe(row, col, cellValue);
							lnnz++;
						}
						col++;
					}
					
					//sanity checks for empty values and number of columns
					IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(cellStr, fill, emptyValuesFound);
					IOUtilFunctions.checkAndRaiseErrorCSVNumColumns(srcInfo, cellStr, parts, clen);
					row++;
				}
			}
		}
		finally {
			IOUtilFunctions.closeSilently(br);
		}
		
		rowPos.setValue(row);
		return lnnz;
	}

	private MatrixBlock computeCSVSize( List<Path> files, JobConf job, FileSystem fs, boolean hasHeader, String delim, boolean fill, double fillValue) 
		throws IOException, DMLRuntimeException 
	{		
		int nrow = -1;
		int ncol = -1;
		String value = null;
		
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
						ncol = StringUtils.countMatches(cellStr, delim) + 1;
						nrow = 1;
					}
				}
				
				while ( (value = br.readLine()) != null ) {
					nrow++;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(br);
			}
		}
		
		// allocate target matrix block based on given size; 
		return createOutputMatrixBlock(nrow, ncol, 
			nrow, ncol, (long)nrow*ncol, true, false);
	}
}

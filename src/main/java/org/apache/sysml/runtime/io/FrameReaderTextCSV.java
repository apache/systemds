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
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.CSVReblockMR;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;

public class FrameReaderTextCSV extends FrameReader
{

	private CSVFileFormatProperties _props = null;
	
	public FrameReaderTextCSV(CSVFileFormatProperties props)
	{
		_props = props;
	}
	

	/**
	 * 
	 * @param fname
	 * @param schema
	 * @param names
	 * @param rlen
	 * @param clen
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	@Override
	public FrameBlock readFrameFromHDFS(String fname, List<ValueType> schema, List<String> names,
			long rlen, long clen)
		throws IOException, DMLRuntimeException 
	{
		//allocate output frame block
		FrameBlock ret = null;
		if( rlen>0 && clen>0 ) //otherwise CSV reblock based on file size for frame w/ unknown dimensions
			ret = createOutputFrameBlock(schema, names, rlen);
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname );
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		ret = readCSVFrameFromHDFS(path, job, fs, ret, schema, names, rlen, clen,  
				   _props.hasHeader(), _props.getDelim(), _props.isFill() );
		
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
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @return
	 * @throws IOException
	 */
	@SuppressWarnings("unchecked")
	private FrameBlock readCSVFrameFromHDFS( Path path, JobConf job, FileSystem fs, FrameBlock dest, 
			List<ValueType> schema, List<String> names, long rlen, long clen, boolean hasHeader, String delim, boolean fill)
		throws IOException
	{
		ArrayList<Path> files=new ArrayList<Path>();
		if(fs.isDirectory(path)) {
			for(FileStatus stat: fs.listStatus(path, CSVReblockMR.hiddenFileFilter))
				files.add(stat.getPath());
			Collections.sort(files);
		}
		else
			files.add(path);
		
		if ( dest == null ) {
			dest = computeCSVSize(files, fs, schema, names, hasHeader, delim);
			clen = dest.getNumColumns();
		}
		
		/////////////////////////////////////////
		String value = null;
		int row = 0;
		int col = -1;
		
		for(int fileNo=0; fileNo<files.size(); fileNo++)
		{
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));
			if(fileNo==0 && hasHeader ) 
				br.readLine(); //ignore header
			
			// Read the data
			boolean emptyValuesFound = false;
			try
			{
				while( (value=br.readLine())!=null ) //foreach line
				{
					String cellStr = value.toString().trim();
					emptyValuesFound = false;
					String[] parts = IOUtilFunctions.split(cellStr, delim);
					col = 0;
					
					for( String part : parts ) //foreach cell
					{
						part = part.trim();
						if ( part.isEmpty() ) {
							//TODO: Do we need to handle empty cell condition?
							emptyValuesFound = true;
						}
						else {
							switch( schema.get(col) ) {
								case STRING:  dest.set(row, col, part); break;
								case BOOLEAN: dest.set(row, col, Boolean.valueOf(part)); break;
								case INT:     dest.set(row, col, Integer.valueOf(part)); break;
								case DOUBLE:  dest.set(row, col, Double.valueOf(part)); break;
								default: throw new RuntimeException("Unsupported value type: " + schema.get(col));
							}
						}
						col++;
					}
					
					//sanity checks for empty values and number of columns
					IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(cellStr, fill, emptyValuesFound);
					IOUtilFunctions.checkAndRaiseErrorCSVNumColumns(path.toString(), cellStr, parts, clen);
					row++;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(br);
			}
		}
		
		return dest;
	}
	
	/**
	 * 
	 * @param files
	 * @param fs
	 * @param schema
	 * @param names
	 * @param hasHeader
	 * @param delim
	 * @return
	 * @throws IOException
	 */
	private FrameBlock computeCSVSize ( List<Path> files, FileSystem fs, List<ValueType> schema, List<String> names, boolean hasHeader, String delim) 
		throws IOException 
	{		
		int nrow = 0;
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
				}
				
				while ( br.readLine() != null ) {
					nrow++;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(br);
			}
		}
		
		//create new frame block
		FrameBlock frameBlock = new FrameBlock(schema, names);
		frameBlock.ensureAllocatedColumns(nrow);
		return frameBlock;
	}

}

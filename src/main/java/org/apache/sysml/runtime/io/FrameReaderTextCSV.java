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

import java.io.IOException;
import java.io.InputStream;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.util.InputStreamInputFormat;
import org.apache.sysml.runtime.util.UtilFunctions;

/**
 * Single-threaded frame text csv reader.
 * 
 */
public class FrameReaderTextCSV extends FrameReader
{
	protected CSVFileFormatProperties _props = null;
	
	public FrameReaderTextCSV(CSVFileFormatProperties props) {
		_props = props;
	}

	@Override
	public final FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names,
			long rlen, long clen)
		throws IOException, DMLRuntimeException 
	{
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		Path path = new Path( fname );
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		FileInputFormat.addInputPath(job, path);
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
		
		//compute size if necessary
		if( rlen <= 0 || clen <= 0 ) {
			Pair<Integer,Integer> size = computeCSVSize(path, job, fs);
			rlen = size.getKey();
			clen = size.getValue();
		}
		
		//allocate output frame block
		ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNames(names, clen);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);
	
		//core read (sequential/parallel) 
		readCSVFrameFromHDFS(path, job, fs, ret, lschema, lnames, rlen, clen);
		
		return ret;
	}
	
	@Override
	public FrameBlock readFrameFromInputStream(InputStream is, ValueType[] schema, String[] names, 
			long rlen, long clen)
		throws IOException, DMLRuntimeException 
	{
		//allocate output frame block
		ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNames(names, clen);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);
	
		//core read (sequential/parallel) 
		InputStreamInputFormat informat = new InputStreamInputFormat(is);
		InputSplit split = informat.getSplits(null, 1)[0];
		readCSVFrameFromInputSplit(split, informat, null, ret, schema, names, rlen, clen, 0, true);
		
		return ret;
	}

	protected void readCSVFrameFromHDFS( Path path, JobConf job, FileSystem fs, 
			FrameBlock dest, ValueType[] schema, String[] names, long rlen, long clen) 
		throws IOException
	{
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);
		for( int i=0, rpos=0; i<splits.length; i++ )
			rpos = readCSVFrameFromInputSplit(splits[i], informat,
				job, dest, schema, names, rlen, clen, rpos, i==0);
	}

	protected final int readCSVFrameFromInputSplit( InputSplit split, InputFormat<LongWritable,Text> informat, JobConf job, 
			FrameBlock dest, ValueType[] schema, String[] names, long rlen, long clen, int rl, boolean first)
		throws IOException
	{
		boolean hasHeader = _props.hasHeader();
		boolean isFill = _props.isFill();
		double dfillValue = _props.getFillValue();
		String sfillValue = String.valueOf(_props.getFillValue());
		String delim = _props.getDelim();
		
		//create record reader
		RecordReader<LongWritable, Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
		LongWritable key = new LongWritable();
		Text value = new Text();
		int row = rl;
		int col = -1;
		
		//handle header if existing
		if(first && hasHeader ) {
			reader.next(key, value); //read header
			dest.setColumnNames(value.toString().split(delim));
		}
			
		// Read the data
		boolean emptyValuesFound = false;
		try
		{
			while( reader.next(key, value) ) //foreach line
			{
				String cellStr = value.toString().trim();
				emptyValuesFound = false; col = 0;
				String[] parts = IOUtilFunctions.splitCSV(cellStr, delim);
				
				//parse frame meta data (missing values / num distinct)
				if( parts[0].equals(TfUtils.TXMTD_MVPREFIX) || parts[0].equals(TfUtils.TXMTD_NDPREFIX) ) {
					if( parts[0].equals(TfUtils.TXMTD_MVPREFIX) )
						for( int j=0; j<dest.getNumColumns(); j++ )
							dest.getColumnMetadata(j).setMvValue(parts[j+1]);
					else if( parts[0].equals(TfUtils.TXMTD_NDPREFIX) )
						for( int j=0; j<dest.getNumColumns(); j++ )
							dest.getColumnMetadata(j).setNumDistinct(Long.parseLong(parts[j+1]));
					continue;
				}
				
				for( String part : parts ) //foreach cell
				{
					part = part.trim();
					if ( part.isEmpty() ) {
						if( isFill && dfillValue!=0 )
							dest.set(row, col, UtilFunctions.stringToObject(schema[col], sfillValue));
						emptyValuesFound = true;
					}
					else {
						dest.set(row, col, UtilFunctions.stringToObject(schema[col], part));
					}
					col++;
				}
				
				//sanity checks for empty values and number of columns
				IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(cellStr, isFill, emptyValuesFound);
				IOUtilFunctions.checkAndRaiseErrorCSVNumColumns("", cellStr, parts, clen);
				row++;
			}
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
		
		return row;
	}

	protected Pair<Integer,Integer> computeCSVSize( Path path, JobConf job, FileSystem fs) 
		throws IOException 
	{	
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);
		
		//compute number of columns
		int ncol = IOUtilFunctions.countNumColumnsCSV(splits, informat, job, _props.getDelim());
		
		//compute number of rows
		int nrow = 0;
		for( int i=0; i<splits.length; i++ ) 
		{
			RecordReader<LongWritable, Text> reader = informat.getRecordReader(splits[i], job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			
			try
			{
				//ignore header of first split
				if( i==0 && _props.hasHeader() )
					reader.next(key, value);
				
				//count remaining number of rows, ignore meta data
				while ( reader.next(key, value) ) {
					String val = value.toString();
					nrow += ( val.startsWith(TfUtils.TXMTD_MVPREFIX)
						|| val.startsWith(TfUtils.TXMTD_NDPREFIX)) ? 0 : 1; 
				}
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
		}
		
		return new Pair<Integer,Integer>(nrow, ncol);
	}
}

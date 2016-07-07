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
import java.util.List;

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
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.FastStringTokenizer;
import org.apache.sysml.runtime.util.UtilFunctions;

/**
 * Single-threaded frame textcell reader.
 * 
 */
public class FrameReaderTextCell extends FrameReader
{
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
	public final FrameBlock readFrameFromHDFS(String fname, List<ValueType> schema, List<String> names, long rlen, long clen)
		throws IOException, DMLRuntimeException
	{
		//allocate output frame block
		List<ValueType> lschema = createOutputSchema(schema, clen);
		List<String> lnames = createOutputNames(names, clen);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname );
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read (sequential/parallel)
		readTextCellFrameFromHDFS(path, job, fs, ret, lschema, lnames, rlen, clen);
		
		return ret;
	}

	/**
	 * 
	 * @param is
	 * @param rlen
	 * @param clen
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	public final FrameBlock readFrameFromInputStream(InputStream is, long rlen, long clen) 
		throws IOException, DMLRuntimeException {
		return readFrameFromInputStream(is, getDefSchema(clen), getDefColNames(clen), rlen, clen);
	}
	
	/**
	 * 
	 * @param is
	 * @param schema
	 * @param names
	 * @param rlen
	 * @param clen
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	public final FrameBlock readFrameFromInputStream(InputStream is, List<ValueType> schema, List<String> names, long rlen, long clen) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output frame block
		List<ValueType> lschema = createOutputSchema(schema, clen);
		List<String> lnames = createOutputNames(names, clen);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);
	
		//core read 
		readRawTextCellFrameFromInputStream(is, ret, lschema, lnames, rlen, clen);
		
		return ret;
	}
	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param fs
	 * @param dest
	 * @param schema
	 * @param names
	 * @param rlen
	 * @param clen
	 * @throws IOException
	 */
	protected void readTextCellFrameFromHDFS( Path path, JobConf job, FileSystem fs, FrameBlock dest, 
			List<ValueType> schema, List<String> names, long rlen, long clen)
		throws IOException
	{
		if( fs.isDirectory(path) ) {
			FileInputFormat.addInputPath(job, path);
			TextInputFormat informat = new TextInputFormat();
			informat.configure(job);
			InputSplit[] splits = informat.getSplits(job, 1);
			for(InputSplit split: splits)
				readTextCellFrameFromInputSplit(split, informat, job, dest);
		}
		else {
			readRawTextCellFrameFromHDFS(path, job, fs, dest, schema, names, rlen, clen);
		}
	}

	/**
	 * 
	 * @param split
	 * @param dest
	 * @param schema
	 * @param names
	 * @param rlen
	 * @param clen
	 * @throws IOException
	 */
	protected final void readTextCellFrameFromInputSplit( InputSplit split, TextInputFormat informat, JobConf job, FrameBlock dest)
		throws IOException
	{
		List<ValueType> schema = dest.getSchema();
		int rlen = dest.getNumRows();
		int clen = dest.getNumColumns();
		
		//create record reader
		RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
		
		LongWritable key = new LongWritable();
		Text value = new Text();
		FastStringTokenizer st = new FastStringTokenizer(' ');
		int row = -1;
		int col = -1;
		
		try
		{
			while( reader.next(key, value) ) {
				st.reset( value.toString() ); //reinit tokenizer
				row = st.nextInt()-1;
				col = st.nextInt()-1;
				if( row == -3 )
					dest.getColumnMetadata(col).setMvValue(st.nextToken());
				else if( row == -2 )
					dest.getColumnMetadata(col).setNumDistinct(st.nextLong());
				else
					dest.set(row, col, UtilFunctions.stringToObject(schema.get(col), st.nextToken()));
			}
		}
		catch(Exception ex) 
		{
			//post-mortem error handling and bounds checking
			if( row < 0 || row + 1 > rlen || col < 0 || col + 1 > clen ) {
				throw new IOException("Frame cell ["+(row+1)+","+(col+1)+"] " +
									  "out of overall frame range [1:"+rlen+",1:"+clen+"].");
			}
			else {
				throw new IOException( "Unable to read frame in text cell format.", ex );
			}
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}		
	}

	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param fs
	 * @param dest
	 * @param schema
	 * @param names
	 * @param rlen
	 * @param clen
	 * @return
	 * @throws IOException
	 */
	protected final void readRawTextCellFrameFromHDFS( Path path, JobConf job, FileSystem fs, FrameBlock dest, 
			List<ValueType> schema, List<String> names, long rlen, long clen)
		throws IOException
	{
		//create input stream for path
		InputStream inputStream = fs.open(path);
		
		//actual read
		readRawTextCellFrameFromInputStream(inputStream, dest, schema, names, rlen, clen);
	}
	
	/**
	 * 
	 * @param is
	 * @param dest
	 * @param schema
	 * @param names
	 * @param rlen
	 * @param clen
	 * @return
	 * @throws IOException
	 */
	protected final void readRawTextCellFrameFromInputStream( InputStream is, FrameBlock dest, List<ValueType> schema, List<String> names, long rlen, long clen)
		throws IOException
	{
		//create buffered reader
		BufferedReader br = new BufferedReader(new InputStreamReader( is ));	
		
		String value = null;
		FastStringTokenizer st = new FastStringTokenizer(' ');
		int row = -1;
		int col = -1;
		
		try
		{			
			while( (value=br.readLine())!=null ) {
				st.reset( value ); //reinit tokenizer
				row = st.nextInt()-1;
				col = st.nextInt()-1;
				if( row == -3 )
					dest.getColumnMetadata(col).setMvValue(st.nextToken());
				else if (row == -2)
					dest.getColumnMetadata(col).setNumDistinct(st.nextLong());
				else
					dest.set(row, col, UtilFunctions.stringToObject(schema.get(col), st.nextToken()));
			}
		}
		catch(Exception ex)
		{
			//post-mortem error handling and bounds checking
			if( row < 0 || row + 1 > rlen || col < 0 || col + 1 > clen ) {
				throw new IOException("Frame cell ["+(row+1)+","+(col+1)+"] " +
									  "out of overall frame range [1:"+rlen+",1:"+clen+"].", ex);
			}
			else {
				throw new IOException( "Unable to read frame in raw text cell format.", ex );
			}
		}
		finally {
			IOUtilFunctions.closeSilently(br);
		}
	}
}

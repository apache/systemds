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
	public FrameBlock readFrameFromHDFS(String fname, List<ValueType> schema, List<String> names,
			long rlen, long clen)
			throws IOException, DMLRuntimeException
	{
		//allocate output frame block
		FrameBlock ret = createOutputFrameBlock(schema, names, rlen);
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname );
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		if( fs.isDirectory(path) )
			readTextCellFrameFromHDFS(path, job, ret, schema, names, rlen, clen);
		else
			readRawTextCellFrameFromHDFS(path, job, fs, ret, schema, names, rlen, clen);
		
		return ret;
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
	public FrameBlock readFrameFromInputStream(InputStream is, List<ValueType> schema, List<String> names, long rlen, long clen) 
			throws IOException, DMLRuntimeException 
	{
		//allocate output frame block
		FrameBlock ret = createOutputFrameBlock(schema, names, rlen);
	
		//core read 
		readRawTextCellFrameFromInputStream(is, ret, schema, names, rlen, clen);
		
		return ret;
	}
	

	/**
	 * 
	 * @param path
	 * @param job
	 * @param dest
	 * @param schema
	 * @param names
	 * @param rlen
	 * @param clen
	 * @return
	 * @throws IOException
	 */
	private void readTextCellFrameFromHDFS( Path path, JobConf job, FrameBlock dest, 
			List<ValueType> schema, List<String> names, long rlen, long clen)
		throws IOException
	{
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
					while( reader.next(key, value) )
					{
						st.reset( value.toString() ); //reinit tokenizer
						row = st.nextInt()-1;
						col = st.nextInt()-1;
						switch( schema.get(col) ) {
							case STRING:  dest.set(row, col, st.nextToken()); break;
							case BOOLEAN: dest.set(row, col, Boolean.valueOf(st.nextToken())); break;
							case INT:     dest.set(row, col, st.nextInt()); break;
							case DOUBLE:  dest.set(row, col, st.nextDouble()); break;
							default: throw new RuntimeException("Unsupported value type: " + schema.get(col));
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
				throw new IOException("Frame cell ["+(row+1)+","+(col+1)+"] " +
									  "out of overall frame range [1:"+rlen+",1:"+clen+"].");
			}
			else
			{
				throw new IOException( "Unable to read frame in text cell format.", ex );
			}
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
	private void readRawTextCellFrameFromHDFS( Path path, JobConf job, FileSystem fs, FrameBlock dest, 
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
	private void readRawTextCellFrameFromInputStream( InputStream is, FrameBlock dest, List<ValueType> schema, List<String> names, long rlen, long clen)
			throws IOException
	{
		BufferedReader br = new BufferedReader(new InputStreamReader( is ));	
		
		String value = null;
		int row = -1;
		int col = -1;
		
		try
		{			
			FastStringTokenizer st = new FastStringTokenizer(' ');
			
			while( (value=br.readLine())!=null )
			{
				st.reset( value ); //reinit tokenizer
				row = st.nextInt()-1;
				col = st.nextInt()-1;	
				switch( schema.get(col) ) {
					case STRING:  dest.set(row, col, st.nextToken()); break;
					case BOOLEAN: dest.set(row, col, Boolean.valueOf(st.nextToken())); break;
					case INT:     dest.set(row, col, st.nextInt()); break;
					case DOUBLE:  dest.set(row, col, st.nextDouble()); break;
					default: throw new RuntimeException("Unsupported value type: " + schema.get(col));
				}
			}
		}
		catch(Exception ex)
		{
			//post-mortem error handling and bounds checking
			if( row < 0 || row + 1 > rlen || col < 0 || col + 1 > clen ) 
			{
				throw new IOException("Frame cell ["+(row+1)+","+(col+1)+"] " +
									  "out of overall frame range [1:"+rlen+",1:"+clen+"].", ex);
			}
			else
			{
				throw new IOException( "Unable to read frame in raw text cell format.", ex );
			}
		}
		finally
		{
			IOUtilFunctions.closeSilently(br);
		}
	}

}

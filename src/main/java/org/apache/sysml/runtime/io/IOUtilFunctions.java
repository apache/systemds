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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Comparator;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.sysml.runtime.util.UtilFunctions;

public class IOUtilFunctions 
{
	private static final Log LOG = LogFactory.getLog(UtilFunctions.class.getName());

	/**
	 * 
	 * @param io
	 */
	public static void closeSilently( Closeable io ) {
		try {
			if( io != null )
				io.close();
        } 
		catch (Exception ex) {
           LOG.error("Failed to close IO resource.", ex);
		}
	}

	/**
	 * 
	 * @param rr
	 */
	public static void closeSilently( RecordReader<?,?> rr ) 
	{
		try {
			if( rr != null )
				rr.close();
        } 
		catch (Exception ex) {
           LOG.error("Failed to close record reader.", ex);
		}
	}
	
	/**
	 * 
	 * @param br
	 */
	public static double parseDoubleParallel( String str ) 
	{
		//return FloatingDecimal.parseDouble(str);
		return Double.parseDouble(str);
	}

	/**
	 * 
	 * @param row
	 * @param fill
	 * @param emptyFound
	 * @throws IOException
	 */
	public static void checkAndRaiseErrorCSVEmptyField(String row, boolean fill, boolean emptyFound) 
		throws IOException
	{
		if ( !fill && emptyFound) {
			throw new IOException("Empty fields found in delimited file. "
			+ "Use \"fill\" option to read delimited files with empty fields:" + ((row!=null)?row:""));
		}
	}
	
	/**
	 * 
	 * @param fname
	 * @param line
	 * @param parts
	 * @param ncol
	 * @throws IOException 
	 */
	public static void checkAndRaiseErrorCSVNumColumns(String fname, String line, String[] parts, long ncol) 
		throws IOException
	{
		int realncol = parts.length;
		
		if( realncol != ncol ) {
			throw new IOException("Invalid number of columns (" + realncol + ", expected=" + ncol + ") "
					+ "found in delimited file (" + fname + ") for line: " + line);
		}
	}
	
	/**
	 * Splits a string by a specified delimiter into all tokens, including empty.
	 * NOTE: This method is meant as a faster drop-in replacement of the regular 
	 * string split.
	 * 
	 * @param str
	 * @param delim
	 * @return
	 */
	public static String[] split(String str, String delim)
	{
		//note: split via stringutils faster than precompiled pattern / guava splitter
		
		//split by whole separator required for multi-character delimiters, preserve
		//all tokens required for empty cells and in order to keep cell alignment
		return StringUtils.splitByWholeSeparatorPreserveAllTokens(str, delim);
	}
	
	/**
	 * 
	 * @param input
	 * @return
	 * @throws IOException
	 */
	public static InputStream toInputStream(String input) throws IOException {
		if( input == null ) 
			return null;
		return new ByteArrayInputStream(input.getBytes("UTF-8"));
	}
	
	/**
	 * 
	 * @param input
	 * @return
	 * @throws IOException
	 */
	public static String toString(InputStream input) throws IOException {
		if( input == null )
			return null;
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		byte[] buff = new byte[LocalFileUtils.BUFFER_SIZE];
		for( int len=0; (len=input.read(buff))!=-1; )
			bos.write(buff, 0, len);
		input.close();		
		return bos.toString("UTF-8");
	}

	/**
	 * 
	 * @param splits
	 * @return
	 */
	public static InputSplit[] sortInputSplits(InputSplit[] splits) {
		if (splits[0] instanceof FileSplit) {
			// The splits do not always arrive in order by file name.
			// Sort the splits lexicographically by path so that the header will
			// be in the first split.
			// Note that we're assuming that the splits come in order by offset
			Arrays.sort(splits, new Comparator<InputSplit>() {
				@Override
				public int compare(InputSplit o1, InputSplit o2) {
					Path p1 = ((FileSplit) o1).getPath();
					Path p2 = ((FileSplit) o2).getPath();
					return p1.toString().compareTo(p2.toString());
				}
			});
		}		
		return splits;
	}
}

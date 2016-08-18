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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.sysml.runtime.util.UtilFunctions;

public class IOUtilFunctions 
{
	private static final Log LOG = LogFactory.getLog(UtilFunctions.class.getName());

	private static final char CSV_QUOTE_CHAR = '"';
	
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
		//split by whole separator required for multi-character delimiters, preserve
		//all tokens required for empty cells and in order to keep cell alignment
		return StringUtils.splitByWholeSeparatorPreserveAllTokens(str, delim);
	}
	
	/**
	 * Splits a string by a specified delimiter into all tokens, including empty
	 * while respecting the rules for quotes and escapes defined in RFC4180.
	 * 
	 * NOTE: use StringEscapeUtils.unescapeCsv(tmp) if needed afterwards.
	 * 
	 * @param str
	 * @param delim
	 * @return
	 */
	public static String[] splitCSV(String str, String delim)
	{
		// check for empty input
		if( str == null || str.isEmpty() )
			return new String[]{""};
		
		// scan string and create individual tokens
		ArrayList<String> tokens = new ArrayList<String>();
		int from = 0, to = 0; 
		int len = str.length();
		while( from < len  ) { // for all tokens
			if( str.charAt(from) == CSV_QUOTE_CHAR ) {
				to = str.indexOf(CSV_QUOTE_CHAR, from+1);
				// handle escaped inner quotes, e.g. "aa""a"
				while( to+1 < len && str.charAt(to+1)==CSV_QUOTE_CHAR )
					to = str.indexOf(CSV_QUOTE_CHAR, to+2); // to + ""
				to += 1; // last "
			}
			else if(str.regionMatches(from, delim, 0, delim.length())) {
				to = from; // empty string
			}
			else { // default: unquoted non-empty
				to = str.indexOf(delim, from+1);
			}
			
			// slice out token and advance position
			to = (to >= 0) ? to : len;
			tokens.add(str.substring(from, to));
			from = to + delim.length();
		}
		
		// handle empty string at end
		if( from == len )
			tokens.add("");
			
		// return tokens
		return tokens.toArray(new String[0]);
	}
	
	/**
	 * Counts the number of tokens defined by the given delimiter, respecting 
	 * the rules for quotes and escapes defined in RFC4180.
	 * 
	 * @param str
	 * @param delim
	 * @return
	 */
	public static int countTokensCSV(String str, String delim)
	{
		// check for empty input
		if( str == null || str.isEmpty() )
			return 1;
		
		// scan string and compute num tokens
		int numTokens = 0;
		int from = 0, to = 0; 
		int len = str.length();
		while( from < len  ) { // for all tokens
			if( str.charAt(from) == CSV_QUOTE_CHAR ) {
				to = str.indexOf(CSV_QUOTE_CHAR, from+1);
				// handle escaped inner quotes, e.g. "aa""a"
				while( to+1 < len && str.charAt(to+1)==CSV_QUOTE_CHAR ) 
					to = str.indexOf(CSV_QUOTE_CHAR, to+2); // to + ""
				to += 1; // last "
			}
			else if(str.regionMatches(from, delim, 0, delim.length())) {
				to = from; // empty string
			}
			else { // default: unquoted non-empty
				to = str.indexOf(delim, from+1);
			}
			
			//increase counter and advance position
			to = (to >= 0) ? to : len;
			from = to + delim.length();
			numTokens++;
		}
		
		// handle empty string at end
		if( from == len )
			numTokens++;
		
		// return number of tokens
		return numTokens;
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
	
	/**
	 * Counts the number of columns in a given collection of csv file splits. This primitive aborts 
	 * if a row with more than 0 columns is found and hence is robust against empty file splits etc.
	 * 
	 * @param splits
	 * @param informat
	 * @param job
	 * @param delim
	 * @return
	 * @throws IOException 
	 */
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static int countNumColumnsCSV(InputSplit[] splits, InputFormat informat, JobConf job, String delim ) 
		throws IOException 
	{
		LongWritable key = new LongWritable();
		Text value = new Text();
		int ncol = -1;
		for( int i=0; i<splits.length && ncol<=0; i++ ) {
			RecordReader<LongWritable, Text> reader = 
					informat.getRecordReader(splits[i], job, Reporter.NULL);
			try {
				if( reader.next(key, value) ) {
					String row = value.toString().trim();
					if( row.startsWith(TfUtils.TXMTD_MVPREFIX) )
						reader.next(key, value);
					if( row.startsWith(TfUtils.TXMTD_NDPREFIX) )
						reader.next(key, value);
					if( !row.isEmpty() )
						ncol = IOUtilFunctions.countTokensCSV(row, delim);
				}
			}
			finally {
				closeSilently(reader);	
			}
		}
		return ncol;
	}

	public static void deleteCrcFilesFromLocalFileSystem(FileSystem fs, String fname) throws IOException {
		if (fs instanceof LocalFileSystem) {
			Path fnameCrc = new Path("." + fname + ".crc");
			fs.delete(fnameCrc, false);
			Path fnameMtdCrc = new Path("." + fname + ".mtd.crc");
			fs.delete(fnameMtdCrc, false);
		}
	}
}

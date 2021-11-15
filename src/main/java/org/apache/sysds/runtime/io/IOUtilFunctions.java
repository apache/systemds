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

package org.apache.sysds.runtime.io;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import org.apache.commons.io.input.ReaderInputStream;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.util.LocalFileUtils;
import org.apache.sysds.runtime.util.UtilFunctions;

public class IOUtilFunctions 
{
	private static final Log LOG = LogFactory.getLog(UtilFunctions.class.getName());

	public static final PathFilter hiddenFileFilter = new PathFilter(){
		@Override
		public boolean accept(Path p){
			String name = p.getName(); 
			return !name.startsWith("_") && !name.startsWith("."); 
		}
	};
	
	//for empty text lines we use 0-0 despite for 1-based indexing in order
	//to allow matrices with zero rows and columns (consistent with R)
	public static final String EMPTY_TEXT_LINE = "0 0 0\n";
	private static final char CSV_QUOTE_CHAR = '"';
	public static final String LIBSVM_DELIM = " ";
	public static final String LIBSVM_INDEX_DELIM = ":";
	
	public static FileSystem getFileSystem(String fname) throws IOException {
		return getFileSystem(new Path(fname),
			ConfigurationManager.getCachedJobConf());
	}
	
	public static FileSystem getFileSystem(Path fname) throws IOException {
		return getFileSystem(fname, 
			ConfigurationManager.getCachedJobConf());
	}
	
	public static FileSystem getFileSystem(Configuration conf) throws IOException {
		try{
			return FileSystem.get(conf);
		} catch(NoClassDefFoundError err) {
			throw new IOException(err.getMessage(), err);
		}
	}
	
	public static FileSystem getFileSystem(Path fname, Configuration conf) throws IOException {
		try {
			return FileSystem.get(fname.toUri(), conf);
		} catch(NoClassDefFoundError err) {
			throw new IOException(err.getMessage(), err);
		}
	}
	
	public static boolean isSameFileScheme(Path path1, Path path2) {
		if( path1 == null || path2 == null || path1.toUri() == null || path2.toUri() == null)
			return false;
		String scheme1 = path1.toUri().getScheme();
		String scheme2 = path2.toUri().getScheme();
		return (scheme1 == null && scheme2 == null)
			|| (scheme1 != null && scheme1.equals(scheme2));
	}
	
	public static boolean isObjectStoreFileScheme(Path path) {
		if( path == null || path.toUri() == null || path.toUri().getScheme() == null )
			return false;
		String scheme = path.toUri().getScheme();
		//capture multiple alternatives s3, s3n, s3a, swift, swift2d
		return scheme.startsWith("s3") || scheme.startsWith("swift");
	}
	
	public static String getPartFileName(int pos) {
		return String.format("0-m-%05d", pos);
	}
	
	public static void closeSilently( Closeable io ) {
		try {
			if( io != null )
				io.close();
		}
		catch (Exception ex) {
			LOG.error("Failed to close IO resource.", ex);
		}
	}

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
	
	public static void checkAndRaiseErrorCSVEmptyField(String row, boolean fill, boolean emptyFound) 
		throws IOException
	{
		if ( !fill && emptyFound) {
			throw new IOException("Empty fields found in delimited file. "
			+ "Use \"fill\" option to read delimited files with empty fields:" + ((row!=null)?row:""));
		}
	}

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
	 * @param str string to split
	 * @param delim delimiter
	 * @return string array
	 */
	public static String[] split(String str, String delim)
	{
		//split by whole separator required for multi-character delimiters, preserve
		//all tokens required for empty cells and in order to keep cell alignment
	
		return StringUtils.splitByWholeSeparatorPreserveAllTokens(str, delim);
	}
	
	/**
	 * Splits a string by a specified delimiter into all tokens, including empty
	 * while respecting the rules for quotes and escapes defined in RFC4180,
	 * with robustness for various special cases.
	 * 
	 * @param str string to split
	 * @param delim delimiter
	 * @return string array of tokens
	 */
	public static String[] splitCSV(String str, String delim)
	{
		// check for empty input
		if( str == null || str.isEmpty() )
			return new String[]{""};
		
		// scan string and create individual tokens
		ArrayList<String> tokens = new ArrayList<>();
		int from = 0, to = 0; 
		int len = str.length();
		int dlen = delim.length();
		while( from < len  ) { // for all tokens
			if( str.charAt(from) == CSV_QUOTE_CHAR 
				&& str.indexOf(CSV_QUOTE_CHAR, from+1) > 0 ) {
				to = str.indexOf(CSV_QUOTE_CHAR, from+1);
				// handle escaped inner quotes, e.g. "aa""a"
				while( to+1 < len && str.charAt(to+1)==CSV_QUOTE_CHAR )
					to = str.indexOf(CSV_QUOTE_CHAR, to+2); // to + ""
				to += 1; // last "
				// handle remaining non-quoted characters "aa"a 
				if( to<len-1 && !str.regionMatches(to, delim, 0, dlen) )
					to = str.indexOf(delim, to+1);
			}
			else if( str.regionMatches(from, delim, 0, dlen) ) {
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
	 * Splits a string by a specified delimiter into all tokens, including empty
	 * while respecting the rules for quotes and escapes defined in RFC4180,
	 * with robustness for various special cases.
	 * 
	 * @param str string to split
	 * @param delim delimiter
	 * @param tokens array for tokens, length needs to match the number of tokens
	 * @param naStrings the strings to map to null value.
	 * @return string array of tokens
	 */
	public static String[] splitCSV(String str, String delim, String[] tokens, Set<String> naStrings)
	{
		// check for empty input
		if( str == null || str.isEmpty() )
			return new String[]{""};
		
		// scan string and create individual tokens
		int from = 0, to = 0; 
		int len = str.length();
		int dlen = delim.length();
		String curString;
		int pos = 0;
		while( from < len  ) { // for all tokens
			if( str.charAt(from) == CSV_QUOTE_CHAR
				&& str.indexOf(CSV_QUOTE_CHAR, from+1) > 0 ) {
				to = str.indexOf(CSV_QUOTE_CHAR, from+1);
				// handle escaped inner quotes, e.g. "aa""a"
				while( to+1 < len && str.charAt(to+1)==CSV_QUOTE_CHAR )
					to = str.indexOf(CSV_QUOTE_CHAR, to+2); // to + ""
				to += 1; // last "
				// handle remaining non-quoted characters "aa"a 
				if( to<len-1 && !str.regionMatches(to, delim, 0, dlen) )
					to = str.indexOf(delim, to+1);
			}
			else if( str.regionMatches(from, delim, 0, dlen) ) {
				to = from; // empty string
			}
			else { // default: unquoted non-empty
				to = str.indexOf(delim, from+1);
			}
			
			// slice out token and advance position
			to = (to >= 0) ? to : len;
			curString = str.substring(from, to);
			tokens[pos++] = naStrings!= null ? ((naStrings.contains(curString)) ? null: curString): curString;
			from = to + delim.length();
		}
		
		// handle empty string at end
		if( from == len )
			tokens[pos] = "";
			
		// return tokens
		return tokens;
	}
	
	/**
	 * Counts the number of tokens defined by the given delimiter, respecting 
	 * the rules for quotes and escapes defined in RFC4180,
	 * with robustness for various special cases.
	 * 
	 * @param str string to split
	 * @param delim delimiter
	 * @return number of tokens split by the given delimiter
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
		int dlen = delim.length();
		while( from < len  ) { // for all tokens
			if( str.charAt(from) == CSV_QUOTE_CHAR
				&& str.indexOf(CSV_QUOTE_CHAR, from+1) > 0 ) {
				to = str.indexOf(CSV_QUOTE_CHAR, from+1);
				// handle escaped inner quotes, e.g. "aa""a"
				while( to+1 < len && str.charAt(to+1)==CSV_QUOTE_CHAR ) 
					to = str.indexOf(CSV_QUOTE_CHAR, to+2); // to + ""
				to += 1; // last "
				// handle remaining non-quoted characters "aa"a 
				if( to<len-1 && !str.regionMatches(to, delim, 0, dlen) )
					to = str.indexOf(delim, to+1);
			}
			else if( str.regionMatches(from, delim, 0, dlen) ) {
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
	
	public static String[] splitByFirst(String str, String delim) {
		int pos = str.indexOf(delim);
		return new String[]{str.substring(0, pos),
			str.substring(pos+1, str.length())};
	}
	
	public static FileFormatPropertiesMM readAndParseMatrixMarketHeader(String filename) throws DMLRuntimeException {
		String[] header = readMatrixMarketHeader(filename);
		return FileFormatPropertiesMM.parse(header[0]);
	}
	
	public static String[] readMatrixMarketHeader(String filename) {
		String[] retVal = new String[2];
		retVal[0] = new String("");
		retVal[1] = new String("");
		boolean exists = false;
		
		try {
			Path path = new Path(filename);
			FileSystem fs = IOUtilFunctions.getFileSystem(path);
			exists = fs.exists(path);
			boolean getFileStatusIsDir = fs.getFileStatus(path).isDirectory();
			if (exists && getFileStatusIsDir) {
				throw new DMLRuntimeException("MatrixMarket files as directories not supported");
			}
			else if (exists) {
				try( BufferedReader in = new BufferedReader(new InputStreamReader(fs.open(path))) ) {
					retVal[0] = in.readLine();
					// skip all commented lines
					do {
						retVal[1] = in.readLine();
					} while ( retVal[1].charAt(0) == '%' );
					if ( !retVal[0].startsWith("%%") ) {
						throw new DMLRuntimeException("MatrixMarket files must begin with a header line.");
					}
				}
			}
			else {
				throw new DMLRuntimeException("Could not find the file: " + filename);
			}
		}
		catch (IOException e){
			throw new DMLRuntimeException(e);
		}
		return retVal;
	}
	
	/**
	 * Returns the number of non-zero entries but avoids the expensive 
	 * string to double parsing. This function is guaranteed to never
	 * underestimate.
	 * 
	 * @param cols string array
	 * @return number of non-zeros
	 */
	public static int countNnz(String[] cols) {
		return countNnz(cols, 0, cols.length);
	}
	
	/**
	 * Returns the number of non-zero entries but avoids the expensive 
	 * string to double parsing. This function is guaranteed to never
	 * underestimate.
	 * 
	 * @param cols string array
	 * @param pos starting array index
	 * @param len ending array index
	 * @return number of non-zeros
	 */
	public static int countNnz(String[] cols, int pos, int len) {
		int lnnz = 0;
		for( int i=pos; i<pos+len; i++ ) {
			String col = cols[i];
			lnnz += (!col.isEmpty() && !col.equals("0") 
					&& !col.equals("0.0")) ? 1 : 0;
		}
		return lnnz;
	}
	
	/**
	 * Returns the serialized size in bytes of the given string value,
	 * following the modified UTF-8 specification as used by Java's
	 * DataInput/DataOutput.
	 * 
	 * see java docs: docs/api/java/io/DataInput.html#modified-utf-8
	 * 
	 * @param value string value
	 * @return string size for modified UTF-8 specification
	 */
	public static int getUTFSize(String value) {
		if( value == null )
			return 2;
		//size in modified UTF-8 as used by DataInput/DataOutput
		int size = 2; //length in bytes
		for (int i = 0; i < value.length(); i++) {
			char c = value.charAt(i);
			size += ( c>=0x0001 && c<=0x007F) ? 1 :
				(c >= 0x0800) ? 3 : 2;
		}
		return size;
	}

	public static InputStream toInputStream(String input) {
		if( input == null ) 
			return null;
		return new ReaderInputStream(new StringReader(input), "UTF-8");
	}

	public static String toString(InputStream input) throws IOException {
		if( input == null )
			return null;
		try {
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			byte[] buff = new byte[LocalFileUtils.BUFFER_SIZE];
			for( int len=0; (len=input.read(buff))!=-1; )
				bos.write(buff, 0, len);
			return bos.toString("UTF-8");
		}
		finally {
			IOUtilFunctions.closeSilently(input);
		}
	}

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
	 * @param splits input splits
	 * @param informat input format
	 * @param job job configruation
	 * @param delim delimiter
	 * @return the number of columns in the collection of csv file splits
	 * @throws IOException if IOException occurs
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
					boolean hasValue = true;
					if( value.toString().startsWith(TfUtils.TXMTD_MVPREFIX) )
						hasValue = reader.next(key, value);
					if( value.toString().startsWith(TfUtils.TXMTD_NDPREFIX) )
						hasValue = reader.next(key, value);
					String row = value.toString().trim();
					if( hasValue && !row.isEmpty() ) {
						ncol = IOUtilFunctions.countTokensCSV(row, delim);
					}
				}
			}
			finally {
				closeSilently(reader);	
			}
		}
		return ncol;
	}

	public static Path[] getSequenceFilePaths( FileSystem fs, Path file ) 
		throws IOException
	{
		Path[] ret = null;
		
		//Note on object stores: Since the object store file system implementations 
		//only emulate a file system, the directory of a multi-part file does not
		//exist physically and hence the isDirectory call returns false. Furthermore,
		//listStatus call returns all files with the given directory as prefix, which
		//includes the mtd file which needs to be ignored accordingly.
		
		if( fs.getFileStatus(file).isDirectory() 
			|| IOUtilFunctions.isObjectStoreFileScheme(file) )
		{
			LinkedList<Path> tmp = new LinkedList<>();
			FileStatus[] dStatus = fs.listStatus(file);
			for( FileStatus fdStatus : dStatus )
				if( !fdStatus.getPath().getName().startsWith("_") //skip internal files
					&& !fdStatus.getPath().toString().equals(file.toString()+".mtd") ) //mtd file
					tmp.add(fdStatus.getPath());
			ret = tmp.toArray(new Path[0]);
		}
		else {
			ret = new Path[]{ file };
		}
		
		return ret;
	}
	
	public static Path[] getMetadataFilePaths( FileSystem fs, Path file ) 
		throws IOException
	{
		Path[] ret = null;
		if( fs.getFileStatus(file).isDirectory()
			|| IOUtilFunctions.isObjectStoreFileScheme(file) )
		{
			LinkedList<Path> tmp = new LinkedList<>();
			FileStatus[] dStatus = fs.listStatus(file);
			for( FileStatus fdStatus : dStatus )
				if( fdStatus.getPath().toString().endsWith(".mtd") ) //mtd file
					tmp.add(fdStatus.getPath());
			ret = tmp.toArray(new Path[0]);
		}
		else {
			throw new DMLRuntimeException("Unable to read meta data files from directory "+file.toString());
		}
		return ret;
	}
	
	/**
	 * Delete the CRC files from the local file system associated with a
	 * particular file and its metadata file.
	 * 
	 * @param fs
	 *            the file system
	 * @param path
	 *            the path to a file
	 * @throws IOException
	 *             thrown if error occurred attempting to delete crc files
	 */
	public static void deleteCrcFilesFromLocalFileSystem(FileSystem fs, Path path) throws IOException {
		if (fs instanceof LocalFileSystem) {
			Path fnameCrc = new Path(path.getParent(), "." + path.getName() + ".crc");
			fs.delete(fnameCrc, false);
			Path fnameMtdCrc = new Path(path.getParent(), "." + path.getName() + ".mtd.crc");
			fs.delete(fnameMtdCrc, false);
		}
	}
	
	public static int baToShort( byte[] ba, final int off ) {
		//shift and add 2 bytes into single int
		return ((ba[off+0] & 0xFF) << 8)
			+  ((ba[off+1] & 0xFF) << 0);
	}

	public static int baToInt( byte[] ba, final int off ) {
		//shift and add 4 bytes into single int
		return ((ba[off+0] & 0xFF) << 24)
			+  ((ba[off+1] & 0xFF) << 16)
			+  ((ba[off+2] & 0xFF) <<  8)
			+  ((ba[off+3] & 0xFF) <<  0);
	}

	public static long baToLong( byte[] ba, final int off ) {
		//shift and add 8 bytes into single long
		return ((long)(ba[off+0] & 0xFF) << 56)
			+  ((long)(ba[off+1] & 0xFF) << 48)
			+  ((long)(ba[off+2] & 0xFF) << 40)
			+  ((long)(ba[off+3] & 0xFF) << 32)
			+  ((long)(ba[off+4] & 0xFF) << 24)
			+  ((long)(ba[off+5] & 0xFF) << 16)
			+  ((long)(ba[off+6] & 0xFF) <<  8)
			+  ((long)(ba[off+7] & 0xFF) <<  0);
	}
	
	public static void shortToBa( final int val, byte[] ba, final int off ) {
		//shift and mask out 2 bytes
		ba[ off+0 ] = (byte)((val >>>  8) & 0xFF);
		ba[ off+1 ] = (byte)((val >>>  0) & 0xFF);
	}

	public static void intToBa( final int val, byte[] ba, final int off ) {
		//shift and mask out 4 bytes
		ba[ off+0 ] = (byte)((val >>> 24) & 0xFF);
		ba[ off+1 ] = (byte)((val >>> 16) & 0xFF);
		ba[ off+2 ] = (byte)((val >>>  8) & 0xFF);
		ba[ off+3 ] = (byte)((val >>>  0) & 0xFF);
	}

	public static void longToBa( final long val, byte[] ba, final int off ) {
		//shift and mask out 8 bytes
		ba[ off+0 ] = (byte)((val >>> 56) & 0xFF);
		ba[ off+1 ] = (byte)((val >>> 48) & 0xFF);
		ba[ off+2 ] = (byte)((val >>> 40) & 0xFF);
		ba[ off+3 ] = (byte)((val >>> 32) & 0xFF);
		ba[ off+4 ] = (byte)((val >>> 24) & 0xFF);
		ba[ off+5 ] = (byte)((val >>> 16) & 0xFF);
		ba[ off+6 ] = (byte)((val >>>  8) & 0xFF);
		ba[ off+7 ] = (byte)((val >>>  0) & 0xFF);
	}
	
	public static byte[] getBytes(ByteBuffer buff) {
		int len = buff.limit();
		if( buff.hasArray() )
			return Arrays.copyOf(buff.array(), len);
		byte[] ret = new byte[len];
		buff.get(ret, buff.position(), len);
		return ret;
	}
	
	public static <T> T get(Future<T> in) {
		try {
			return in.get();
		} 
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}
	
	public static class CountRowsTask implements Callable<Long> {
		private final InputSplit _split;
		private final TextInputFormat _inputFormat;
		private final JobConf _jobConf;
		private final boolean _hasHeader;

		public CountRowsTask(InputSplit split, TextInputFormat inputFormat, JobConf jobConf) {
			this(split, inputFormat, jobConf, false);
		}
		
		public CountRowsTask(InputSplit split, TextInputFormat inputFormat, JobConf jobConf, boolean header){
			_split = split;
			_inputFormat = inputFormat;
			_jobConf = jobConf;
			_hasHeader = header;
		}

		@Override
		public Long call() throws Exception {
			RecordReader<LongWritable, Text> reader = _inputFormat.getRecordReader(_split, _jobConf, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			long nrows = 0;

			try{
				// count rows from the first non-header row
				if (_hasHeader)
					reader.next(key, value);
				while (reader.next(key, value))
					nrows++;
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
			return nrows;
		}
	}
}

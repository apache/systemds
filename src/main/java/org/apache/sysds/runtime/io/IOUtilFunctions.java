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
import java.io.File;
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
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.BZip2Codec;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.io.compress.DeflateCodec;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.io.compress.Lz4Codec;
import org.apache.hadoop.io.compress.PassthroughCodec;
import org.apache.hadoop.io.compress.SnappyCodec;
import org.apache.hadoop.io.compress.ZStandardCodec;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ArrayWrapper;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixCell;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.util.LocalFileUtils;

import io.airlift.compress.lzo.LzoCodec;
import io.airlift.compress.lzo.LzopCodec;

public class IOUtilFunctions {
	private static final Log LOG = LogFactory.getLog(IOUtilFunctions.class.getName());

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
	public static final char CSV_QUOTE_CHAR = '"';
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

	public static void checkAndRaiseErrorCSVNumColumns(InputSplit split, String line, String[] parts, long ncol) 
		throws IOException
	{
		int realncol = parts==null ? 1 : parts.length;
		if( realncol != ncol ) {
			checkAndRaiseErrorCSVNumColumns(split.toString(), line, parts, realncol);
		}
	}
	
	public static void checkAndRaiseErrorCSVNumColumns(String src, String line, String[] parts, long ncol) 
		throws IOException
	{
		int realncol = parts.length;
		if( realncol != ncol ) {
			throw new IOException("Invalid number of columns (" + realncol + ", expected=" + ncol + ") "
					+ "found in delimited file (" + src + ") for line: " + line + "\n" + Arrays.toString(parts));
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
	public static String[] splitCSV(String str, String delim){
		if(str == null || str.isEmpty())
			return new String[] {""};

		int from = 0, to = 0;
		final int len = str.length();
		final int delimLen = delim.length();
		final ArrayList<String> tokens = new ArrayList<>();

		while(from < len) { // for all tokens
			to = getTo(str, from, delim, len, delimLen);
			tokens.add(str.substring(from, to));
			from = to + delimLen;
		}

		// handle empty string at end
		if(from == len)
			tokens.add("");

		return tokens.toArray(new String[0]);
	}

	public static String[] splitCSV(String str, String delim, int clen){
		if(str == null || str.isEmpty())
			return new String[] {""};

		int from = 0, to = 0;
		final int len = str.length();
		final int delimLen = delim.length();

		final String[] tokens = new String[clen];
		int c = 0;
		while(from < len) { // for all tokens
			to = getTo(str, from, delim, len, delimLen);
			tokens[c++] = str.substring(from, to);
			from = to + delimLen;
		}

		// handle empty string at end
		if(from == len)
			tokens[c++] = "";

		return tokens;
	}

	/**
	 * Splits a string by a specified delimiter into all tokens, including empty
	 * while respecting the rules for quotes and escapes defined in RFC4180,
	 * with robustness for various special cases.
	 * 
	 * @param str string to split
	 * @param delim delimiter
	 * @param cache cachedReturnArray
	 * @return string array of tokens
	 */
	public static String[] splitCSV(String str, String delim, String[] cache) {
		// check for empty input
		final boolean empty = str == null || str.isEmpty();
		if(cache == null)
			if(empty)
				return new String[] {""};
			else
				return splitCSV(str, delim);
		else if(empty) {
			Arrays.fill(cache, "");
			return cache;
		}
		else
			return splitCSVNonNullWithCache(str, delim, cache);
	}

	private static String[] splitCSVNonNullWithCache(final String str, final String delim, final String[] cache) {
		
		final int len = str.length();
		final int delimLen = delim.length();
		
		if(str.contains("\""))
			return splitCSVNonNullWithCacheWithQuote(str, delim, cache, len, delimLen);
		else if(delimLen == 1)
			return splitCSVNonNullCacheNoQuoteCharDelim(str, delim.charAt(0), cache, len);
		else 
			return splitCSVNonNullCacheNoQuote(str, delim, cache,  len, delimLen);
	}

	private static String[] splitCSVNonNullWithCacheWithQuote(final String str, final String delim,
		final String[] cache, final int len, final int delimLen) {
		int from = 0;
		int id = 0;
		while(from < len) { // for all tokens
			final int to = getTo(str, from, delim, len, delimLen);
			cache[id++] = str.substring(from, to);
			from = to + delimLen;
		}

		if(from == len)
			cache[id] = "";
		return cache;
	}

	private static String[] splitCSVNonNullCacheNoQuote(final String str, final String delim, final String[] cache,final int len, final int delimLen) {
		int from = 0;
		int id = 0;
		
		while(from < len) { // for all tokens
			final int to = getToNoQuote(str, from, delim, len, delimLen);
			cache[id++] = str.substring(from, to);
			from = to + delimLen;
		}
		
		if(from == len)
			cache[id] = "";
		return cache;
	}

	private static String[] splitCSVNonNullCacheNoQuoteCharDelim(final String str, final char delim,
		final String[] cache, final int len) {
		int from = 0;
		int id = 0;
		while(from < len) { // for all tokens
			final int to = getToNoQuoteCharDelim(str, from, delim, len);
			cache[id++] = str.substring(from, to);
			from = to + 1;
		}

		if(from == len)
			cache[id] = "";
		return cache;
	}

	private static boolean isEmptyMatch(final String str, final int from, final String delim, final int dLen,
		final int strLen) {
		// return str.regionMatches(from, delim, 0, dLen); equivalent to 
		for(int i = from, off = 0; off < dLen && i < strLen; i++, off++)
			if(str.charAt(i) != delim.charAt(off))
				return false;
		
		return true;
	}

	/**
	 * Get next index of substring after delim, while the string can contain Quotation marks
	 * 
	 * @param str   The string to get the index from
	 * @param from  The index to start searching from
	 * @param delim The delimiter to find
	 * @param len   The length of the str string argument
	 * @param dLen  The length of the delimiter string
	 * @return The next index.
	 */
	public static int getTo(final String str, final int from, final String delim,
		final int len, final int dLen) {
		final char cq = CSV_QUOTE_CHAR;
		final int fromP1 = from + 1;
		int to;

		if(str.charAt(from) == cq && str.indexOf(cq, fromP1) > 0) {
			to = str.indexOf(cq, fromP1);
			// handle escaped inner quotes, e.g. "aa""a"
			while(to + 1 < len && str.charAt(to + 1) == cq)
				to = str.indexOf(cq, to + 2); // to + ""
			to += 1; // last "
			// handle remaining non-quoted characters "aa"a
			if(to < len - 1 && !str.regionMatches(to, delim, 0, dLen))
				to = str.indexOf(delim, to + 1);
		}
		else if(isEmptyMatch(str, from, delim, dLen, len))
			return to = from; // empty string
		else // default: unquoted non-empty
			to = str.indexOf(delim, fromP1);

		// slice out token and advance position
		return to >= 0 ? to : len;
	}

	/**
	 * Get next index of substring after delim
	 * 
	 * @param str   The string to get the index from
	 * @param from  The index to start searching from
	 * @param delim The delimiter to find
	 * @param len   The length of the str string argument
	 * @param dLen  The length of the delimiter string
	 * @return The next index.
	 */
	private static int getToNoQuote(final String str, final int from, final String delim, final int len,
		final int dLen) {
		
		int to;
		final int fromP1 = from + 1;
		if(isEmptyMatch(str, from, delim, dLen, len))
			return to = from; // empty string
		else // default: unquoted non-empty
			to = str.indexOf(delim, fromP1);

		// slice out token and advance position
		return to >= 0 ? to : len;
		
	}

	private static int getToNoQuoteCharDelim(final String str, final int from, final char delim, final int len){
		for(int i = from; i < len; i++)
			if(str.charAt(i) == delim)
				return i;
		return len;
	}

	public static String trim(String str) {
		final int len = str.length();
		if(len == 0)
			return str;
		return trim(str, len);
	}

	/**
	 * Caller must have a string of at least 1 character length.
	 * 
	 * @param str string to trim
	 * @param len length of string
	 * @return the trimmed string.
	 */
	public static String trim(final String str, final int len) {
		try{
			// short the call to return input if not whitespace in ends.
			if(str.charAt(0) <= ' ' || str.charAt(len -1) <= ' ')
				return str.trim();
			else 
				return str;
		}
		catch(NullPointerException e){
			return null;
		}
		catch(Exception e){
			throw new RuntimeException("failed trimming: " + str + " " + str.length(), e);
		}
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
		else if (naStrings == null)
			return splitCSV(str, delim, tokens);
		
		// scan string and create individual tokens
		final int len = str.length();
		final int dLen = delim.length();
		int from = 0; 
		int pos = 0;
		while( from < len  ) { // for all tokens
			final int to = getTo(str, from, delim, len, dLen);
			final String curString = str.substring(from, to);
			tokens[pos++] = naStrings.contains(curString) ? null : curString;
			from = to + dLen;
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
		final int len = str.length();
		final int dlen = delim.length();
		int numTokens = 0;
		int from = 0; 
		while( from < len  ) { // for all tokens
			int to = getTo(str, from, delim, len, dlen);
			from = to + dlen;
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

	@SuppressWarnings("deprecation")
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

	public static Path[] getSequenceFilePaths(FileSystem fs, Path file) throws IOException {
		Path[] ret = null;
		
		// Note on object stores: Since the object store file system implementations
		// only emulate a file system, the directory of a multi-part file does not
		// exist physically and hence the isDirectory call returns false. Furthermore,
		// listStatus call returns all files with the given directory as prefix, which
		// includes the mtd file which needs to be ignored accordingly.

		if(fs instanceof LocalFileSystem) {
			java.io.File f = new java.io.File(file.toString());
			if(f.isDirectory()){
				java.io.File[] r = new java.io.File(file.toString()).listFiles((a) -> {
					final String s = a.getName();
					return !(s.startsWith("_") || (s.endsWith(".crc")) || s.endsWith(".mtd"));
				});
				ret = new Path[r.length];
				for(int i = 0; i < r.length; i++)
					ret[i] = new Path(r[i].toString());
			}
			else{
				return new Path[]{file};
			}
		}
		else if(fs.getFileStatus(file).isDirectory() || IOUtilFunctions.isObjectStoreFileScheme(file)) {
			LinkedList<Path> tmp = new LinkedList<>();
			FileStatus[] dStatus = fs.listStatus(file);
			for(FileStatus fdStatus : dStatus)
				if(!fdStatus.getPath().getName().startsWith("_") // skip internal files
					&& !fdStatus.getPath().toString().equals(file.toString() + ".mtd")) // mtd file
					tmp.add(fdStatus.getPath());
			ret = tmp.toArray(new Path[0]);
		}
		else {
			ret = new Path[] {file};
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
	
	public static void deleteCrcFilesFromLocalFileSystem( JobConf job, Path path) throws IOException {
		final FileSystem fs = getFileSystem(path,job );
		deleteCrcFilesFromLocalFileSystem(fs, path);
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
			fs.deleteOnExit(new Path(path.getParent(), "." + path.getName() + ".crc"));
			fs.deleteOnExit(new Path(path.getParent(), "." + path.getName() + ".mtd.crc"));
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

	public static boolean isFileCPReadable(String path){
		try{

			JobConf job = ConfigurationManager.getCachedJobConf();
			Path p = new Path(path);
			FileSystem fs = getFileSystem(p,job);
			if(fs instanceof LocalFileSystem){
				// fast java path.
				File f = new File(path);
				return ! f.isDirectory() && f.length() < Integer.MAX_VALUE;
			}
			else{
				FileStatus fstat = fs.getFileStatus(p);
				return !fstat.isDirectory() && fstat.getLen() < Integer.MAX_VALUE;
			}
		}
		catch(Exception e){
			return false;
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

	public static Writer getSeqWriter(Path path, Configuration job, int replication) throws IOException {
		return SequenceFile.createWriter(job, Writer.file(path), Writer.bufferSize(4096),
			Writer.replication((short) (replication > 0 ? replication : 1)),
			Writer.compression(getCompressionEncodingType(), getCompressionCodec()), Writer.keyClass(MatrixIndexes.class),
			Writer.valueClass(MatrixBlock.class));
	}

	public static Writer getSeqWriterFrame(Path path, Configuration job, int replication) throws IOException {
		return SequenceFile.createWriter(job, Writer.file(path), Writer.bufferSize(4096),
			Writer.keyClass(LongWritable.class), Writer.valueClass(FrameBlock.class),
			Writer.compression(getCompressionEncodingType(), getCompressionCodec()),
			Writer.replication((short) (replication > 0 ? replication : 1)));
	}

	public static Writer getSeqWriterArray(Path path, Configuration job, int replication) throws IOException {
		return SequenceFile.createWriter(job, Writer.file(path), Writer.bufferSize(4096),
			Writer.keyClass(LongWritable.class), Writer.valueClass(ArrayWrapper.class),
			Writer.compression(getCompressionEncodingType(), getCompressionCodec()),
			Writer.replication((short) (replication > 0 ? replication : 1)));
	}

	public static Writer getSeqWriterTensor(Path path, Configuration job, int replication) throws IOException {
		return SequenceFile.createWriter(job, Writer.file(path), Writer.bufferSize(4096),
		Writer.replication((short) (replication > 0 ? replication : 1)),
		Writer.compression(getCompressionEncodingType(),getCompressionCodec()), Writer.keyClass(TensorIndexes.class),
		Writer.valueClass(TensorBlock.class));
	}

	public static Writer getSeqWriterCell(Path path, Configuration job, int replication) throws IOException {
		return SequenceFile.createWriter(job, Writer.file(path), Writer.bufferSize(4096),
			Writer.replication((short) (replication > 0 ? replication : 1)),
			Writer.compression(getCompressionEncodingType(), getCompressionCodec()),
			Writer.keyClass(MatrixIndexes.class),
			Writer.valueClass(MatrixCell.class));
	}

	public static SequenceFile.CompressionType getCompressionEncodingType() {
		String v = ConfigurationManager.getDMLConfig().getTextValue(DMLConfig.IO_COMPRESSION_CODEC);
		if(v.equals("none"))
			return SequenceFile.CompressionType.NONE;
		else
			return SequenceFile.CompressionType.RECORD;
	}

	public static CompressionCodec getCompressionCodec() {
		String v = ConfigurationManager.getDMLConfig().getTextValue(DMLConfig.IO_COMPRESSION_CODEC);

		switch(v) {
			case "Lz4":
				return new Lz4Codec();
			case "Lzo":
				return new LzoCodec();
			case "Lzop":
				return new LzopCodec();
			case "Snappy":
				return new SnappyCodec();
			case "BZip2":
				return new BZip2Codec();
			case "deflate":
				return new DeflateCodec();
			case "Gzip":
				return new GzipCodec();
			case "pass":
				return new PassthroughCodec();
			case "Zstd":
				return new ZStandardCodec();
			case "none":
				return null;
			case "default":
			default:
				return new DefaultCodec();
		}

	}

}

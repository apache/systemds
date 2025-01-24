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

import java.io.IOException;
import java.io.InputStream;
import java.util.Set;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.InputStreamInputFormat;

/**
 * Single-threaded frame text csv reader.
 * 
 */
public class FrameReaderTextCSV extends FrameReader {
	protected final FileFormatPropertiesCSV _props;

	public FrameReaderTextCSV(FileFormatPropertiesCSV props) {
		//if unspecified use default properties for robustness
		_props = props != null ? props : new FileFormatPropertiesCSV();
	}

	@Override
	public final FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		// prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		FileInputFormat.addInputPath(job, path);

		// check existence and non-empty file
		checkValidInputFile(fs, path);

		// compute size if necessary
		if(rlen <= 0 || clen <= 0) {
			Pair<Integer, Integer> size = computeCSVSize(path, job, fs);
			rlen = size.getKey();
			clen = size.getValue();
		}

		// allocate output frame block
		ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNames(names, clen);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);

		// core read (sequential/parallel)
		readCSVFrameFromHDFS(path, job, fs, ret, lschema, lnames, rlen, clen);

		return ret;
	}

	@Override
	public FrameBlock readFrameFromInputStream(InputStream is, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException
	{
		// allocate output frame block
		ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNames(names, clen);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);

		// core read (sequential/parallel)
		InputStreamInputFormat informat = new InputStreamInputFormat(is);
		InputSplit split = informat.getSplits(null, 1)[0];
		readCSVFrameFromInputSplit(split, informat, null, ret, schema, names, rlen, clen, 0, true);

		return ret;
	}

	protected void readCSVFrameFromHDFS(Path path, JobConf job, FileSystem fs, FrameBlock dest, ValueType[] schema,
		String[] names, long rlen, long clen) throws IOException
	{
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		if(HDFSTool.isDirectory(fs, path))
			splits = IOUtilFunctions.sortInputSplits(splits);
		for(int i = 0, rpos = 0; i < splits.length; i++)
			rpos = readCSVFrameFromInputSplit(splits[i], informat, job, dest, schema, names, rlen, clen, rpos, i == 0);
	}

	protected final int readCSVFrameFromInputSplit(InputSplit split, InputFormat<LongWritable, Text> informat,
		JobConf job, FrameBlock dest, ValueType[] schema, String[] names, long rlen, long clen, int rl, boolean first)
		throws IOException {
		
		if(rl > rlen) // in case this method is called wrongly
			throw new DMLRuntimeException("Invalid offset");
		// return (int) rlen;
		final boolean hasHeader = _props.hasHeader();
		final boolean isFill = _props.isFill();
		final double dfillValue = _props.getFillValue();
		final String sfillValue = String.valueOf(_props.getFillValue());
		final Set<String> naValues = _props.getNAStrings();
		final String delim = _props.getDelim();
		final CellAssigner f;
		if(naValues != null )
			f = FrameReaderTextCSV::assignCellGeneric;
		else if(isFill && dfillValue != 0)
			f = FrameReaderTextCSV::assignCellFill;
		else 
			f = FrameReaderTextCSV::assignCellNoFill;
		
		final RecordReader<LongWritable, Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
		final LongWritable key = new LongWritable();
		final Text value = new Text();
		

		// handle header if existing
		if(first && hasHeader) {
			reader.next(key, value); // read header
			dest.setColumnNames(value.toString().split(delim));
		}

		// Read the data
		int row = rl;
		try {
			Array<?>[] destA = dest.getColumns();
			while(reader.next(key, value)) // foreach line
			{
				String line = value.toString();
				if(isMetaStart(line)){
					parseMeta(line, delim , dest);
					continue;
				}

				parseLine(line, delim, destA, row, (int) clen, dfillValue, sfillValue, isFill, naValues, f);
				row++;
			}
		}
		catch(Exception e){
			throw new DMLRuntimeException("Failed parsing string: \"" + value +"\"", e);
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}

		return row;
	}

	private static boolean isMetaStart(String s){
		return s.charAt(0) == '#' && s.substring(0, 5).equals("#Meta");

	} 

	private static void parseMeta(String s, String delim, FrameBlock dest){

			String[] parts = IOUtilFunctions.splitCSV(s, delim);

			final boolean mtdP = parts[0].equals(TfUtils.TXMTD_MVPREFIX);
			final boolean mtdx = parts[0].equals(TfUtils.TXMTD_NDPREFIX);

			if(parts.length != dest.getNumColumns() + 1){
				LOG.warn("Invalid metadata ");
				parts = null;
				return;
			}
			else if(mtdP)
				for(int j = 0; j < dest.getNumColumns(); j++)
					dest.getColumnMetadata(j).setMvValue(parts[j + 1]);
			else if(mtdx)
				for(int j = 0; j < dest.getNumColumns(); j++)
					dest.getColumnMetadata(j).setNumDistinct(Long.parseLong(parts[j + 1]));
			parts = null;
		
	}

	private static void parseLine(final String cellStr, final String delim, final Array<?>[] destA, final int row, final int clen, final double dfillValue,
		final String sfillValue, final boolean isFill, final Set<String> naValues,final CellAssigner assigner) {
		try {
			final String trimmed = IOUtilFunctions.trim( cellStr);
			final int len = trimmed.length();
			final int delimLen = delim.length();
			parseLineSpecialized(trimmed, delim, destA, row, dfillValue, sfillValue, isFill, naValues, len, delimLen, assigner);
		}
		catch(Exception e) {
			throw new RuntimeException("failed to parse: " + cellStr, e);
		}
	}

	private static void parseLineSpecialized(String cellStr, String delim, Array<?>[] destA, int row, double dfillValue, String sfillValue,
		boolean isFill, Set<String> naValues, final int len, final int delimLen, final CellAssigner assigner) {
		int from = 0, to = 0, c = 0;
		while(from < len) { // for all tokens
			to = IOUtilFunctions.getTo(cellStr, from, delim, len, delimLen);
			String s = cellStr.substring(from, to);
			assigner.assign(row, destA[c], s, to - from, naValues, isFill, dfillValue, sfillValue);
			c++;
			from = to + delimLen;
		}
	}

	@FunctionalInterface
	private interface CellAssigner{
		void assign(int row, Array<?> dest, String val, int length, Set<String> naValues, boolean isFill,
		double dfillValue, String sfillValue);
	}


	private static void assignCellNoFill(int row, Array<?> dest, String val, int length, Set<String> naValues, boolean isFill,
		double dfillValue, String sfillValue) {
		if(length != 0){
			final String part = IOUtilFunctions.trim(val, length);
			if(part.isEmpty())
				return;
			dest.set(row, part);
		}
	}


	private static void assignCellFill(int row, Array<?> dest, String val, int length, Set<String> naValues, boolean isFill,
		double dfillValue, String sfillValue) {
		if(length == 0){
			dest.set(row, sfillValue);
		} else {
			final String part = IOUtilFunctions.trim(val, length);
			if(part == null || part.isEmpty())
				dest.set(row, sfillValue);
			else
				dest.set(row, part);
		}
	}

	private static void assignCellGeneric(int row, Array<?> dest, String val, int length, Set<String> naValues, boolean isFill,
		double dfillValue, String sfillValue) {
		if(length == 0) {
			if(isFill && dfillValue != 0)
				dest.set(row, sfillValue);
		}
		else {
			final String part = IOUtilFunctions.trim(val, length);
			if(part == null || part.isEmpty() || (naValues != null && naValues.contains(part))) {
				if(isFill && dfillValue != 0)
					dest.set(row, sfillValue);
			}
			else
				dest.set(row, part);
		}
	}

	protected Pair<Integer, Integer> computeCSVSize(Path path, JobConf job, FileSystem fs) throws IOException {
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);

		// compute number of columns
		int ncol = IOUtilFunctions.countNumColumnsCSV(splits, informat, job, _props.getDelim());

		// compute number of rows
		int nrow = 0;
		for(int i = 0; i < splits.length; i++) {
			boolean header = i == 0 && _props.hasHeader();
			nrow += countLinesInSplit(splits[i], informat, job, header);
		}

		return new Pair<>(nrow, ncol);
	}

	protected static long countLinesInSplit(InputSplit split, TextInputFormat inFormat, JobConf job, boolean header)
		throws IOException
	{
		RecordReader<LongWritable, Text> reader = inFormat.getRecordReader(split, job, Reporter.NULL);
		try {
			return countLinesInReader(reader, header);
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
	}

	private static int countLinesInReader(RecordReader<LongWritable, Text> reader, boolean header)
		throws IOException {
		final LongWritable key = new LongWritable();
		final Text value = new Text();

		int nrow = 0;
		// ignore header of first split
		if(header)
			reader.next(key, value);
		while(reader.next(key, value)) {
			// (but only at beginning of individual part files)
			if(nrow < 3){
				String sval = IOUtilFunctions.trim(value.toString());
				boolean containsMTD =
					(sval.startsWith(TfUtils.TXMTD_MVPREFIX)
					|| sval.startsWith(TfUtils.TXMTD_NDPREFIX));
				nrow += containsMTD ? 0 : 1;
			}
			else 
				nrow++;
		}
		return nrow;
	}
}

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
		
		if( rl > rlen) // in case this method is called wrongly
			throw new DMLRuntimeException("Invalid offset");
		// return (int) rlen;
		boolean hasHeader = _props.hasHeader();
		boolean isFill = _props.isFill();
		double dfillValue = _props.getFillValue();
		String sfillValue = String.valueOf(_props.getFillValue());
		Set<String> naValues = _props.getNAStrings();
		String delim = _props.getDelim();

		// create record reader
		RecordReader<LongWritable, Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
		LongWritable key = new LongWritable();
		Text value = new Text();
		int row = rl;

		// handle header if existing
		if(first && hasHeader) {
			reader.next(key, value); // read header
			dest.setColumnNames(value.toString().split(delim));
		}

		// Read the data
		try {
			Array<?>[] destA = dest.getColumns();
			while(reader.next(key, value)) // foreach line
			{
				String cellStr = IOUtilFunctions.trim(value.toString());
				parseLine(cellStr, delim, destA, row, (int) clen, dfillValue, sfillValue, isFill, naValues);
				row++;
			}

		}
		catch(Exception e){
			throw new DMLRuntimeException("Failed parsing string: \"" + value +"\"", e);
		}
		finally {
			// if(pool != null)
			// 	pool.shutdown();
			IOUtilFunctions.closeSilently(reader);
		}

		return row;
	}

	private void parseLine(String cellStr, String delim, Array<?>[] destA , int row,
		 int clen,  double dfillValue, String sfillValue, boolean isFill,
		Set<String> naValues) {
			try{
				int from = 0, to = 0; 
				final int len = cellStr.length();
				final int delimLen = delim.length();
				int c = 0;
				while(from < len) { // for all tokens
					to = IOUtilFunctions.getTo(cellStr, from, delim, len, delimLen);
					assignCellGeneric(row, destA, cellStr.substring(from, to), naValues, isFill, dfillValue, sfillValue,
						false, c);
					c++;
					from = to + delimLen;
				}

			}
			catch(Exception e){
				throw new RuntimeException("failed to parse: " + cellStr, e);
			}
	}

	private boolean assignColumns(int row, int nCol,  Array<?>[] destA, String[] parts, Set<String> naValues,
		boolean isFill, double dfillValue, String sfillValue) {
		if(!isFill && naValues == null)
			return assignColumnsNoFillNoNan(row, nCol, destA, parts);
		else 
			return assignColumnsGeneric(row, nCol, destA, parts, naValues, isFill, dfillValue, sfillValue);
	}

	private boolean assignColumnsGeneric(int row, int nCol,  Array<?>[] destA, String[] parts, Set<String> naValues,
		boolean isFill, double dfillValue, String sfillValue) {
		boolean emptyValuesFound = false;
		for(int col = 0; col < nCol; col++) {
			emptyValuesFound = assignCellGeneric(row, destA, parts[col], naValues, isFill, dfillValue, sfillValue, emptyValuesFound, col);
		}
		return emptyValuesFound;
	}

	private boolean assignColumnsNoFillNoNan(int row, int nCol, Array<?>[] destA, String[] parts){
		boolean emptyValuesFound = false;
		for(int col = 0; col < nCol; col++) {
			emptyValuesFound = assignCellNoNan(row, destA, parts[col], emptyValuesFound, col);
		}
		return emptyValuesFound;
	}


	private static boolean assignCellGeneric(int row, Array<?>[] destA, String val, Set<String> naValues, boolean isFill,
		double dfillValue, String sfillValue, boolean emptyValuesFound, int col) {
		String part = IOUtilFunctions.trim(val);
		if(part == null || part.isEmpty() || (naValues != null && naValues.contains(part))) {
			if(isFill && dfillValue != 0)
				destA[col].set(row, sfillValue);
			emptyValuesFound = true;
		}
		else
			destA[col].set(row, part);
		return emptyValuesFound;
	}

	private static boolean assignCellNoNan(int row, Array<?>[] destA, String val, boolean emptyValuesFound, int col) {
		String part = IOUtilFunctions.trim(val);
		if(part.isEmpty()) 
			emptyValuesFound = true;
		else
			destA[col].set(row, part);
		return emptyValuesFound;
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

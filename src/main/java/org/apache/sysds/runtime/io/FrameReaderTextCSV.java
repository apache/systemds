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
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.transform.TfUtils;
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
		LOG.debug("readFrameFromHDFS csv");
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
		throws IOException, DMLRuntimeException {
		LOG.debug("readFrameFromInputStream csv");
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
		String[] names, long rlen, long clen) throws IOException {
		LOG.debug("readCSVFrameFromHDFS csv");
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
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
		final int nCol = dest.getNumColumns();

		// handle header if existing
		if(first && hasHeader) {
			reader.next(key, value); // read header
			dest.setColumnNames(value.toString().split(delim));
		}

		// Read the data
		try {
			String[] parts = null; // cache array for line reading.
			while(reader.next(key, value)) // foreach line
			{
				String cellStr = value.toString();
				boolean emptyValuesFound = false;
				cellStr = IOUtilFunctions.trim(cellStr);
				parts = IOUtilFunctions.splitCSV(cellStr, delim, parts);
				// sanity checks for empty values and number of columns

				final boolean mtdP = parts[0].equals(TfUtils.TXMTD_MVPREFIX);
				final boolean mtdx = parts[0].equals(TfUtils.TXMTD_NDPREFIX);
				// parse frame meta data (missing values / num distinct)
				if(mtdP || mtdx) {
					parts = IOUtilFunctions.splitCSV(cellStr, delim);
					if(parts.length != dest.getNumColumns() + 1){
						LOG.warn("Invalid metadata ");
						parts = null;
						continue;
					}
					if(mtdP)
						for(int j = 0; j < dest.getNumColumns(); j++)
							dest.getColumnMetadata(j).setMvValue(parts[j + 1]);
					else if(mtdx)
						for(int j = 0; j < dest.getNumColumns(); j++)
							dest.getColumnMetadata(j).setNumDistinct(Long.parseLong(parts[j + 1]));
					parts = null;
					continue;
				}

				for(int col = 0; col < nCol; col++) {
					String part = IOUtilFunctions.trim(parts[col]);
					if(part.isEmpty() || (naValues != null && naValues.contains(part))) {
						if(isFill && dfillValue != 0)
							dest.set(row, col, sfillValue);
						emptyValuesFound = true;
					}
					else
						dest.set(row, col, part);
				}
				IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(cellStr, isFill, emptyValuesFound);
				IOUtilFunctions.checkAndRaiseErrorCSVNumColumns("", cellStr, parts, clen);
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
			nrow += countLinesInReader(splits[i], informat, job, ncol, header);
		}

		return new Pair<>(nrow, ncol);
	}


	protected static int countLinesInReader(InputSplit split, TextInputFormat inFormat, JobConf job, long ncol,
		boolean header) throws IOException {
		RecordReader<LongWritable, Text> reader = inFormat.getRecordReader(split, job, Reporter.NULL);
		try {
			return countLinesInReader(reader, ncol, header);
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
	}

	protected static int countLinesInReader(RecordReader<LongWritable, Text> reader, long ncol, boolean header)
		throws IOException {
		final LongWritable key = new LongWritable();
		final Text value = new Text();

		int nrow = 0;
		try {
			// ignore header of first split
			if(header)
				reader.next(key, value);
			while(reader.next(key, value)) {
				// note the metadata can be located at any row when spark.
				nrow += containsMetaTag(value) ? 0 : 1;
			}
			return nrow;
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
	}

	private final static boolean containsMetaTag(Text val) {
		if(val.charAt(0) == '#')
			return val.find(TfUtils.TXMTD_MVPREFIX) > -1//
				|| val.find(TfUtils.TXMTD_NDPREFIX) > -1;
		else 
			return false;
	}
}

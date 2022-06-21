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
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.util.InputStreamInputFormat;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public class FrameReaderTextAMiner extends FrameReader {
	protected final FileFormatPropertiesAMiner _props;
	private DatasetMetaDataPaper paperMetaData;
	private DatasetMetaDataAuthor authorMetaData;
	private ArrayList<Integer>[] rowIndexs;
	private ArrayList<Integer>[] colBeginIndexs;

	public FrameReaderTextAMiner(FileFormatPropertiesAMiner props) {
		//if unspecified use default properties for robustness
		_props = props;
	}

	@Override public final FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		LOG.debug("readFrameFromHDFS AMiner");
		// prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		FileInputFormat.addInputPath(job, path);

		// check existence and non-empty file
		checkValidInputFile(fs, path);

		ValueType[] lschema = null;
		String[] lnames = null;
		if(_props.getType().equals("paper")) {
			paperMetaData = computeAMinerSizePaper(job);
			rlen = paperMetaData.nrow;
			lschema = paperMetaData.schema;
			lnames = paperMetaData.names;
		}
		else {
			authorMetaData = computeAMinerSizeAuthor(job);
			rlen = authorMetaData.nrow;
			lschema = authorMetaData.schema;
			lnames = authorMetaData.names;
		}
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);
		// core read (sequential/parallel)
		readAMinerFrameFromHDFS(job, ret, lschema);

		return ret;
	}

	@Override public FrameBlock readFrameFromInputStream(InputStream is, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		LOG.debug("readFrameFromInputStream csv");
		ValueType[] lschema = null;
		String[] lnames = null;
		if(_props.getType().equals("paper")) {
			paperMetaData = computeAMinerSizePaper(null);
			rlen = paperMetaData.nrow;
			lschema = paperMetaData.schema;
			lnames = paperMetaData.names;
		}
		else {
			authorMetaData = computeAMinerSizeAuthor(null);
			rlen = authorMetaData.nrow;
			lschema = authorMetaData.schema;
			lnames = authorMetaData.names;
		}
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);

		// core read (sequential/parallel)
		InputStreamInputFormat informat = new InputStreamInputFormat(is);
		InputSplit split = informat.getSplits(null, 1)[0];
		if(_props.getType().equals("paper")) {
			readAMinerPaperFrameFromInputSplit(split, rowIndexs[0], colBeginIndexs[0], informat, null, ret, schema);
		}
		else {
			readAMinerAuthorFrameFromInputSplit(split, rowIndexs[0], informat, null, ret, schema);
		}

		return ret;
	}

	protected void readAMinerFrameFromHDFS(JobConf job, FrameBlock dest, ValueType[] schema) throws IOException {
		LOG.debug("readAMinerFrameFromHDFS csv");
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);
		if(_props.getType().equals("paper")) {
			for(int i = 0; i < splits.length; i++)
				readAMinerPaperFrameFromInputSplit(splits[i], rowIndexs[i], colBeginIndexs[i], informat, job, dest, schema);
		}
		else {
			for(int i = 0; i < splits.length; i++)
				readAMinerAuthorFrameFromInputSplit(splits[i], rowIndexs[i], informat, job, dest, schema);
		}
	}

	//	#index ---- index id of this paper
	//	#*     ---- paper title
	//	#@     ---- authors (separated by semicolons)
	//	#o     ---- affiliations (separated by semicolons, and each affiliaiton corresponds to an author in order)
	//	#t     ---- year
	//	#c     ---- publication venue
	//	#%     ---- the id of references of this paper (there are multiple lines, with each indicating a reference)
	//	#!     ---- abstract
	protected final void readAMinerPaperFrameFromInputSplit(InputSplit split, ArrayList<Integer> rowIndex, ArrayList<Integer> colBeginIndex,
		InputFormat<LongWritable, Text> informat, JobConf job, FrameBlock dest, ValueType[] schema) throws IOException {

		// create record reader
		RecordReader<LongWritable, Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
		LongWritable key = new LongWritable();
		Text value = new Text();
		int row, col;
		int colBegin = 0;
		int index = -1;
		String valStr;
		// Read the data
		try {
			while(reader.next(key, value)) // foreach line
			{
				index++;
				String rowStr = value.toString().trim();
				if(rowStr.length() == 0)
					continue;
				row = rowIndex.get(index);
				colBegin = colBeginIndex.get(index);

				if(rowStr.startsWith("#index ")) {
					col = colBegin;
					valStr = rowStr.substring("#index ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
				else if(rowStr.startsWith("#* ")) {
					col = colBegin + 1;
					valStr = rowStr.substring("#* ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
				else if(rowStr.startsWith("#@ ")) {
					col = colBegin + paperMetaData.authorStartCol;
					valStr = rowStr.substring("#@ ".length()).trim();
					String[] valList = IOUtilFunctions.splitCSV(valStr, ";");
					for(int i = 0; i < valList.length; i++)
						dest.set(row, col + i, UtilFunctions.stringToObject(schema[col], valList[i]));
				}
				else if(rowStr.startsWith("#o ")) {
					col = colBegin + paperMetaData.getAffiliationStartCol();
					valStr = rowStr.substring("#o ".length()).trim();
					String[] valList = IOUtilFunctions.splitCSV(valStr, ";");
					for(int i = 0; i < valList.length; i++)
						dest.set(row, col + i, UtilFunctions.stringToObject(schema[col], valList[i]));
				}
				else if(rowStr.startsWith("#t ")) {
					col = colBegin + 2;
					valStr = rowStr.substring("#t ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
				else if(rowStr.startsWith("#c ")) {
					col = colBegin + 3;
					valStr = rowStr.substring("#c ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
				else if(rowStr.startsWith("#! ")) {
					col = colBegin + 4;
					valStr = rowStr.substring("#! ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
				else if(rowStr.startsWith("#% ")) {
					col = colBegin + paperMetaData.referenceStartCol;
					valStr = rowStr.substring("#! ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
			}
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
	}

	//	#index ---- index id of this author
	//	#n     ---- name  (separated by semicolons)
	//	#a     ---- affiliations  (separated by semicolons)
	//	#pc    ---- the count of published papers of this author
	//	#cn    ---- the total number of citations of this author
	//	#hi    ---- the H-index of this author
	//	#pi    ---- the P-index with equal A-index of this author
	//	#upi   ---- the P-index with unequal A-index of this author
	//	#t     ---- research interests of this author  (separated by semicolons)
	protected final void readAMinerAuthorFrameFromInputSplit(InputSplit split, ArrayList<Integer> rowIndex, InputFormat<LongWritable, Text> informat,
		JobConf job, FrameBlock dest, ValueType[] schema) throws IOException {

		// create record reader
		RecordReader<LongWritable, Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
		LongWritable key = new LongWritable();
		Text value = new Text();
		int row, col;
		int index = -1;
		String valStr;
		try {
			while(reader.next(key, value)) // foreach line
			{
				index++;
				String rowStr = value.toString().trim();
				if(rowStr.length() == 0)
					continue;
				row = rowIndex.get(index);

				if(rowStr.startsWith("#index ")) {
					col = 0;
					valStr = rowStr.substring("#index ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
				else if(rowStr.startsWith("#n ")) {
					col = 1;
					valStr = rowStr.substring("#n ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
				else if(rowStr.startsWith("#a ")) {
					col = authorMetaData.getAffiliationStartCol();
					valStr = rowStr.substring("#a ".length()).trim();
					String[] valList = IOUtilFunctions.splitCSV(valStr, ";");
					for(int i = 0; i < valList.length; i++)
						dest.set(row, col + i, UtilFunctions.stringToObject(schema[col], valList[i]));
				}
				else if(rowStr.startsWith("#t ")) {
					col = authorMetaData.getResearchInterestStartCol();
					valStr = rowStr.substring("#t ".length()).trim();
					String[] valList = IOUtilFunctions.splitCSV(valStr, ";");
					for(int i = 0; i < valList.length; i++)
						dest.set(row, col + i, UtilFunctions.stringToObject(schema[col], valList[i]));
				}

				else if(rowStr.startsWith("#pc ")) {
					col = 2;
					valStr = rowStr.substring("#pc ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
				else if(rowStr.startsWith("#cn ")) {
					col = 3;
					valStr = rowStr.substring("#cn ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
				else if(rowStr.startsWith("#hi ")) {
					col = 4;
					valStr = rowStr.substring("#hi ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
				else if(rowStr.startsWith("#pi ")) {
					col = 5;
					valStr = rowStr.substring("#pi ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
				else if(rowStr.startsWith("#upi ")) {
					col = 6;
					valStr = rowStr.substring("#upi ".length()).trim();
					dest.set(row, col, UtilFunctions.stringToObject(schema[col], valStr));
				}
			}
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
	}

	protected DatasetMetaDataPaper computeAMinerSizePaper(JobConf job) throws IOException {
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);
		this.rowIndexs = new ArrayList[splits.length];
		this.colBeginIndexs = new ArrayList[splits.length];

		LongWritable key = new LongWritable();
		Text value = new Text();
		int ncol = 5;
		int maxAuthors = 0;
		int maxAffiliations = 0;
		int maxReferences = 0;
		int row = -1;
		int lastRefCount = 0;

		for(int i = 0; i < splits.length; i++) {
			RecordReader<LongWritable, Text> reader = informat.getRecordReader(splits[i], job, Reporter.NULL);
			int refCount = 0;
			boolean splitRowFlag = false;
			this.rowIndexs[i] = new ArrayList<>();
			this.colBeginIndexs[i] = new ArrayList<>();
			while(reader.next(key, value)) {
				String raw = value.toString().trim();
				if(raw.startsWith("#index ")) {
					row++;
					if(splitRowFlag)
						maxReferences = Math.max(maxReferences, refCount);
					else
						maxReferences = Math.max(maxReferences, refCount + lastRefCount);

					splitRowFlag = true;
					lastRefCount = refCount;
					refCount = 0;
					this.colBeginIndexs[i].add(0);
				}
				else if(raw.startsWith("#@")) { //authors (separated by semicolons)
					maxAuthors = Math.max(maxAuthors, IOUtilFunctions.countTokensCSV(raw, ";"));
					this.colBeginIndexs[i].add(0);
				}
				else if(raw.startsWith("#o")) { //(separated by semicolons, and each affiliaiton corresponds to an author in order)
					maxAffiliations = Math.max(maxAffiliations, IOUtilFunctions.countTokensCSV(raw, ";"));
					this.colBeginIndexs[i].add(0);
				}
				else if(raw.startsWith("#%")) { // the id of references of this paper (there are multiple lines, with each indicating a reference)

					if(!splitRowFlag)
						this.colBeginIndexs[i].add(refCount + lastRefCount);
					else
						this.colBeginIndexs[i].add(refCount);
					refCount++;
				}
				else
					this.colBeginIndexs[i].add(0);

				this.rowIndexs[i].add(row);
			}
		}
		ncol += maxAuthors + maxAffiliations + maxReferences;

		DatasetMetaDataPaper datasetMetaDataPaper = new DatasetMetaDataPaper(ncol, row + 1, maxAuthors, maxAffiliations);
		return datasetMetaDataPaper;
	}

	protected DatasetMetaDataAuthor computeAMinerSizeAuthor(JobConf job) throws IOException {
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);
		this.rowIndexs = new ArrayList[splits.length];
		this.colBeginIndexs = new ArrayList[splits.length];

		LongWritable key = new LongWritable();
		Text value = new Text();
		int ncol = 7;
		int maxAffiliations = 0;
		int maxResearchInterest = 0;
		int row = -1;

		for(int i = 0; i < splits.length; i++) {
			RecordReader<LongWritable, Text> reader = informat.getRecordReader(splits[i], job, Reporter.NULL);
			this.rowIndexs[i] = new ArrayList<>();
			this.colBeginIndexs[i] = new ArrayList<>();
			while(reader.next(key, value)) {
				String raw = value.toString().trim();
				if(raw.startsWith("#index "))
					row++;
				else if(raw.startsWith("#a")) //affiliations  (separated by semicolons)
					maxAffiliations = Math.max(maxAffiliations, IOUtilFunctions.countTokensCSV(raw, ";"));

				else if(raw.startsWith("#t")) // research interests of this author  (separated by semicolons)
					maxResearchInterest = Math.max(maxResearchInterest, IOUtilFunctions.countTokensCSV(raw, ";"));

				this.rowIndexs[i].add(row);
			}
		}
		ncol += maxAffiliations + maxResearchInterest;

		DatasetMetaDataAuthor datasetMetaDataAuthor = new DatasetMetaDataAuthor(ncol, row + 1, maxAffiliations, maxResearchInterest);
		return datasetMetaDataAuthor;
	}

	protected static abstract class DatasetMetaData {
		protected final int ncol;
		protected final int nrow;
		protected ValueType[] schema;
		protected String[] names;
		private int affiliationStartCol;

		public DatasetMetaData(int ncol, int nrow, int affiliationStartCol) {
			this.ncol = ncol;
			this.nrow = nrow;
			this.names = new String[ncol];
			this.affiliationStartCol = affiliationStartCol;
			for(int i = 0; i < ncol; i++)
				this.names[i] = "col_" + i;
		}

		public String[] getNames() {
			return names;
		}

		public ValueType[] getSchema() {
			return schema;
		}

		public int getAffiliationStartCol() {
			return affiliationStartCol;
		}

		public int getNcol() {
			return ncol;
		}

		public int getNrow() {
			return nrow;
		}
	}

	protected static class DatasetMetaDataPaper extends DatasetMetaData {
		private final int authorStartCol;
		private final int referenceStartCol;

		public DatasetMetaDataPaper(int ncol, int nrow, int maxAuthor, int maxAffiliation) {
			super(ncol, nrow, 5 + maxAuthor);
			this.schema = new ValueType[ncol];
			this.schema[0] = ValueType.INT64; // index id of this paper
			this.schema[1] = ValueType.STRING; //paper title
			this.schema[2] = ValueType.INT32; //year
			this.schema[3] = ValueType.STRING; //publication venue
			this.schema[4] = ValueType.STRING; // abstract

			for(int i = 5; i < maxAuthor + maxAffiliation + 5; i++)
				this.schema[i] = ValueType.STRING;

			for(int i = maxAuthor + maxAffiliation + 5; i < ncol; i++)
				this.schema[i] = ValueType.FP64;

			this.authorStartCol = 5;
			this.referenceStartCol = maxAuthor + maxAffiliation + 5;
		}

		public int getAuthorStartCol() {
			return authorStartCol;
		}

		public int getReferenceStartCol() {
			return referenceStartCol;
		}
	}

	protected static class DatasetMetaDataAuthor extends DatasetMetaData {
		private final int researchInterestStartCol;

		public DatasetMetaDataAuthor(int ncol, int nrow, int maxAffiliation, int maxResearchInterest) {
			super(ncol, nrow, 7);
			this.schema = new ValueType[ncol];
			this.schema[0] = ValueType.INT64; // index id of this author
			this.schema[1] = ValueType.STRING; // name  (separated by semicolons)
			this.schema[2] = ValueType.INT32; // the count of published papers of this author
			this.schema[3] = ValueType.INT32; // the total number of citations of this author
			this.schema[4] = ValueType.FP32; // the H-index of this author
			this.schema[5] = ValueType.FP32; // the P-index with equal A-index of this author
			this.schema[6] = ValueType.FP32; // the P-index with unequal A-index of this author

			for(int i = 7; i < maxAffiliation + maxResearchInterest + 7; i++)
				this.schema[i] = ValueType.STRING;
			this.researchInterestStartCol = 7 + maxAffiliation;
		}

		public int getResearchInterestStartCol() {
			return researchInterestStartCol;
		}
	}
}

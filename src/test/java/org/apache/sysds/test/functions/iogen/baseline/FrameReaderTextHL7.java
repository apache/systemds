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

package org.apache.sysds.test.functions.iogen.baseline;

import ca.uhn.hl7v2.model.Composite;
import ca.uhn.hl7v2.model.DataTypeException;
import ca.uhn.hl7v2.model.Group;
import ca.uhn.hl7v2.model.Message;
import ca.uhn.hl7v2.model.Primitive;
import ca.uhn.hl7v2.model.Segment;
import ca.uhn.hl7v2.model.Structure;
import ca.uhn.hl7v2.model.Type;
import ca.uhn.hl7v2.model.Varies;
import ca.uhn.hl7v2.HL7Exception;
import ca.uhn.hl7v2.parser.EncodingCharacters;
import ca.uhn.hl7v2.parser.PipeParser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.iogen.template.TemplateUtil;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public class FrameReaderTextHL7 extends FrameReader {
	protected int _numThreads;
	protected static JobConf job;
	protected static TemplateUtil.SplitOffsetInfos _offsets;
	protected int _rLen;
	protected int _cLen;
	protected static FileFormatPropertiesHL7 _props;

	public FrameReaderTextHL7(FileFormatPropertiesHL7 props) {
		//if unspecified use default properties for robustness
		_props = props;
		_numThreads = 1;
	}

	@Override
	public FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		LOG.debug("readFrameFromHDFS HL7");
		// prepare file access
		job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		FileInputFormat.addInputPath(job, path);

		// check existence and non-empty file
		checkValidInputFile(fs, path);

		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);

		String[] lnames = createOutputNames(names, clen);
		FrameBlock ret = computeSizeAndCreateOutputFrameBlock(informat, job, schema, lnames, splits, "MSH|");

		// core read (sequential/parallel)
		readHL7FrameFromHDFS(informat, splits, ret, schema);
		return ret;
	}

	protected FrameBlock computeSizeAndCreateOutputFrameBlock(TextInputFormat informat, JobConf job,
		Types.ValueType[] schema, String[] names, InputSplit[] splits, String beginToken)
		throws IOException, DMLRuntimeException {
		_rLen = 0;
		_cLen = names.length;

		// count rows in parallel per split
		try {
			_offsets = new TemplateUtil.SplitOffsetInfos(splits.length);
			for(int i = 0; i < splits.length; i++) {
				TemplateUtil.SplitInfo splitInfo = new TemplateUtil.SplitInfo();
				_offsets.setSeqOffsetPerSplit(i, splitInfo);
				_offsets.setOffsetPerSplit(i, 0);
			}

			int splitIndex = 0;
			for(InputSplit split : splits) {
				Integer nextOffset = splitIndex + 1 == splits.length ? null : splitIndex + 1;
				Integer nrows = countRows(_offsets, splitIndex, nextOffset, split, informat, job, beginToken);
				_offsets.setOffsetPerSplit(splitIndex, _rLen);
				_offsets.setLenghtPerSplit(splitIndex,_rLen+ nrows);
				_rLen += nrows;
				splitIndex++;
			}
		}
		catch(Exception e) {
			throw new IOException("Compute Size and Create Output Frame Block Error " + e.getMessage(), e);
		}
		FrameBlock ret = createOutputFrameBlock(schema, names, _rLen);
		return ret;
	}

	@Override
	public FrameBlock readFrameFromInputStream(InputStream is, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {

		//		// TODO: fix stream reader. incomplete
		//		LOG.debug("readFrameFromInputStream csv");
		//		ValueType[] lschema = null;
		//		String[] lnames = null;
		//
		//		InputStreamInputFormat informat = new InputStreamInputFormat(is);
		//		InputSplit[] splits = informat.getSplits(null, 1);
		//		splits = IOUtilFunctions.sortInputSplits(splits);
		//
		//		if(_props.getType().equals("paper")) {
		//			paperMetaData = computeAMinerSizePaper(null,null, splits);
		//			rlen = paperMetaData.nrow;
		//			lschema = paperMetaData.schema;
		//			lnames = paperMetaData.names;
		//		}
		//		else {
		//			authorMetaData = computeAMinerSizeAuthor(null,null, splits);
		//			rlen = authorMetaData.nrow;
		//			lschema = authorMetaData.schema;
		//			lnames = authorMetaData.names;
		//		}
		//		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);
		//
		//		// core read (sequential/parallel)
		//		if(_props.getType().equals("paper")) {
		//			readAMinerPaperFrameFromInputSplit(splits[0], rowIndexs[0], colBeginIndexs[0], informat, null, ret, schema);
		//		}
		//		else {
		//			readAMinerAuthorFrameFromInputSplit(splits[0], rowIndexs[0], informat, null, ret, schema);
		//		}
		//		return ret;

		return null;

	}

	protected void readHL7FrameFromHDFS(TextInputFormat informat, InputSplit[] splits, FrameBlock dest, ValueType[] schema) throws IOException {
		LOG.debug("readHL7FrameFromHDFS Message");

		for(int i = 0; i < splits.length; i++)
			readHL7FrameFromInputSplit(informat, splits[i], i, schema, dest);

	}

	private static void addRow(String messageString, PipeParser pipeParser, FrameBlock dest, Types.ValueType[] schema, int row) throws IOException {
		if(messageString.length() > 0) {
			try {
				// parse HL7 message
				Message message = pipeParser.parse(messageString.toString());
				ArrayList<String> values = new ArrayList<>();
				groupEncode(message, values);
				if(_props.isReadAllValues()) {
					int col = 0;
					for(String s : values)
						dest.set(row, col++, UtilFunctions.stringToObject(schema[col], s));
				}
				else if(_props.isRangeBaseRead()) {
					for(int i = 0; i < _props.getMaxColumnIndex(); i++)
						dest.set(row, i, UtilFunctions.stringToObject(schema[i], values.get(i)));
				}
				else {
					for(int i = 0; i < _props.getSelectedIndexes().length; i++) {
						dest.set(row, i, UtilFunctions.stringToObject(schema[i], values.get(_props.getSelectedIndexes()[i])));
					}
				}
			}
			catch(Exception exception) {
				throw new IOException("Can't part hel7 message:", exception);
			}
		}
	}

	protected static int readHL7FrameFromInputSplit(TextInputFormat informat, InputSplit split,  int splitCount, Types.ValueType[] schema, FrameBlock dest) throws IOException {
		RecordReader<LongWritable, Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
		LongWritable key = new LongWritable();
		Text value = new Text();
		int row = _offsets.getOffsetPerSplit(splitCount);
		TemplateUtil.SplitInfo splitInfo = _offsets.getSeqOffsetPerSplit(splitCount);

		int rlen = _offsets.getLenghtPerSplit(splitCount);
		int ri;
		int beginPosStr, endPosStr;
		String remainStr = "";
		String str = "";
		StringBuilder sb = new StringBuilder(splitInfo.getRemainString());
		long beginIndex = splitInfo.getRecordIndexBegin(0);
		long endIndex = splitInfo.getRecordIndexEnd(0);
		boolean flag;

		PipeParser pipeParser = new PipeParser();
		if(sb.length() > 0) {
			ri = 0;
			while(ri < beginIndex) {
				reader.next(key, value);
				sb.append(value.toString());
				ri++;
			}
			reader.next(key, value);
			String valStr = value.toString();
			sb.append(valStr.substring(0, splitInfo.getRecordPositionBegin(0)));

			addRow(sb.toString(), pipeParser, dest, schema, row);
			row++;
			sb = new StringBuilder(valStr.substring(splitInfo.getRecordPositionBegin(0)));
		}
		else {
			ri = -1;
		}

		int rowCounter = 0;
		while(row < rlen) {
			flag = reader.next(key, value);
			if(flag) {
				ri++;
				String valStr = value.toString();
				if(ri >= beginIndex && ri <= endIndex) {
					beginPosStr = ri == beginIndex ? splitInfo.getRecordPositionBegin(rowCounter) : 0;
					endPosStr = ri == endIndex ? splitInfo.getRecordPositionEnd(rowCounter) : valStr.length();
					sb.append(valStr.substring(beginPosStr, endPosStr));
					remainStr = valStr.substring(endPosStr);
					continue;
				}
				else {
					str = sb.toString();
					sb = new StringBuilder();
					sb.append(remainStr).append(valStr);
					if(rowCounter + 1 < splitInfo.getListSize()) {
						beginIndex = splitInfo.getRecordIndexBegin(rowCounter + 1);
						endIndex = splitInfo.getRecordIndexEnd(rowCounter + 1);
					}
					rowCounter++;
				}
			}
			else {
				str = sb.toString();
				sb = new StringBuilder();
			}
			addRow(str, pipeParser, dest, schema, row);
			row++;
		}
		return row;
	}

	protected static void groupEncode(Group groupObject, ArrayList<String> values) {
		String[] childNames = groupObject.getNames();
		try {
			String[] var5 = childNames;
			int var6 = childNames.length;

			for(int var7 = 0; var7 < var6; ++var7) {
				String name = var5[var7];
				Structure[] reps = groupObject.getAll(name);
				Structure[] var10 = reps;
				int var11 = reps.length;

				for(int var12 = 0; var12 < var11; ++var12) {
					Structure rep = var10[var12];
					if(rep instanceof Group) {
						groupEncode((Group) rep, values);
					}
					else if(rep instanceof Segment) {
						segmentEncode((Segment) rep, values);
					}
				}
			}
		}
		catch(Exception exception) {
			exception.printStackTrace();
		}
	}

	protected static void segmentEncode(Segment segmentObject, ArrayList<String> values) throws HL7Exception {
		int n = segmentObject.numFields();
		for(int i = 1; i <= n; ++i) {
			Type[] reps = segmentObject.getField(i);
			Type[] var8 = reps;
			int var9 = reps.length;
			for(int var10 = 0; var10 < var9; ++var10) {
				Type rep = var8[var10];
				encode(rep, values);
			}
		}
	}

	protected static void encode(Type datatypeObject, ArrayList<String> values) throws DataTypeException {
		if(_props.isRangeBaseRead() && values.size()>_props.getMaxColumnIndex())
			return;
		else if(_props.isQueryFilter() && values.size() > _props.getMaxColumnIndex())
			return;
		else {
			if(datatypeObject instanceof Varies) {
				encodeVaries((Varies) datatypeObject, values);
			}
			else if(datatypeObject instanceof Primitive) {
				encodePrimitive((Primitive) datatypeObject, values);
			}
			else if(datatypeObject instanceof Composite) {
				encodeComposite((Composite) datatypeObject, values);
			}
		}
	}

	protected static void encodeVaries(Varies datatypeObject, ArrayList<String> values) throws DataTypeException {
		if(datatypeObject.getData() != null) {
			encode(datatypeObject.getData(), values);
		}
	}

	protected static void encodePrimitive(Primitive datatypeObject, ArrayList<String> values) throws DataTypeException {
		String value = datatypeObject.getValue();
		boolean hasValue = value != null && value.length() > 0;
		if(hasValue) {
			try {
				EncodingCharacters ec = EncodingCharacters.getInstance(datatypeObject.getMessage());
				char esc = ec.getEscapeCharacter();
				int oldpos = 0;

				int pos;
				boolean escaping;
				for(escaping = false; (pos = value.indexOf(esc, oldpos)) >= 0; oldpos = pos + 1) {
					String v = value.substring(oldpos, pos);
					if(!escaping) {
						escaping = true;
					}
					else if(!v.startsWith(".") && !"H".equals(v) && !"N".equals(v)) {
					}
					else {
						escaping = false;
					}
				}

				if(oldpos <= value.length()) {
					StringBuilder sb = new StringBuilder();
					if(escaping) {
						sb.append(esc);
					}
					sb.append(value.substring(oldpos));
					values.add(sb.toString());
				}
			}
			catch(Exception var12) {
				throw new DataTypeException("Exception encoding Primitive: ", var12);
			}
		}
	}

	protected static void encodeComposite(Composite datatypeObject, ArrayList<String> values) throws DataTypeException {
		Type[] components = datatypeObject.getComponents();
		for(int i = 0; i < components.length; ++i) {
			encode(components[i], values);
		}
	}

	protected static int countRows(TemplateUtil.SplitOffsetInfos offsets, Integer curOffset, Integer nextOffset,
		InputSplit split, TextInputFormat inputFormat, JobConf job, String beginToken) throws IOException {
		int nrows = 0;

		ArrayList<Pair<Long, Integer>> beginIndexes = TemplateUtil.getTokenIndexOnMultiLineRecords(split, inputFormat, job, beginToken).getKey();
		ArrayList<Pair<Long, Integer>> endIndexes = new ArrayList<>();
		for(int i = 1; i < beginIndexes.size(); i++)
			endIndexes.add(beginIndexes.get(i));
		int i = 0;
		int j = 0;

		if(beginIndexes.get(0).getKey() > 0)
			nrows++;

		while(i < beginIndexes.size() && j < endIndexes.size()) {
			Pair<Long, Integer> p1 = beginIndexes.get(i);
			Pair<Long, Integer> p2 = endIndexes.get(j);
			int n = 0;
			while(p1.getKey() < p2.getKey() || (p1.getKey() == p2.getKey() && p1.getValue() < p2.getValue())) {
				n++;
				i++;
				if(i == beginIndexes.size()) {
					break;
				}
				p1 = beginIndexes.get(i);
			}
			j += n - 1;
			offsets.getSeqOffsetPerSplit(curOffset).addIndexAndPosition(beginIndexes.get(i - n).getKey(), endIndexes.get(j).getKey(),
					beginIndexes.get(i - n).getValue(), endIndexes.get(j).getValue());
			j++;
			nrows++;
		}
		if(nextOffset != null) {
			RecordReader<LongWritable, Text> reader = inputFormat.getRecordReader(split, job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();

			StringBuilder sb = new StringBuilder();

			for(long ri = 0; ri < beginIndexes.get(beginIndexes.size() - 1).getKey(); ri++) {
				reader.next(key, value);
			}
			if(reader.next(key, value)) {
				String strVar = value.toString();
				sb.append(strVar.substring(beginIndexes.get(beginIndexes.size() - 1).getValue()));
				while(reader.next(key, value)) {
					sb.append(value.toString());
				}
				offsets.getSeqOffsetPerSplit(nextOffset).setRemainString(sb.toString());
			}
		}
		else {
			nrows++;
			offsets.getSeqOffsetPerSplit(curOffset).addIndexAndPosition(endIndexes.get(endIndexes.size() -1).getKey(),	split.getLength()-1,0, 0);
		}
		offsets.getSeqOffsetPerSplit(curOffset).setNrows(nrows);
		offsets.setOffsetPerSplit(curOffset, nrows);
		return nrows;
	}
}
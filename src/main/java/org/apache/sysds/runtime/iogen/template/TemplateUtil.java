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

package org.apache.sysds.runtime.iogen.template;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.runtime.matrix.data.Pair;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class TemplateUtil {

	public static class SplitOffsetInfos {
		// offset & length info per split
		private int[] offsetPerSplit = null;
		private int[] lenghtPerSplit = null;
		private SplitInfo[] seqOffsetPerSplit = null;

		public SplitOffsetInfos(int numSplits) {
			lenghtPerSplit = new int[numSplits];
			offsetPerSplit = new int[numSplits];
			seqOffsetPerSplit = new SplitInfo[numSplits];
		}

		public int getLenghtPerSplit(int split) {
			return lenghtPerSplit[split];
		}

		public void setLenghtPerSplit(int split, int r) {
			lenghtPerSplit[split] = r;
		}

		public int getOffsetPerSplit(int split) {
			return offsetPerSplit[split];
		}

		public void setOffsetPerSplit(int split, int o) {
			offsetPerSplit[split] = o;
		}

		public SplitInfo getSeqOffsetPerSplit(int split) {
			return seqOffsetPerSplit[split];
		}

		public void setSeqOffsetPerSplit(int split, SplitInfo splitInfo) {
			seqOffsetPerSplit[split] = splitInfo;
		}
	}

	public static class SplitInfo {
		private int nrows;
		private ArrayList<Integer> recordIndexBegin;
		private ArrayList<Integer> recordIndexEnd;
		private ArrayList<Integer> recordPositionBegin;
		private ArrayList<Integer> recordPositionEnd;
		private String remainString;

		public SplitInfo() {
			recordIndexBegin = new ArrayList<>();
			recordIndexEnd = new ArrayList<>();
			recordPositionBegin = new ArrayList<>();
			recordPositionEnd = new ArrayList<>();
		}

		public void addIndexAndPosition(int beginIndex, int endIndex, int beginPos, int endPos) {
			recordIndexBegin.add(beginIndex);
			recordIndexEnd.add(endIndex);
			recordPositionBegin.add(beginPos);
			recordPositionEnd.add(endPos);
		}

		public int getNrows() {
			return nrows;
		}

		public void setNrows(int nrows) {
			this.nrows = nrows;
		}

		public String getRemainString() {
			return remainString;
		}

		public void setRemainString(String remainString) {
			this.remainString = remainString;
		}

		public int getRecordIndexBegin(int index) {
			return recordIndexBegin.get(index);
		}

		public int getRecordIndexEnd(int index) {
			return recordIndexEnd.get(index);
		}

		public int getRecordPositionBegin(int index) {
			return recordPositionBegin.get(index);
		}

		public int getRecordPositionEnd(int index) {
			return recordPositionEnd.get(index);
		}
	}

	public static ArrayList<Pair<Integer, Integer>> getTokenIndexOnMultiLineRecords(InputSplit split, TextInputFormat inputFormat, JobConf job,
		String token) throws IOException {
		RecordReader<LongWritable, Text> reader = inputFormat.getRecordReader(split, job, Reporter.NULL);
		LongWritable key = new LongWritable();
		Text value = new Text();
		ArrayList<Pair<Integer, Integer>> result = new ArrayList<>();

		int ri = 0;
		while(reader.next(key, value)) {
			String raw = value.toString();
			int index;
			int fromIndex = 0;
			do {
				index = raw.indexOf(token, fromIndex);
				if(index != -1) {
					result.add(new Pair<>(ri, index));
					fromIndex = index + token.length();
				}
				else
					break;
			}
			while(true);
			ri++;
		}
		result.add(new Pair<>(ri, 0));
		return result;
	}

	public static int getEndPos(String str, int strLen, int currPos, HashSet<String> endWithValueString) {
		int endPos = strLen;
		for(String d : endWithValueString) {
			int pos = d.length() > 0 ? str.indexOf(d, currPos) : strLen;
			if(pos != -1)
				endPos = Math.min(endPos, pos);
		}
		return endPos;
	}


	public static String getStringChunkOfBufferReader(BufferedReader br, String remainedStr,int chunkSize){
		StringBuilder sb = new StringBuilder();
		String str;
		int readSize = 0;
		try {
			while((str = br.readLine()) != null && readSize<chunkSize) {
				sb.append(str).append("\n");
				readSize += str.length();
			}
		}
		catch(Exception ex){

		}
		if(sb.length() >0) {
			if(remainedStr!=null && remainedStr.length() >0)
				return remainedStr + sb;
			else
				return sb.toString();
		}
		else
			return null;
	}
	protected int getColIndex(HashMap<String, Integer> colKeyPatternMap, String key){
		return colKeyPatternMap.getOrDefault(key, -1);
	}
}



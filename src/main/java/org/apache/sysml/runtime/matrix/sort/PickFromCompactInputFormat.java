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


package org.apache.sysml.runtime.matrix.sort;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.NumItemsByEachReducerMetaData;
import org.apache.sysml.runtime.matrix.data.Pair;

//key class to read has to be DoubleWritable
public class PickFromCompactInputFormat extends FileInputFormat<MatrixIndexes, MatrixCell>
{		
	public static final String VALUE_IS_WEIGHT="value.is.weight";
	public static final String INPUT_IS_VECTOR="input.is.vector";
	public static final String SELECTED_RANGES="selected.ranges";
	public static final String SELECTED_POINTS_PREFIX="selected.points.in.";
	public static final String VALUE_CLASS="value.class.to.read";
	public static final String PARTITION_OF_ZERO="partition.of.zero";
	public static final String NUMBER_OF_ZERO="number.of.zero";
	
	protected  boolean isSplitable(FileSystem fs, Path filename) {
		return false;
	}
	
	public static void setZeroValues(JobConf job, NumItemsByEachReducerMetaData metadata)
	{
		job.setInt(PARTITION_OF_ZERO, metadata.getPartitionOfZero());
		job.setLong(NUMBER_OF_ZERO, metadata.getNumberOfZero());
	}
	
	@SuppressWarnings("rawtypes")
	public static void setKeyValueClasses(JobConf job, 
			Class<? extends WritableComparable> keyClass, Class<? extends Writable> valueClass)
	{
	//	job.setClass(KEY_CLASS, keyClass, WritableComparable.class);
		job.setClass(VALUE_CLASS, valueClass, Writable.class);
	}
	
	private static class SortElement implements Comparable<SortElement>
	{
		double prob;
		int index;
		public SortElement(double p, int i)
		{
			prob=p;
			index=i;
		}
		
		@Override
		public int compareTo(SortElement other) {
			return Double.compare(this.prob, other.prob);
		}
		
		@Override
		public boolean equals(Object o) {
			if( !(o instanceof SortElement) )
				return false;
			SortElement that = (SortElement)o;
			return (prob == that.prob);
		}
		
		@Override
		public int hashCode() {
			throw new RuntimeException("hashCode() should never be called on instances of this class.");
		}
	}
	
	private static void getPointsInEachPartFile(long[] counts, double[] probs, HashMap<Integer, 
			ArrayList<Pair<Integer, Integer>>> posMap)
	{	
		long[] ranges=new long[counts.length];
		ranges[0]=counts[0];
		for(int i=1; i<counts.length; i++)
			ranges[i]=ranges[i-1]+counts[i];
		
		long total=ranges[ranges.length-1];
		
		SortElement[] sortedProbs=new SortElement[probs.length];
		for(int i=0; i<sortedProbs.length; i++)
			sortedProbs[i]=new SortElement(probs[i], i+1);
		Arrays.sort(sortedProbs);
		
		int currentPart=0;
		for(SortElement e: sortedProbs )
		{
			long pos=(long)Math.ceil(total*e.prob);
			while(ranges[currentPart]<pos)
				currentPart++;
			ArrayList<Pair<Integer, Integer>> vec=posMap.get(currentPart);
			if(vec==null)
			{
				vec=new ArrayList<Pair<Integer, Integer>>();
				posMap.put(currentPart, vec);
			}
			
			//set the actual position starting from 0
			if(currentPart>0)
				vec.add( new Pair<Integer, Integer>( (int)(pos-ranges[currentPart-1]-1), e.index));
			else
				vec.add( new Pair<Integer, Integer>( (int)pos-1,  e.index));
		}
	}
	
	public static Set<Integer> setPickRecordsInEachPartFile(JobConf job, NumItemsByEachReducerMetaData metadata, double[] probs)
	{
		HashMap<Integer, ArrayList<Pair<Integer, Integer>>> posMap
		=new HashMap<Integer, ArrayList<Pair<Integer, Integer>>>();
		
		getPointsInEachPartFile(metadata.getNumItemsArray(), probs, posMap);
		
		for(Entry<Integer, ArrayList<Pair<Integer, Integer>>> e: posMap.entrySet())
		{
			job.set(SELECTED_POINTS_PREFIX+e.getKey(), getString(e.getValue()));
			//System.out.println(e.getKey()+": "+getString(e.getValue()));
		}
		job.setBoolean(INPUT_IS_VECTOR, true);
		return posMap.keySet();
	}
	
	public static void setRangePickPartFiles(JobConf job, NumItemsByEachReducerMetaData metadata, double lbound, double ubound) {
		
		if(lbound<0 || lbound > 1 || ubound <0 || ubound >1 || lbound >= ubound ) {
			throw new RuntimeException("Invalid ranges for range pick: [" + lbound + "," + ubound + "]");
		}
		
		long[] counts = metadata.getNumItemsArray();
		long[] ranges=new long[counts.length];
		ranges[0]=counts[0];
		for(int i=1; i<counts.length; i++)
			ranges[i]=ranges[i-1]+counts[i];
		long sumwt=ranges[ranges.length-1];
		
		double qbegin = lbound*sumwt;
		double qend   = ubound*sumwt;
		
		// Find part files that overlap with range [qbegin,qend]
		int partID = -1;
		long wt = 0;
		
		// scan until the part containing qbegin
		while(wt < qbegin) {
			partID++;
			wt += counts[partID];
		}
		
		StringBuilder sb = new StringBuilder();
		while(wt <= qend) {
			sb.append(partID+"," + (wt-counts[partID]) + ";"); // partID, weight until this part
			partID++;
			if(partID < counts.length) 
				wt += counts[partID];
		}
		sb.append(partID+"," + (wt-counts[partID]) + ";"); 
		sb.append(sumwt + "," + lbound + "," + ubound);
		
		job.set(SELECTED_RANGES, sb.toString());
		job.setBoolean(INPUT_IS_VECTOR, false);
	}
	
	private static String getString(ArrayList<Pair<Integer, Integer>> value) 
	{
		StringBuilder sb = new StringBuilder();
		for(Pair<Integer, Integer> i: value) {
			sb.append(i.getKey());
			sb.append(":");
			sb.append(i.getValue());
			sb.append(",");
		}
		
		return sb.substring(0, sb.length()-1);
	}

	public RecordReader<MatrixIndexes, MatrixCell> getRecordReader(InputSplit split
			, JobConf job, Reporter reporter) throws IOException {
		if(job.getBoolean(INPUT_IS_VECTOR, true))
			return new PickRecordReader(job, (FileSplit) split);
		else
			return new RangePickRecordReader(job, (FileSplit) split);
	}
	
	public static class RangePickRecordReader implements RecordReader<MatrixIndexes, MatrixCell>
	{
		private Path path;
	    
		private FSDataInputStream currentStream;
		protected long totLength;
		private DoubleWritable readKey=new DoubleWritable();
		private IntWritable readValue = new IntWritable(0);
		private boolean noRecordsNeeded=false;
		private int index=0;
		private int beginPart=-1, endPart=-1, currPart=-1;
		private double sumwt = 0.0, readWt;  // total weight (set in JobConf)
		private double lbound, ubound;
		private HashMap<Integer,Double> partWeights = null;
		private boolean isFirstRecord = true;
		private ReadWithZeros reader=null;
		
		public RangePickRecordReader(JobConf job, FileSplit split) 
			throws IOException 
		{
			parseSelectedRangeString(job.get(SELECTED_RANGES));
			
			// check if the current part file needs to be processed
	    	path = split.getPath();
	    	totLength = split.getLength();
	    	currentStream = IOUtilFunctions.getFileSystem(path, job).open(path);
	    	currPart = getIndexInTheArray(path.getName());
	    	
	    	if ( currPart < beginPart || currPart > endPart ) {
	    		noRecordsNeeded = true;
	    		return;
	    	}
	    	
			int part0=job.getInt(PARTITION_OF_ZERO, -1);
			boolean contain0s=false;
			long numZeros =0;
	    	if(part0==currPart) {
	    		contain0s = true;
	    		numZeros = job.getLong(NUMBER_OF_ZERO, 0);
	    	}
	    	reader=new ReadWithZeros(currentStream, contain0s, numZeros);
		}
		
		private int getIndexInTheArray(String name) {
			int i=name.indexOf("part-");
			return Integer.parseInt(name.substring(i+5));
		}
		
		private void parseSelectedRangeString(String str) {
			String[] f1 = str.split(";");
			String[] f2 = null;
			
			// Each field of the form: "pid,wt" where wt is the total wt until the part pid
			partWeights = new HashMap<Integer,Double>();
			for(int i=0; i < f1.length-1; i++) {
				f2 = f1[i].split(",");
				if(i==0) {
					beginPart = Integer.parseInt(f2[0]);
				}
				if (i==f1.length-2) {
					endPart = Integer.parseInt(f2[0]);
				}
				partWeights.put(i, Double.parseDouble(f2[1]));
			}
			
			// last field: "sumwt, lbound, ubound"
			f2 = f1[f1.length-1].split(",");
			sumwt  = Double.parseDouble(f2[0]);
			lbound = Double.parseDouble(f2[1]);
			ubound = Double.parseDouble(f2[2]);
		}
		
		public boolean next(MatrixIndexes key, MatrixCell value) 
			throws IOException 
		{	
			if(noRecordsNeeded)
				return false;

			double qLowerD = sumwt*lbound; // lower quantile in double 
			double qUpperD = sumwt*ubound;
			
			// weight until current part
			if (isFirstRecord) {
				if( !partWeights.containsKey(currPart) )
					return false; //noRecordsNeeded
				readWt = partWeights.get(currPart);
				isFirstRecord = false;
			}
			double tmpWt = 0;
			
			if ( currPart == beginPart || currPart == endPart ) {
				boolean ret = reader.readNextKeyValuePairs(readKey, readValue);
				tmpWt = readValue.get();

				while(readWt+tmpWt < qLowerD) {
					readWt += tmpWt;
					ret &= reader.readNextKeyValuePairs(readKey, readValue);
					tmpWt = readValue.get();
				}
				
				if((readWt<qLowerD && readWt+tmpWt >= qLowerD) || (readWt+tmpWt <= qUpperD) || (readWt<qUpperD && readWt+tmpWt>=qUpperD)) {
					key.setIndexes(++index,1);
					value.setValue(readKey.get()*tmpWt);
					readWt += tmpWt;
					return ret;
				}
				else {
					return false;
				}
			}
			else { 
				// read full part
				boolean ret = reader.readNextKeyValuePairs(readKey, readValue);
				tmpWt = readValue.get();
				
				key.setIndexes(++index,1);
				value.setValue(readKey.get()*tmpWt);
				
				readWt += tmpWt;
				return ret;
			}
		}
		
		@Override
		public void close() throws IOException {
			IOUtilFunctions.closeSilently(currentStream);
		}

		@Override
		public MatrixIndexes createKey() {
			return new MatrixIndexes();
		}

		@Override
		public MatrixCell createValue() {
			return new MatrixCell();
		}

		@Override
		public long getPos() throws IOException {
			long currentOffset = currentStream == null ? 0 : currentStream.getPos();
			return currentOffset;
		}

		@Override
		public float getProgress() throws IOException {
			float progress = (float) getPos() / totLength;
			return (progress>=0 && progress<=1) ? progress : 1.0f;
		}
	}

	public static class PickRecordReader implements RecordReader<MatrixIndexes, MatrixCell>
	{
		private boolean valueIsWeight=true;
		private FileSystem fs;
		private Path path;
		
	    private FSDataInputStream currentStream;
		private int posIndex=0;
		
		private int[] pos=null; //starting from 0
		private int[] indexes=null;
		private DoubleWritable readKey = new DoubleWritable();
		private IntWritable readValue = new IntWritable();
		private int numRead=0;
		private boolean noRecordsNeeded=false;
		ReadWithZeros reader=null;
		
		private int getIndexInTheArray(String name)
		{
			int i=name.indexOf("part-");
			return Integer.parseInt(name.substring(i+5));
		}
		
		public PickRecordReader(JobConf job, FileSplit split)
			throws IOException
		{
			path = split.getPath();
			fs = IOUtilFunctions.getFileSystem(path, job);
			currentStream = fs.open(path);
	    	int partIndex=getIndexInTheArray(path.getName());
	    	String arrStr=job.get(SELECTED_POINTS_PREFIX+partIndex);
	    	if(arrStr==null || arrStr.isEmpty()) {
	    		noRecordsNeeded=true;
	    		return;
	    	}
	    	
	    	String[] strs=arrStr.split(",");
	    	pos=new int[strs.length];
	    	indexes=new int[strs.length];
	    	for(int i=0; i<strs.length; i++) {
	    		String[] temp=strs[i].split(":");
	    		pos[i]=Integer.parseInt(temp[0]);
	    		indexes[i]=Integer.parseInt(temp[1]);
	    	}
	    	
			valueIsWeight=job.getBoolean(VALUE_IS_WEIGHT, true);
			
			int part0=job.getInt(PARTITION_OF_ZERO, -1);
			boolean contain0s=false;
			long numZeros =0;
	    	if(part0==partIndex) {
	    		contain0s = true;
	    		numZeros = job.getLong(NUMBER_OF_ZERO, 0);
	    	}
	    	reader=new ReadWithZeros(currentStream, contain0s, numZeros);
		}
		
		public boolean next(MatrixIndexes key, MatrixCell value)
			throws IOException 
		{
			if(noRecordsNeeded || posIndex>=pos.length) {
				return false;
			}
		
			while(numRead<=pos[posIndex])
			{
				reader.readNextKeyValuePairs(readKey, readValue);
				if(valueIsWeight)
					numRead+=readValue.get();
				else
					numRead++;
			}
			
			key.setIndexes(indexes[posIndex], 1);
			value.setValue(readKey.get());
			posIndex++;
			return true;
		}

		@Override
		public void close() throws IOException {
			IOUtilFunctions.closeSilently(currentStream);			
		}

		@Override
		public MatrixIndexes createKey() {
			return new MatrixIndexes();
		}

		@Override
		public MatrixCell createValue() {
			return new MatrixCell();
		}

		@Override
		public long getPos() throws IOException {
			long currentOffset = currentStream == null ? 0 : currentStream.getPos();
			return currentOffset;
		}

		@Override
		public float getProgress() throws IOException {
			if(pos!=null)
				return (float)posIndex/(float)pos.length;
			else return 100.0f;
		}
	}
}

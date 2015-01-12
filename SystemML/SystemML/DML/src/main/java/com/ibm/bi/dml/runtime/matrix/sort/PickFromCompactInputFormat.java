/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.sort;

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

import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.data.Pair;

//key class to read has to be DoubleWritable
public class PickFromCompactInputFormat extends FileInputFormat<MatrixIndexes, MatrixCell>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
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
		//System.out.println("range string: " + sb.toString());
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
		//private boolean valueIsWeight=true;
		protected long totLength;
		protected FileSystem fs;
		protected Path path;
		protected JobConf conf;
	    
	    protected FSDataInputStream currentStream;
		
		private int startPos=0;
		private int numToRead=0;
		DoubleWritable readKey=new DoubleWritable();
		Writable readValue = new IntWritable(0);
		private boolean noRecordsNeeded=false;
		private int rawKeyValuesRead=0;
		private int index=0;
		//private int currentRepeat=0;
		
		int beginPart=-1, endPart=-1, currPart=-1;
		double sumwt = 0.0, readWt, wtUntilCurrPart;  // total weight (set in JobConf)
		double lbound, ubound;
		double[] partWeights = null;
		boolean isFirstRecord = true;
		
		
		//to handle zeros
		ReadWithZeros reader=null;
		/*private boolean contain0s=false;
		private boolean justFound0=false;
		private DoubleWritable keyAfterZero;
		private Writable valueAfterZero; 
		private long numZeros=0;*/
		
		private int getIndexInTheArray(String name)
		{
			int i=name.indexOf("part-");
			assert(i>=0);
			return Integer.parseInt(name.substring(i+5));
		}
		
		private void parseSelectedRangeString(String str) {
			String[] f1 = str.split(";");
			String[] f2 = null;
			
			// Each field of the form: "pid,wt" where wt is the total wt until the part pid
			partWeights = new double[f1.length-1];
			for(int i=0; i < f1.length-1; i++) {
				f2 = f1[i].split(",");
				if(i==0) {
					beginPart = Integer.parseInt(f2[0]);
				}
				if (i==f1.length-2) {
					endPart = Integer.parseInt(f2[0]);
				}
				partWeights[i] = Double.parseDouble(f2[1]);
			}
			
			// last field: "sumwt, lbound, ubound"
			f2 = f1[f1.length-1].split(",");
			sumwt  = Double.parseDouble(f2[0]);
			lbound = Double.parseDouble(f2[1]);
			ubound = Double.parseDouble(f2[2]);
		}

		public RangePickRecordReader(JobConf job, FileSplit split) throws IOException {
			parseSelectedRangeString(job.get(SELECTED_RANGES));
			
			// check if the current part file needs to be processed
	    	path = split.getPath();
	    	totLength = split.getLength();
	    	currentStream = FileSystem.get(job).open(path);
	    	currPart = getIndexInTheArray(path.getName());
	    	//System.out.println("RangePickRecordReader(): sumwt " + sumwt + " currPart " + currPart + " partRange [" + beginPart + "," + endPart + "]");
	    	
	    	if ( currPart < beginPart || currPart > endPart ) {
	    		System.out.println("    currPart is out of range. Skipping part!");
	    		noRecordsNeeded = true;
	    		return;
	    	}
	    	
	    	Class<? extends Writable> valueClass=(Class<? extends Writable>) job.getClass(VALUE_CLASS, Writable.class);
	    	try {
	    	//	readKey=keyClass.newInstance();
				readValue=valueClass.newInstance();
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			//valueIsWeight=job.getBoolean(VALUE_IS_WEIGHT, true);
			
			int part0=job.getInt(PARTITION_OF_ZERO, -1);
			boolean contain0s=false;
			long numZeros =0;
	    	if(part0==currPart)
	    	{
	    		contain0s = true;
	    		numZeros = job.getLong(NUMBER_OF_ZERO, 0);
	    	}
	    	reader=new ReadWithZeros(currentStream, contain0s, numZeros);
		}
		
		public boolean next(MatrixIndexes key, MatrixCell value) throws IOException {
			assert(currPart!=-1);
			
			if(noRecordsNeeded)
				// this part file does not fall within the required range of values
				return false;

			double qLowerD = sumwt*lbound; // lower quantile in double 
			double qUpperD = sumwt*ubound;
			
			// weight until current part
			if (isFirstRecord) {
				readWt = partWeights[currPart];
				wtUntilCurrPart = partWeights[currPart];
				isFirstRecord = false;
			}
			double tmpWt = 0;
			
			if ( currPart == beginPart || currPart == endPart ) {
				//readWt = partWeights[currPart];
				
				reader.readNextKeyValuePairs(readKey, (IntWritable)readValue);
				tmpWt = ((IntWritable)readValue).get();

				while(readWt+tmpWt < qLowerD) {
					readWt += tmpWt;
					reader.readNextKeyValuePairs(readKey, (IntWritable)readValue);
					tmpWt = ((IntWritable)readValue).get();
				}
				
				if((readWt<qLowerD && readWt+tmpWt >= qLowerD) || (readWt+tmpWt <= qUpperD) || (readWt<qUpperD && readWt+tmpWt>=qUpperD)) {
					key.setIndexes(++index,1);
					value.setValue(readKey.get()*tmpWt);
					readWt += tmpWt;
					//System.out.println("currpart " + currPart + ", return (" + index +",1): " + readKey.get()*tmpWt + "| [" + readKey.get() + "," + tmpWt +"] readWt=" + readWt);
					return true;
				}
				else {
					return false;
				}
			}
			else { //if(currPart != beginPart && currPart != endPart) {
				// read full part
				reader.readNextKeyValuePairs(readKey, (IntWritable)readValue);
				tmpWt = ((IntWritable)readValue).get();
				
				key.setIndexes(++index,1);
				value.setValue(readKey.get()*tmpWt);
				
				readWt += tmpWt;
				//System.out.println("currpart " + currPart + ", return (" + index +",1): " + readKey.get()*tmpWt + "| [" + readKey.get() + "," + tmpWt +"] readWt=" + readWt);
				return true;
			}
		}
		
		@Override
		public void close() throws IOException {
			//DO Nothing
			currentStream.close();
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
			if(numToRead>0)
				return (float)(rawKeyValuesRead-startPos)/(float)numToRead;
			else
				return 100.0f;
		}
	}
	
	public static class PickRecordReader implements RecordReader<MatrixIndexes, MatrixCell>
	{
		private boolean valueIsWeight=true;
		protected long totLength;
		protected FileSystem fs;
		protected Path path;
		protected JobConf conf;
	    
	    protected FSDataInputStream currentStream;
		private int posIndex=0;
		
		private int[] pos=null; //starting from 0
		private int[] indexes=null;
		DoubleWritable readKey=new DoubleWritable();
		Writable readValue;
		private int numRead=0;
		private boolean noRecordsNeeded=false;
		
		//to handle zeros
		ReadWithZeros reader=null;
		/*private boolean contain0s=false;
		private boolean justFound0=false;
		private DoubleWritable keyAfterZero;
		private Writable valueAfterZero; 
		private long numZeros=0;*/
		
		private int getIndexInTheArray(String name)
		{
			int i=name.indexOf("part-");
			assert(i>=0);
			return Integer.parseInt(name.substring(i+5));
		}
		
		public PickRecordReader(JobConf job, FileSplit split)
				throws IOException{
			fs = FileSystem.get(job);
	    	path = split.getPath();
	    	totLength = split.getLength();
	    	currentStream = fs.open(path);
	    	int partIndex=getIndexInTheArray(path.getName());
	    	String arrStr=job.get(SELECTED_POINTS_PREFIX+partIndex);
	    	//System.out.println("read back in the recordreader: "+arrStr);
	    	if(arrStr==null || arrStr.isEmpty())
	    	{
	    		noRecordsNeeded=true;
	    		return;
	    	}
	    	String[] strs=arrStr.split(",");
	    	pos=new int[strs.length];
	    	indexes=new int[strs.length];
	    	for(int i=0; i<strs.length; i++)
	    	{
	    		String[] temp=strs[i].split(":");
	    		pos[i]=Integer.parseInt(temp[0]);
	    		indexes[i]=Integer.parseInt(temp[1]);
	    	}
	    //	Class<? extends WritableComparable> keyClass=(Class<? extends WritableComparable>) job.getClass(KEY_CLASS, WritableComparable.class);
	    	Class<? extends Writable> valueClass=(Class<? extends Writable>) job.getClass(VALUE_CLASS, Writable.class);
	    	try {
	    	//	readKey=keyClass.newInstance();
				readValue=valueClass.newInstance();
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			valueIsWeight=job.getBoolean(VALUE_IS_WEIGHT, true);
			
			int part0=job.getInt(PARTITION_OF_ZERO, -1);
			boolean contain0s=false;
			long numZeros =0;
	    	if(part0==partIndex)
	    	{
	    		contain0s = true;
	    		numZeros = job.getLong(NUMBER_OF_ZERO, 0);
	    	}
	    	reader=new ReadWithZeros(currentStream, contain0s, numZeros);
		}
		
		public boolean next(MatrixIndexes key, MatrixCell value)
		throws IOException {
			
			if(noRecordsNeeded || posIndex>=pos.length)
			{
				//System.out.println("return false");
				//System.out.println("noRecordsNeeded="+noRecordsNeeded+", currentStream.getPos()="+currentStream.getPos()
				//		+", totLength="+totLength+", posIndex="+posIndex+", pos.length="+pos.length);
				return false;
			}
		
			//System.out.println("numRead="+numRead+" pos["+posIndex+"]="+pos[posIndex]);
			while(numRead<=pos[posIndex])
			{
				reader.readNextKeyValuePairs(readKey, (IntWritable)readValue);
			//	System.out.println("**** numRead "+numRead+" -- "+readKey+": "+readValue);
				if(valueIsWeight)
					numRead+=((IntWritable)readValue).get();
				else
					numRead++;
			}
			
			key.setIndexes(indexes[posIndex], 1);
			value.setValue(readKey.get());
			posIndex++;
			//System.out.println("next: "+key+", "+value);
			return true;
		}

		@Override
		public void close() throws IOException {
			currentStream.close();
			
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

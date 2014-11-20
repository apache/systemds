/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.sort;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;
import java.util.TreeMap;
import java.util.Vector;
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
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	static final String VALUE_IS_WEIGHT="value.is.weight";
	private boolean inputIsVector=true;
	public static final String INPUT_IS_VECTOR="input.is.vector";
	public static final String SELECTED_RANGES_PREFIX="selected.ranges.in.";
	public static final String SELECTED_POINTS_PREFIX="selected.points.in.";
	//public static final String KEY_CLASS="key.class.to.read";
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
	
	public static void setValueIsWeight(JobConf job, boolean viw)
	{
		job.setBoolean(VALUE_IS_WEIGHT, viw);
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
			Vector<Pair<Integer, Integer>>> posMap)
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
			Vector<Pair<Integer, Integer>> vec=posMap.get(currentPart);
			if(vec==null)
			{
				vec=new Vector<Pair<Integer, Integer>>();
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
		HashMap<Integer, Vector<Pair<Integer, Integer>>> posMap
		=new HashMap<Integer, Vector<Pair<Integer, Integer>>>();
		
		getPointsInEachPartFile(metadata.getNumItemsArray(), probs, posMap);
		
		for(Entry<Integer, Vector<Pair<Integer, Integer>>> e: posMap.entrySet())
		{
			job.set(SELECTED_POINTS_PREFIX+e.getKey(), getString(e.getValue()));
			//System.out.println(e.getKey()+": "+getString(e.getValue()));
		}
		job.setBoolean(INPUT_IS_VECTOR, true);
		return posMap.keySet();
	}
	
	public static Set<Integer> setPickRecordsInEachPartFile(JobConf job, NumItemsByEachReducerMetaData metadata, double lbound, double ubound)
	{
		TreeMap<Integer, Pair<Integer, Integer>> posMap
		=new TreeMap<Integer, Pair<Integer, Integer>>();
		getRangeInEachPartitionFile(metadata.getNumItemsArray(), lbound, ubound, posMap);
		
		int startIndex=0;
		for(Entry<Integer, Pair<Integer, Integer>> e: posMap.entrySet())
		{
			job.set(SELECTED_RANGES_PREFIX+e.getKey(), e.getValue().getKey()+":"+e.getValue().getValue()+":"+(startIndex+1));
			//System.out.println("++ range for "+e.getKey()+" is "+e.getValue().getKey()+":"+e.getValue().getValue()+":"+(startIndex+1));
			startIndex+=e.getValue().getValue();
		}
		job.setBoolean(INPUT_IS_VECTOR, false);
		return posMap.keySet();
	}
	
	
	private static void getRangeInEachPartitionFile(long[] counts,
			double lbound, double ubound,
			TreeMap<Integer, Pair<Integer, Integer>> posMap) {
		
		long[] ranges=new long[counts.length];
		ranges[0]=counts[0];
		for(int i=1; i<counts.length; i++)
			ranges[i]=ranges[i-1]+counts[i];
		long total=ranges[ranges.length-1];
		long lpos=(long)Math.ceil(total*lbound)+1;//lower bound is inclusive
		long upos=(long)Math.ceil(total*ubound)+1;//upper bound is non inclusive
		//System.out.println("bound: "+lpos+", "+upos);
		
		int i=0;
		while(ranges[i]<lpos)
			i++;
		int start;
		if(i>0)
			start=(int) (lpos-ranges[i-1]-1);
		else
			start=(int) (lpos-1);
		
		while(i<ranges.length && ranges[i]<upos)
		{
			posMap.put(i, new Pair<Integer, Integer>(start, (int)(counts[i]-start)));
			//System.out.println("add range: "+start+" num:"+(int)(counts[i]-start));
			i++;
			start=0;
		}
		int endCount;
		if(i>0)
			endCount=(int) (upos-ranges[i-1]-1);
		else
			endCount=(int) (upos-1);
		if(endCount>0)
		{
			posMap.put(i, new Pair<Integer, Integer>(start, endCount-start));
			//System.out.println("add range: "+start+" num:"+(int)(endCount-start));
		}
	}

	private static String getString(Vector<Pair<Integer, Integer>> value) {
		String str="";
		for(Pair<Integer, Integer> i: value)
			str+=i.getKey()+":"+i.getValue()+",";
		return str.substring(0, str.length()-1);
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
		private boolean valueIsWeight=true;
		protected long totLength;
		protected FileSystem fs;
		protected Path path;
		protected JobConf conf;
	    
	    protected FSDataInputStream currentStream;
		
		private int startPos=0;
		private int numToRead=0;
		DoubleWritable readKey=new DoubleWritable();
		Writable readValue;
		private boolean noRecordsNeeded=false;
		private int rawKeyValuesRead=0;
		private int index=0;
		private int currentRepeat=0;
		
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
		
		public RangePickRecordReader(JobConf job, FileSplit split)
				throws IOException{
			fs = FileSystem.get(job);
	    	path = split.getPath();
	    	totLength = split.getLength();
	    	currentStream = fs.open(path);
	    	int partIndex=getIndexInTheArray(path.getName());
	    	
	    	String str=job.get(SELECTED_RANGES_PREFIX+partIndex);
	    	if(str==null || str.isEmpty()) 
	    	{
	    		noRecordsNeeded=true;
	    		return;
	    	}
	    	String[] temp=str.split(":");
	    	startPos=Integer.parseInt(temp[0]);
	    	numToRead=Integer.parseInt(temp[1]);
	    	index=Integer.parseInt(temp[2]);
	    	assert(numToRead>0);
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
		/*
		private void readNextKeyValuePairs()throws IOException 
		{
			if(contain0s && justFound0)
			{
				readKey=keyAfterZero;
				readValue=valueAfterZero;
				contain0s=false;
			}else
			{
				readKey.readFields(currentStream);
				readValue.readFields(currentStream);
			}
			
			if(contain0s && !justFound0 && readKey.get()>=0)
			{
				justFound0=true;
				keyAfterZero=readKey;
				valueAfterZero=readValue;
				readKey=new DoubleWritable(0);
				readValue=new IntWritable((int)numZeros);
			}
		}*/
		
		public boolean next(MatrixIndexes key, MatrixCell value)
		throws IOException {
			
			if(noRecordsNeeded || rawKeyValuesRead>=startPos+numToRead)
				return false;
			
			//search from start
			while(rawKeyValuesRead+currentRepeat<=startPos)
			{
				rawKeyValuesRead+=currentRepeat;
				reader.readNextKeyValuePairs(readKey, (IntWritable)readValue);
				//System.out.println("**** numRead "+rawKeyValuesRead+" -- "+readKey+": "+readValue);
			//	LOG.info("**** numRead "+rawKeyValuesRead+" -- "+readKey+": "+readValue);
				if(valueIsWeight)
					currentRepeat=((IntWritable)readValue).get();
				else
					currentRepeat=1;
			}
			if(rawKeyValuesRead<=startPos && rawKeyValuesRead+currentRepeat>startPos)
			{
				currentRepeat=rawKeyValuesRead+currentRepeat-startPos;
				rawKeyValuesRead=startPos;
			}
			
			if(currentRepeat<=0)
			{
				reader.readNextKeyValuePairs(readKey, (IntWritable)readValue);
			//	System.out.println("**** numRead "+rawKeyValuesRead+" -- "+readKey+": "+readValue);
			//	LOG.info("**** numRead "+rawKeyValuesRead+" -- "+readKey+": "+readValue);
				if(valueIsWeight)
					currentRepeat=((IntWritable)readValue).get();
				else
					currentRepeat=1;
			}

			if(currentRepeat>0)
			{
				key.setIndexes(index, 1);
				index++;
				value.setValue(readKey.get());
			//	System.out.println("next: "+key+", "+value);
			//	LOG.info("next: "+key+", "+value);
				rawKeyValuesRead++;
				currentRepeat--;
				return true;
			}
			
			return false;
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

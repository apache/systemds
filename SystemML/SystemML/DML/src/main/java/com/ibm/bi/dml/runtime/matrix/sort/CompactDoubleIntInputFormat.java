/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.sort;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import java.util.TreeMap;
import java.util.Map.Entry;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.Pair;


public class CompactDoubleIntInputFormat extends FileInputFormat<MatrixIndexes, MatrixCell>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String SELECTED_RANGES_PREFIX="selected.ranges.in.";
	public static final String SELECTED_POINTS_PREFIX="selected.points.in.";
	
	public static void getPointsInEachPartFile(long[] counts, double[] probs, HashMap<Integer, 
			ArrayList<Pair<Integer, Integer>>> posMap)
	{	
		long[] ranges=new long[counts.length];
		ranges[0]=counts[0];
		for(int i=1; i<counts.length; i++)
			ranges[i]=ranges[i-1]+counts[i];
		
		long total=ranges[ranges.length-1];
		
		TreeMap<Double, Integer> sortedProbs=new TreeMap<Double, Integer>();
		for(int i=0; i<probs.length; i++)
			sortedProbs.put(probs[i], i);
		
		int currentPart=0;
		for(Entry<Double, Integer> e: sortedProbs.entrySet() )
		{
			long pos=(long)Math.ceil(total*e.getKey());
			while(ranges[currentPart]<pos)
				currentPart++;
			ArrayList<Pair<Integer, Integer>> vec=posMap.get(currentPart);
			if(vec==null)
			{
				vec=new ArrayList<Pair<Integer, Integer>>();
				posMap.put(currentPart, vec);
			}
			if(currentPart>0)
				vec.add( new Pair<Integer, Integer>( (int)(pos-ranges[currentPart-1]-1), e.getValue()));
			else
				vec.add( new Pair<Integer, Integer>( (int)pos-1, e.getValue()));
		}
	}
	
	public static Set<Integer> setRecordCountInEachPartFile(JobConf job, long[] counts, double[] probs)
	{
		HashMap<Integer, ArrayList<Pair<Integer, Integer>>> posMap
		=new HashMap<Integer, ArrayList<Pair<Integer, Integer>>>();
		
		getPointsInEachPartFile(counts, probs, posMap);
		
		for(Entry<Integer, ArrayList<Pair<Integer, Integer>>> e: posMap.entrySet())
		{
			job.set(SELECTED_POINTS_PREFIX+e.getKey(), getString(e.getValue()));
		}
		
		return posMap.keySet();
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
		return new DoubleIntRecordReader(job, (FileSplit) split);
	}
	
	public static class DoubleIntRecordReader implements RecordReader<MatrixIndexes, MatrixCell>
	{
		protected long totLength;
		protected FileSystem fs;
		protected Path path;
		protected JobConf conf;
	    
	    protected FSDataInputStream currentStream;
		private int posIndex=0;
		
		private int[] pos; //starting from 0
		private int[] indexes;
		DoubleWritable readKey=new DoubleWritable();
		IntWritable readValue=new IntWritable();
		private int numRead=0;
		private boolean noRecordsNeeded=false;
		
		private int getIndexInTheArray(String name)
		{
			int i=name.indexOf("part-");
			assert(i>=0);
			return Integer.parseInt(name.substring(i+5));
		}
		
		public DoubleIntRecordReader(JobConf job, FileSplit split)
				throws IOException {
			fs = FileSystem.get(job);
	    	path = split.getPath();
	    	totLength = split.getLength();
	    	currentStream = fs.open(path);
	    	int partIndex=getIndexInTheArray(path.getName());
	    	String arrStr=job.get(SELECTED_POINTS_PREFIX+partIndex);
	    	if(arrStr==null || arrStr.isEmpty()) noRecordsNeeded=true;
	    	String[] strs=arrStr.split(",");
	    	pos=new int[strs.length];
	    	indexes=new int[strs.length];
	    	for(int i=0; i<strs.length; i++)
	    	{
	    		String[] temp=strs[i].split(":");
	    		pos[i]=Integer.parseInt(temp[0]);
	    		indexes[i]=Integer.parseInt(temp[1]);
	    	}
		}
		
		public boolean next(MatrixIndexes key, MatrixCell value)
		throws IOException {
			
			if(noRecordsNeeded || currentStream.getPos()>=totLength || posIndex>=pos.length)
				return false;
			
			while(numRead<=pos[posIndex])
			{
				readKey.readFields(currentStream);
				readValue.readFields(currentStream);
				numRead=readValue.get();
			}
			
			key.setIndexes(indexes[posIndex]+1, 1);
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
			return (float)posIndex/(float)pos.length;
		}
	}
}

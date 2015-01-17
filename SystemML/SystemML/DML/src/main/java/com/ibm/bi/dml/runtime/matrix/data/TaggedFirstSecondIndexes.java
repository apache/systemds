/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;

import com.ibm.bi.dml.runtime.instructions.mr.CSVWriteInstruction;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


//sorted by first index, tag, and second index
public class TaggedFirstSecondIndexes implements WritableComparable<TaggedFirstSecondIndexes>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected long first=-1;
	protected byte tag=-1;
	protected long second=-1;
	
	public TaggedFirstSecondIndexes(){};
	public TaggedFirstSecondIndexes(long i1, byte t, long i2)
	{
		setIndexes(i1,i2);
		setTag(t);
	}
	
	public void setTag(byte t) {
		tag=t;
		
	}
	public TaggedFirstSecondIndexes(TaggedFirstSecondIndexes other) {
		setIndexes(other.first, other.second);
		setTag(other.tag);
	}
	
	public String toString()
	{
		return "("+first+", "+second+") tag: "+tag;
	}
	
	public byte getTag()
	{
		return tag;
	}
	
	public long getFirstIndex()
	{
		return first;
	}
	public long getSecondIndex()
	{
		return second;
	}
	
	public void setIndexes(long i1, long i2)
	{
		first=i1;
		second=i2;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		first=in.readLong();
		tag=in.readByte();
		second=in.readLong();
		
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong(first);
		out.writeByte(tag);
		out.writeLong(second);
		
	}
	
	public int compareTo(TaggedFirstSecondIndexes other)
	{
		if(this.first!=other.first)
			return (this.first>other.first? 1:-1);
		else if(this.tag!=other.tag)
			return this.tag-other.tag;
		else if(this.second!=other.second)
			return (this.second>other.second? 1:-1);
		return 0;
	}

	@Override
	public boolean equals(Object other)
	{
		if( !(other instanceof TaggedFirstSecondIndexes))
			return false;
		
		TaggedFirstSecondIndexes tother = (TaggedFirstSecondIndexes)other;
		return (this.first==tother.first && this.tag==tother.tag && this.second==tother.second);
	}
	
	@Override
	public int hashCode() {
		 return UtilFunctions.longHashFunc((first<<32)+second+tag+MatrixIndexes.ADD_PRIME1)%MatrixIndexes.DIVIDE_PRIME;
	}
	
	/** A Comparator optimized for TaggedFirstSecondIndexes. */ 
	public static class Comparator implements RawComparator<TaggedFirstSecondIndexes>
	{
		@Override
		public int compare(byte[] b1, int s1, int l1,
                byte[] b2, int s2, int l2)
		{
			return WritableComparator.compareBytes(b1, s1, l1, b2, s2, l2);
		}

		@Override
		public int compare(TaggedFirstSecondIndexes m1, TaggedFirstSecondIndexes m2) {
			return m1.compareTo(m2);
		}
		
	}
	
	/**
	   * Partition based on the first index.
	   */
	  public static class FirstIndexPartitioner implements Partitioner<TaggedFirstSecondIndexes, Writable>{
	    @Override
	    public int getPartition(TaggedFirstSecondIndexes key, Writable value, int numPartitions) 
	    {
	      return UtilFunctions.longHashFunc(key.getFirstIndex()*127)%10007%numPartitions;
	    }

		@Override
		public void configure(JobConf arg0) {
			
		}
	  }
	  
	  /**
	   * Partition based on the first index.
	   */
	  public static class FirstIndexRangePartitioner implements Partitioner<TaggedFirstSecondIndexes, Writable>{
		  long[] rstep=null;//some parts of the array may be empty, but it is for performance
		  @Override
	    public int getPartition(TaggedFirstSecondIndexes key, Writable value, int numPartitions) 
	    {
	      	return (int) ((key.first-1)/rstep[key.tag]);
	    }

		@Override
		public void configure(JobConf job) {
			String[] matrices=MRJobConfiguration.getInputPaths(job);
			int partitions = job.getNumReduceTasks();
			//get the dimension of all the representative matrices
			long[] inRstep=new long[matrices.length];
			for(int i=0; i<matrices.length; i++)
				inRstep[i]=(long) Math.ceil((double)MRJobConfiguration.getNumRows(job, (byte)i)/(double)partitions);
			byte maxIndex=0;
			HashMap<Byte, Long> outRsteps=new HashMap<Byte, Long>();
			try {
				CSVWriteInstruction[] ins = MRJobConfiguration.getCSVWriteInstructions(job);
				for(CSVWriteInstruction in: ins)
				{
					outRsteps.put(in.output, inRstep[in.input]);
					if(in.output>maxIndex)
						maxIndex=in.output;
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			rstep=new long[maxIndex+1];
			for(Entry<Byte, Long> outRstep: outRsteps.entrySet())
				rstep[outRstep.getKey()]=outRstep.getValue();
		}
	  }
	  
	  /**
	   * Partition based on the first index.
	   */
	  public static class TagPartitioner implements Partitioner<TaggedFirstSecondIndexes, MatrixValue>{
	    @Override
	    public int getPartition(TaggedFirstSecondIndexes key, MatrixValue value, int numPartitions) 
	    {
	      return key.tag%numPartitions;
	    }

		@Override
		public void configure(JobConf arg0) {
			
		}
	  }
}

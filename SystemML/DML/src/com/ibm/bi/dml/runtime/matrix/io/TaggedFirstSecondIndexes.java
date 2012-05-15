package com.ibm.bi.dml.runtime.matrix.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;

import com.ibm.bi.dml.runtime.util.UtilFunctions;


//sorted by first index, tag, and second index
public class TaggedFirstSecondIndexes implements WritableComparable<TaggedFirstSecondIndexes>{

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

	public boolean equals(TaggedFirstSecondIndexes other)
	{
		return (this.first==other.first && this.tag==other.tag && this.second==other.second);
	}
	
	public boolean equals(Object other)
	{
		if( !(other instanceof TaggedFirstSecondIndexes))
			return false;
		return equals((TaggedFirstSecondIndexes)other);
	}
	
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
	  public static class FirstIndexPartitioner implements Partitioner<TaggedFirstSecondIndexes, MatrixValue>{
	    @Override
	    public int getPartition(TaggedFirstSecondIndexes key, MatrixValue value, int numPartitions) 
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

package dml.runtime.matrix.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;

import dml.runtime.util.UtilFunctions;


public class TaggedTripleIndexes extends TaggedFirstSecondIndexes{

	private long third=-1;
	public TaggedTripleIndexes(){}
	public TaggedTripleIndexes(long i1, long i2, long i3, byte t)
	{
		super(i1, t, i2);
		third=i3;
	}
	
	public TaggedTripleIndexes(TaggedTripleIndexes that)
	{
		setIndexes(that);
	}
	public void setIndexes(TaggedTripleIndexes that) {
		
		this.first=that.first;
		this.second=that.second;
		this.third=that.third;
		this.tag=that.tag;
	}
	
	public String toString()
	{
		return "("+first+", "+second+") k: "+third+", tag: "+tag;
	}
	
	public long getThirdIndex()
	{
		return third;
	}
	
	public void setIndexes(long i1, long i2, long i3)
	{
		super.setIndexes(i1, i2);
		third=i3;
	}
	
	public void readFields(DataInput in) throws IOException {
		first=in.readLong();
		second=in.readLong();
		third=in.readLong();
		tag=in.readByte();
	}
	
	public void write(DataOutput out) throws IOException {
		out.writeLong(first);
		out.writeLong(second);
		out.writeLong(third);
		out.writeByte(tag);
	}
	
	public int compareTo(TaggedTripleIndexes other)
	{
		if(this.first!=other.first)
			return (this.first>other.first? 1:-1);
		else if(this.second!=other.second)
			return (this.second>other.second? 1:-1);
		else if(this.third!=other.third)
			return (this.third>other.third? 1:-1);
		else if(this.tag!=other.tag)
			return this.tag-other.tag;
		return 0;
	}

	public boolean equals(TaggedTripleIndexes other)
	{
		return (this.first==other.first && this.tag==other.tag 
				&& this.second==other.second && this.third==other.third);
	}
	
	public boolean equals(Object other)
	{
	//	LOG.info("calling equals for MatrixCellIndexes!");
		if( !(other instanceof TaggedTripleIndexes))
			return false;
		return equals((TaggedTripleIndexes)other);
	}
	
	 public int hashCode() {
		 return UtilFunctions.longHashFunc((first<<32)+(second<<16)+third+tag+MatrixIndexes.ADD_PRIME1)%MatrixIndexes.DIVIDE_PRIME;
	 }
	
	public static class Comparator implements RawComparator<TaggedTripleIndexes>
	{
		@Override
		public int compare(byte[] b1, int s1, int l1,
                byte[] b2, int s2, int l2)
		{
			return WritableComparator.compareBytes(b1, s1, l1, b2, s2, l2);
		}

		@Override
		public int compare(TaggedTripleIndexes m1, TaggedTripleIndexes m2) {
			return m1.compareTo(m2);
		}	
	}
	
	 /**
	   * Partition based on the first and second index.
	   */
	  public static class FirstTwoIndexesPartitioner implements Partitioner<TaggedTripleIndexes, MatrixBlock>{
	    @Override
	    public int getPartition(TaggedTripleIndexes key, MatrixBlock value, 
	                            int numPartitions) {
	   
	    	return UtilFunctions.longHashFunc((key.getFirstIndex()*127)
	    			+key.getSecondIndex()+MatrixIndexes.ADD_PRIME1)
	    			%MatrixIndexes.DIVIDE_PRIME%numPartitions;
	    	/*return UtilFunctions.longHashFunc(((key.getFirstIndex()+MatrixIndexes.ADD_PRIME2)<<32)
	    			+key.getSecondIndex()+MatrixIndexes.ADD_PRIME1)
	    			%MatrixIndexes.DIVIDE_PRIME%numPartitions;*/
	     // return UtilFunctions.longHashFunc(key.getFirstIndex()*127 + key.getSecondIndex())%10007%numPartitions;
	    }

		@Override
		public void configure(JobConf arg0) {
			
		}
	  }

}

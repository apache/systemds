package dml.runtime.matrix.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;

import dml.runtime.util.UtilFunctions;

public class TripleIndexes implements WritableComparable<TripleIndexes>{
	private long first=-1;
	private long second=-1;
	private long third=-1;

	public TripleIndexes(){}
	public TripleIndexes(long i1, long i2, long i3)
	{
		first=i1;
		second=i2;
		third=i3;
	}
	
	public TripleIndexes(TripleIndexes that)
	{
		setIndexes(that);
	}
	public void setIndexes(TripleIndexes that) {
		
		this.first=that.first;
		this.second=that.second;
		this.third=that.third;
	}
	
	public String toString()
	{
		return "("+first+", "+second+") k: "+third;
	}
	
	public long getFirstIndex()
	{
		return first;
	}
	public long getSecondIndex()
	{
		return second;
	}
	
	public long getThirdIndex()
	{
		return third;
	}
	
	public void setIndexes(long i1, long i2, long i3)
	{
		first=i1;
		second=i2;
		third=i3;
	}
	
	public void readFields(DataInput in) throws IOException {
		first=in.readLong();
		second=in.readLong();
		third=in.readLong();
	}
	
	public void write(DataOutput out) throws IOException {
		out.writeLong(first);
		out.writeLong(second);
		out.writeLong(third);
	}
	
	public int compareTo(TripleIndexes other)
	{
		if(this.first!=other.first)
			return (this.first>other.first? 1:-1);
		else if(this.second!=other.second)
			return (this.second>other.second? 1:-1);
		else if(this.third!=other.third)
			return (this.third>other.third? 1:-1);
		return 0;
	}

	public boolean equals(TripleIndexes other)
	{
		return (this.first==other.first && this.second==other.second && this.third==other.third);
	}
	
	public boolean equals(Object other)
	{
	//	LOG.info("calling equals for MatrixCellIndexes!");
		if( !(other instanceof TripleIndexes))
			return false;
		return equals((TripleIndexes)other);
	}
	
	 public int hashCode() {
		 return UtilFunctions.longHashFunc((first<<32)+(second<<16)+third+MatrixIndexes.ADD_PRIME1)%MatrixIndexes.DIVIDE_PRIME;
	 }
	
	public static class Comparator implements RawComparator<TripleIndexes>
	{
		@Override
		public int compare(byte[] b1, int s1, int l1,
                byte[] b2, int s2, int l2)
		{
			return WritableComparator.compareBytes(b1, s1, l1, b2, s2, l2);
		}

		@Override
		public int compare(TripleIndexes m1, TripleIndexes m2) {
			return m1.compareTo(m2);
		}	
	}
	
	 /**
	   * Partition based on the first and second index.
	   */
	  public static class FirstTwoIndexesPartitioner implements Partitioner<TripleIndexes, TaggedMatrixValue>{
	    @Override
	    public int getPartition(TripleIndexes key, TaggedMatrixValue value, 
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

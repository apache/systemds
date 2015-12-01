/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */


package org.apache.sysml.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;

import org.apache.sysml.runtime.util.UtilFunctions;



public class TaggedTripleIndexes extends TaggedFirstSecondIndexes
{
	
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

	@Override
	public boolean equals(Object other)
	{
		if( !(other instanceof TaggedTripleIndexes))
			return false;
		
		TaggedTripleIndexes tother = (TaggedTripleIndexes)other;
		return (this.first==tother.first && this.tag==tother.tag 
				&& this.second==tother.second && this.third==tother.third);
	}
	
	@Override
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

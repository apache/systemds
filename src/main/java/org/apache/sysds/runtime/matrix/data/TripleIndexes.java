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


package org.apache.sysds.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.sysds.runtime.util.UtilFunctions;


public class TripleIndexes implements WritableComparable<TripleIndexes>, Serializable
{
	
	private static final long serialVersionUID = -4514135024726916288L;

	private long first=-1;
	private long second=-1;
	private long third=-1;

	public TripleIndexes(){}
	
	public TripleIndexes(long i1, long i2, long i3) {
		first=i1;
		second=i2;
		third=i3;
	}
	
	public TripleIndexes(TripleIndexes that) {
		setIndexes(that);
	}
	public void setIndexes(TripleIndexes that) {
		this.first=that.first;
		this.second=that.second;
		this.third=that.third;
	}
	
	@Override
	public String toString() {
		return "("+first+", "+second+") k: "+third;
	}
	
	public long getFirstIndex() {
		return first;
	}
	public long getSecondIndex() {
		return second;
	}
	
	public long getThirdIndex() {
		return third;
	}
	
	public void setIndexes(long i1, long i2, long i3) {
		first=i1;
		second=i2;
		third=i3;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		first=in.readLong();
		second=in.readLong();
		third=in.readLong();
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong(first);
		out.writeLong(second);
		out.writeLong(third);
	}
	
	@Override
	public int compareTo(TripleIndexes other) {
		if(this.first!=other.first)
			return (this.first>other.first? 1:-1);
		else if(this.second!=other.second)
			return (this.second>other.second? 1:-1);
		else if(this.third!=other.third)
			return (this.third>other.third? 1:-1);
		return 0;
	}

	@Override
	public boolean equals(Object other)
	{
		if( !(other instanceof TripleIndexes))
			return false;
		
		TripleIndexes tother = (TripleIndexes)other;
		return (this.first==tother.first && this.second==tother.second && this.third==tother.third);
	}
	
	@Override
	public int hashCode() {
		return UtilFunctions.longHashCode(first, second, third);
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
}

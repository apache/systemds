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

import org.apache.sysml.runtime.DMLUnsupportedOperationException;


public class WeightedPair extends WeightedCell 
{

	private static final long serialVersionUID = 8772815876289553196L;

	private double other=0;
	public String toString()
	{
		return value+", "+other+": "+weight;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		value=in.readDouble();
		other=in.readDouble();
		weight=in.readDouble();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(value);
		out.writeDouble(other);
		out.writeDouble(weight);
	}

	private static WeightedPair checkType(MatrixValue cell) throws DMLUnsupportedOperationException
	{
		if( cell!=null && !(cell instanceof WeightedPair))
			throw new DMLUnsupportedOperationException("the Matrix Value is not WeightedPair!");
		return (WeightedPair) cell;
	}
	public void copy(MatrixValue that){
		WeightedPair c2;
		try {
			c2 = checkType(that);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		}
		value=c2.getValue();
		other=c2.getOtherValue();
		weight=c2.getWeight();
	}
	
	public double getOtherValue() {
		return other;
	}
	
	public void setOtherValue(double ov)
	{
		other=ov;
	}

	@Override
	public int compareTo(Object o) 
	{
		if( !(o instanceof WeightedPair) )
			return -1;
	
		WeightedPair that = (WeightedPair)o;
		if(this.value!=that.value)
			return Double.compare(this.value, that.value);
		else if(this.other!=that.other)
			return Double.compare(this.other, that.other);
		else if(this.weight!=that.weight)
			return Double.compare(this.weight, that.weight);
		else return 0;
	}
	
	@Override 
	public boolean equals(Object o)
	{	
		if( !(o instanceof WeightedPair) )
			return false;
		
		WeightedPair that = (WeightedPair)o;
		return (value==that.value && other==that.other && weight == that.weight);
	}
	
	@Override
	public int hashCode() {
		throw new RuntimeException("hashCode() should never be called on instances of this class.");
	}
}

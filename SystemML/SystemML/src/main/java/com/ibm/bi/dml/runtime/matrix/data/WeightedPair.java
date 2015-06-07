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

import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;


public class WeightedPair extends WeightedCell 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

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

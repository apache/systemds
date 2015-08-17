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
import java.util.Collection;
import java.util.Map;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;


public class WeightedCell extends MatrixCell
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -2283995259000895325L;
	
	protected double weight=0;
	
	public String toString()
	{
		return value+": "+weight;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		value=in.readDouble();
		weight=in.readDouble();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(value);
		out.writeDouble(weight);
	}

	private static WeightedCell checkType(MatrixValue cell) 
	throws DMLUnsupportedOperationException
	{
		if( cell!=null && !(cell instanceof WeightedCell))
			throw new DMLUnsupportedOperationException("the Matrix Value is not WeightedCell!");
		return (WeightedCell) cell;
	}
	public void copy(MatrixValue that){
		WeightedCell c2;
		try {
			c2 = checkType(that);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		}
		value=c2.getValue();
		weight=c2.getWeight();
	}
	
	@Override
	public int compareTo(Object o) 
	{
		if( !(o instanceof WeightedCell) )
			return -1;
			
		WeightedCell that=(WeightedCell)o;
		if(this.value!=that.value)
			return Double.compare(this.value, that.value);
		else if(this.weight!=that.weight)
			return Double.compare(this.weight, that.weight);
		else return 0;
	}
	
	@Override
	public boolean equals(Object o)
	{
		if( !(o instanceof WeightedCell) )
			return false;
		
		WeightedCell that=(WeightedCell)o;
		return (value==that.value && weight==that.weight);
	}

	@Override
	public int hashCode() {
		throw new RuntimeException("hashCode() should never be called on instances of this class.");
	}
	
	public void setWeight(double w)
	{
		weight=w;
	}
	
	public double getWeight()
	{
		return weight;
	}

	public double getValue()
	{
		return value;
	}
	
	@Override
	public MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op,
			MatrixValue result, int brlen, int bclen,
			MatrixIndexes indexesIn) throws DMLUnsupportedOperationException {
		super.aggregateUnaryOperations(op, result, brlen, bclen, indexesIn);
		WeightedCell c3=checkType(result);
		c3.setWeight(weight);
		return c3;
	}

	//TODO: how to handle -minus left vs. minus right
	public void denseScalarOperationsInPlace(ScalarOperator op)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		value=op.executeScalar(value);
	}

	@Override
	public MatrixValue reorgOperations(ReorgOperator op, MatrixValue result,
			int startRow, int startColumn, int length)
			throws DMLUnsupportedOperationException {
		super.reorgOperations(op, result, startRow, startColumn, length);
		WeightedCell c3=checkType(result);
		c3.setWeight(weight);
		return c3;
	}

	@Override
	public MatrixValue scalarOperations(ScalarOperator op, MatrixValue result) 
	throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		WeightedCell c3=checkType(result);
		c3.setValue(op.fn.execute(value, op.getConstant()));
		c3.setWeight(weight);
		return c3;
	}

	public void sparseScalarOperationsInPlace(ScalarOperator op)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		value=op.executeScalar(value);
	}

	public void sparseUnaryOperationsInPlace(UnaryOperator op)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		value=op.fn.execute(value);
	}

	@Override
	public MatrixValue unaryOperations(UnaryOperator op, MatrixValue result)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		WeightedCell c3=checkType(result);
		c3.setValue(op.fn.execute(value));
		c3.setWeight(weight);
		return c3;
	}

	@Override
	public void unaryOperationsInPlace(UnaryOperator op)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		value=op.fn.execute(value);
	}

	@Override
	public void getCellValues(Collection<Double> ret) {
		ret.add(value);	
	}

	@Override
	public void getCellValues(Map<Double, Integer> ret) {
		ret.put(value, 1);
	}
}

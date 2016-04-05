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


package org.apache.sysml.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;


public class WeightedCell extends MatrixCell
{

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
	throws DMLRuntimeException
	{
		if( cell!=null && !(cell instanceof WeightedCell))
			throw new DMLRuntimeException("the Matrix Value is not WeightedCell!");
		return (WeightedCell) cell;
	}
	public void copy(MatrixValue that){
		WeightedCell c2;
		try {
			c2 = checkType(that);
		} catch (DMLRuntimeException e) {
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
			MatrixIndexes indexesIn) throws DMLRuntimeException {
		super.aggregateUnaryOperations(op, result, brlen, bclen, indexesIn);
		WeightedCell c3=checkType(result);
		c3.setWeight(weight);
		return c3;
	}

	//TODO: how to handle -minus left vs. minus right
	public void denseScalarOperationsInPlace(ScalarOperator op)
			throws DMLRuntimeException {
		value=op.executeScalar(value);
	}

	@Override
	public MatrixValue reorgOperations(ReorgOperator op, MatrixValue result,
			int startRow, int startColumn, int length)
			throws DMLRuntimeException {
		super.reorgOperations(op, result, startRow, startColumn, length);
		WeightedCell c3=checkType(result);
		c3.setWeight(weight);
		return c3;
	}

	@Override
	public MatrixValue scalarOperations(ScalarOperator op, MatrixValue result) 
		throws DMLRuntimeException 
	{
		WeightedCell c3=checkType(result);
		c3.setValue(op.fn.execute(value, op.getConstant()));
		c3.setWeight(weight);
		return c3;
	}

	public void sparseScalarOperationsInPlace(ScalarOperator op)
			throws DMLRuntimeException {
		value=op.executeScalar(value);
	}

	public void sparseUnaryOperationsInPlace(UnaryOperator op)
			throws DMLRuntimeException {
		value=op.fn.execute(value);
	}

	@Override
	public MatrixValue unaryOperations(UnaryOperator op, MatrixValue result)
			throws DMLRuntimeException {
		WeightedCell c3=checkType(result);
		c3.setValue(op.fn.execute(value));
		c3.setWeight(weight);
		return c3;
	}

	@Override
	public void unaryOperationsInPlace(UnaryOperator op)
			throws DMLRuntimeException {
		value=op.fn.execute(value);
	}
}

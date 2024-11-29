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
import java.util.ArrayList;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.IndexRange;

public class MatrixCell extends MatrixValue implements Serializable
{
	private static final long serialVersionUID = -7755996717411912578L;
	
	protected double value;

	public MatrixCell() {
		value=0;
	}

	public MatrixCell(double v) {
		value=v;
	}

	private static MatrixCell checkType(MatrixValue cell) {
		if( cell!=null && !(cell instanceof MatrixCell))
			throw new DMLRuntimeException("the Matrix Value is not MatrixCell!");
		return (MatrixCell) cell;
	}

	public void setValue(double v) {
		value=v;
	}

	public double getValue() {
		return value;
	}

	
	@Override
	public void copy(MatrixValue that, boolean sp){
		copy(that);
	}

	@Override
	public void copy(MatrixValue that) {
		if( that==null || !(that instanceof MatrixCell))
			throw new RuntimeException("the Matrix Value is not MatrixCell!");	
		
		MatrixCell c2 = (MatrixCell)that;
		value = c2.getValue();
	}

	@Override
	public long getNonZeros() {
		return 1;
	}

	@Override
	public int getNumColumns() {
		return 1;
	}

	@Override
	public int getNumRows() {
		return 1;
	}

	@Override
	public double get(int r, int c) {
		return value;
	}

	@Override
	public boolean isInSparseFormat() {
		return false;
	}
	
	@Override
	public boolean isEmpty(){
		return (value==0);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		value=in.readDouble();		
	}

	@Override
	public void reset() {
		value=0;
	}

	@Override
	public void reset(int rl, int cl) {
		value=0;
	}

	@Override
	public void reset(int rl, int cl, boolean sp) {		
		value=0;
	}
	
	@Override
	public void reset(int rl, int cl, boolean sp, long nnzs) {
		value=0;
	}

	@Override
	public void reset(int rl, int cl, double v) {
		value=v;
	}

	@Override
	public void set(int r, int c, double v) {
		value=v;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(value);
	}


	@Override
	public MatrixValue binaryOperations(BinaryOperator op,
			MatrixValue thatValue, MatrixValue result) {
		throw new UnsupportedOperationException();
	}

	@Override
	public MatrixValue binaryOperationsInPlace(BinaryOperator op, MatrixValue thatValue) {
		throw new UnsupportedOperationException();
	}

	public void denseScalarOperationsInPlace(ScalarOperator op) {
		value=op.executeScalar(value);
	}

	@Override
	public MatrixValue reorgOperations(ReorgOperator op, MatrixValue result,
			int startRow, int startColumn, int length) { 
		throw new UnsupportedOperationException();
	}

	@Override
	public MatrixValue scalarOperations(ScalarOperator op, MatrixValue result) {
		MatrixCell c3=checkType(result);
		c3.setValue(op.fn.execute(value, op.getConstant()));
		return c3;
	}

	public void sparseScalarOperationsInPlace(ScalarOperator op) {
		value=op.executeScalar(value);
	}

	public void sparseUnaryOperationsInPlace(UnaryOperator op) {
		value=op.fn.execute(value);
	}

	@Override
	public MatrixValue unaryOperations(UnaryOperator op, MatrixValue result) {
		MatrixCell c3=checkType(result);
		c3.setValue(op.fn.execute(value));
		return c3;
	}

	public int compareTo(MatrixCell o) {
		return Double.compare(this.value, o.value);
	}

	@Override
	public int compareTo(Object o) {
		if(!(o instanceof MatrixCell))
			return -1;	
		return Double.compare(this.value, ((MatrixCell) o).value);
	}
	
	@Override
	public boolean equals(Object o) {
		if(!(o instanceof MatrixCell))
			return false;	
		return (value==((MatrixCell) o).value);
	}
	
	@Override
	public int hashCode() {
		throw new RuntimeException("hashCode() should never be called on instances of this class.");
	}

	@Override
	public void incrementalAggregate(AggregateOperator aggOp,
			MatrixValue correction, MatrixValue newWithCorrection, boolean deep) {
		throw new UnsupportedOperationException();
	}

	@Override
	public MatrixValue zeroOutOperations(MatrixValue result, IndexRange range) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void incrementalAggregate(AggregateOperator aggOp,
			MatrixValue newWithCorrection) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void ctableOperations(Operator op, MatrixValue that,
			MatrixValue that2, CTableMap resultMap, MatrixBlock resultBlock) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void ctableOperations(Operator op, MatrixValue that, double scalarThat2, boolean ignoreZeros, 
			CTableMap ctableResult, MatrixBlock ctableResultBlock) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void ctableOperations(Operator op, double scalarThat,
			double scalarThat2, CTableMap resultMap, MatrixBlock resultBlock) {
		throw new UnsupportedOperationException();
	}
	
	@Override
	public void ctableOperations(Operator op, MatrixIndexes ix1, double scalarThat, boolean left, int blen,
			CTableMap resultMap, MatrixBlock resultBlock) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void ctableOperations(Operator op, double scalarThat,
			MatrixValue that2, CTableMap resultMap, MatrixBlock resultBlock) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void slice(ArrayList<IndexedMatrixValue> outlist,
			IndexRange range, int rowCut, int colCut, int blen, int boundaryRlen, int boundaryClen) {
		throw new UnsupportedOperationException();
	}
	
	@Override
	public MatrixValue replaceOperations(MatrixValue result, double pattern, double replacement) {
		throw new UnsupportedOperationException();
	}

	@Override
	public MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, int blen,
		MatrixIndexes indexesIn, boolean inCP) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void append(MatrixValue valueIn2, ArrayList<IndexedMatrixValue> outlist,
			int blen, boolean cbind, boolean m2IsLast, int nextNCol) {
		throw new UnsupportedOperationException();
	}
}

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
import org.apache.sysds.runtime.functionobjects.CTable;
import org.apache.sysds.runtime.functionobjects.ReduceDiag;
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

	public MatrixCell()
	{
		value=0;
	}

	public MatrixCell(MatrixCell that)
	{
		this.value=that.value;
	}
	
	public MatrixCell(MatrixValue that) {
		if(that instanceof MatrixCell)
			this.value=((MatrixCell)that).value;
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
	public double getValue(int r, int c) {
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
	public void setValue(int r, int c, double v) {
		value=v;	
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(value);
	}

	@Override
	public MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op,
			MatrixValue result, int blen,
			MatrixIndexes indexesIn) {
		
		MatrixCell c3=checkType(result);
		if(c3==null)
			c3=new MatrixCell();
		
		if(op.indexFn instanceof ReduceDiag)
		{
			if(indexesIn.getRowIndex()==indexesIn.getColumnIndex())
				c3.setValue(getValue());
			else
				c3.setValue(0);
		}
		else
			c3.setValue(getValue());
		return c3;
	}

	@Override
	public MatrixValue binaryOperations(BinaryOperator op,
			MatrixValue thatValue, MatrixValue result) {
		MatrixCell c2=checkType(thatValue);
		MatrixCell c3=checkType(result);
		if(c3==null)
			c3=new MatrixCell();
		c3.setValue(op.fn.execute(this.getValue(), c2.getValue()));
		return c3;
	}

	@Override
	public MatrixValue binaryOperationsInPlace(BinaryOperator op,
			MatrixValue thatValue) {
		MatrixCell c2=checkType(thatValue);
		setValue(op.fn.execute(this.getValue(), c2.getValue()));
		return this;
	}

	public void denseScalarOperationsInPlace(ScalarOperator op) {
		value=op.executeScalar(value);
	}

	@Override
	public MatrixValue reorgOperations(ReorgOperator op, MatrixValue result,
			int startRow, int startColumn, int length) { 
		MatrixCell c3=checkType(result);
		if(c3==null)
			c3=new MatrixCell();
		c3.setValue(getValue());
		return c3;
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
		throw new DMLRuntimeException("MatrixCell.incrementalAggregate should never be called");
	}

	@Override
	public MatrixValue zeroOutOperations(MatrixValue result, IndexRange range, boolean complementary) {
		if(range.rowStart!=0 || range.rowEnd!=0 || range.colStart!=0 || range.colEnd!=0)
			throw new DMLRuntimeException("wrong range: "+range+" for matrixCell");
		MatrixCell c3=checkType(result);
		c3.setValue(value);
		return c3;
	}

	@Override
	public void incrementalAggregate(AggregateOperator aggOp,
			MatrixValue newWithCorrection) {
		throw new DMLRuntimeException("MatrixCell.incrementalAggregate should never be called");
	}

	@Override
	public void ctableOperations(Operator op, MatrixValue that,
			MatrixValue that2, CTableMap resultMap, MatrixBlock resultBlock) {
		MatrixCell c2=checkType(that);
		MatrixCell c3=checkType(that2);
		CTable ctable = CTable.getCTableFnObject();
		if ( resultMap != null)
			ctable.execute(this.value, c2.value, c3.value, false, resultMap);
		else
			ctable.execute(this.value, c2.value, c3.value, false, resultBlock);
	}

	@Override
	public void ctableOperations(Operator op, MatrixValue that, double scalarThat2, boolean ignoreZeros, 
			CTableMap ctableResult, MatrixBlock ctableResultBlock) {
		MatrixCell c2=checkType(that);
		CTable ctable = CTable.getCTableFnObject();
		if ( ctableResult != null)
			ctable.execute(this.value, c2.value, scalarThat2, ignoreZeros, ctableResult);
		else
			ctable.execute(this.value, c2.value, scalarThat2, ignoreZeros, ctableResultBlock);
	}

	@Override
	public void ctableOperations(Operator op, double scalarThat,
			double scalarThat2, CTableMap resultMap, MatrixBlock resultBlock) {
		CTable ctable = CTable.getCTableFnObject();
		if ( resultMap != null)
			ctable.execute(this.value, scalarThat, scalarThat2, false, resultMap);
		else
			ctable.execute(this.value, scalarThat, scalarThat2, false, resultBlock);
	}
	
	@Override
	public void ctableOperations(Operator op, MatrixIndexes ix1, double scalarThat, boolean left, int blen,
			CTableMap resultMap, MatrixBlock resultBlock) {
		//ctable expand (column vector to ctable)
		CTable ctable = CTable.getCTableFnObject();
		if ( resultMap != null ) {
			if( left )
				ctable.execute(ix1.getRowIndex(), this.value, scalarThat, false, resultMap);
			else
				ctable.execute(this.value, ix1.getRowIndex(), scalarThat, false, resultMap);
		} 
		else {
			if( left )
				ctable.execute(ix1.getRowIndex(), this.value, scalarThat, false, resultBlock);
			else
				ctable.execute(this.value, ix1.getRowIndex(), scalarThat, false, resultBlock);
		}
	}

	@Override
	public void ctableOperations(Operator op, double scalarThat,
			MatrixValue that2, CTableMap resultMap, MatrixBlock resultBlock) {
		MatrixCell c3=checkType(that2);
		CTable ctable = CTable.getCTableFnObject();
		if ( resultMap != null)
			ctable.execute(this.value, scalarThat, c3.value, false, resultMap);
		else 
			ctable.execute(this.value, scalarThat, c3.value, false, resultBlock);

	}

	@Override
	public void slice(ArrayList<IndexedMatrixValue> outlist,
			IndexRange range, int rowCut, int colCut, int blen, int boundaryRlen, int boundaryClen) {
		((MatrixCell)outlist.get(0).getValue()).setValue(this.value);
	}
	
	@Override
	public MatrixValue replaceOperations(MatrixValue result, double pattern, double replacement) {
		MatrixCell out = checkType(result);
		if( value == pattern || (Double.isNaN(pattern) && Double.isNaN(value)) )
			out.value = replacement;
		else
			out.value = value;
		return out;
	}

	@Override
	public MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op,
			MatrixValue result, int blen,
			MatrixIndexes indexesIn, boolean inCP) {
		return aggregateUnaryOperations(op,	result, blen,indexesIn);
	}

	@Override
	public void append(MatrixValue valueIn2, ArrayList<IndexedMatrixValue> outlist,
			int blen, boolean cbind, boolean m2IsLast, int nextNCol) {
		((MatrixCell)outlist.get(0).getValue()).setValue(this.value);
		MatrixCell c2=checkType(valueIn2);
		((MatrixCell)outlist.get(1).getValue()).setValue(c2.getValue());
	}
}

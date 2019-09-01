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


package org.tugraz.sysds.runtime.matrix.data;

import java.util.ArrayList;

import org.apache.hadoop.io.WritableComparable;
import org.tugraz.sysds.runtime.matrix.mapred.IndexedMatrixValue;
import org.tugraz.sysds.runtime.matrix.operators.AggregateOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.BinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.matrix.operators.ReorgOperator;
import org.tugraz.sysds.runtime.matrix.operators.ScalarOperator;
import org.tugraz.sysds.runtime.matrix.operators.UnaryOperator;
import org.tugraz.sysds.runtime.util.IndexRange;
import org.tugraz.sysds.runtime.util.UtilFunctions;

@SuppressWarnings("rawtypes")
public abstract class MatrixValue implements WritableComparable 
{
	static public class CellIndex {
		public int row;
		public int column;

		public CellIndex(int r, int c) {
			row = r;
			column = c;
		}

		@Override
		public boolean equals(Object that) 
		{
			if( !(that instanceof CellIndex) )
				return false;
				
			CellIndex cthat = (CellIndex) that;
			return (this.row == cthat.row && this.column == cthat.column);
		}

		@Override
		public int hashCode() {
			return UtilFunctions.longHashCode((row << 16) + column);
		}
	
		public void set(int r, int c) {
			row = r;
			column = c;
		}
		
		@Override
		public String toString() {
			return "("+row+","+column+")";
		}
	}

	public MatrixValue() {
		//do nothing
	}

	public MatrixValue(MatrixValue that) {
		copy(that);
	}

	public abstract int getNumRows();
	public abstract int getNumColumns();
	public abstract long getNonZeros();

	public abstract void setValue(int r, int c, double v);
	public abstract double getValue(int r, int c);

	public abstract boolean isInSparseFormat();
	public abstract boolean isEmpty();

	public abstract void reset();
	public abstract void reset(int rl, int cl);
	public abstract void reset(int rl, int cl, boolean sp);
	public abstract void reset(int rl, int cl, boolean sp, long nnzs);
	public abstract void reset(int rl, int cl, double v);

	public abstract void copy(MatrixValue that);
	public abstract void copy(MatrixValue that, boolean sp);
	
	public abstract MatrixValue scalarOperations(ScalarOperator op, MatrixValue result);
	
	public abstract MatrixValue binaryOperations(BinaryOperator op, MatrixValue thatValue, MatrixValue result);
	
	public abstract void binaryOperationsInPlace(BinaryOperator op, MatrixValue thatValue);
	
	public abstract MatrixValue reorgOperations(ReorgOperator op, MatrixValue result,
			int startRow, int startColumn, int length);
	
	public abstract void ctableOperations(Operator op, MatrixValue that, MatrixValue that2, CTableMap resultMap, MatrixBlock resultBlock);
	
	public abstract void ctableOperations(Operator op, MatrixValue that, double scalar_that2, boolean ignoreZeros, CTableMap resultMap, MatrixBlock resultBlock);
	
	public abstract void ctableOperations(Operator op, double scalar_that, double scalar_that2, CTableMap resultMap, MatrixBlock resultBlock);
	
	public abstract void ctableOperations(Operator op, MatrixIndexes ix1, double scalar_that, boolean left, int blen, CTableMap resultMap, MatrixBlock resultBlock);

	public abstract void ctableOperations(Operator op, double scalarThat, MatrixValue that2, CTableMap ctableResult, MatrixBlock ctableResultBlock);
	
	public abstract MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, 
		int blen, MatrixIndexes indexesIn);
	
	public abstract MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, 
		int blen, MatrixIndexes indexesIn, boolean inCP);
	
	public abstract MatrixValue unaryOperations(UnaryOperator op, MatrixValue result);
	
	public abstract void incrementalAggregate(AggregateOperator aggOp, MatrixValue correction, MatrixValue newWithCorrection, boolean deep);
	
	public abstract void incrementalAggregate(AggregateOperator aggOp, MatrixValue newWithCorrection);

	public abstract MatrixValue zeroOutOperations(MatrixValue result, IndexRange range, boolean complementary);
	
	public abstract void slice(ArrayList<IndexedMatrixValue> outlist, IndexRange range, int rowCut, int colCut, 
		int blen, int boundaryRlen, int boundaryClen);

	public abstract MatrixValue replaceOperations( MatrixValue result, double pattern, double replacement );

	public abstract void append(MatrixValue valueIn2, ArrayList<IndexedMatrixValue> outlist,
		int blen, boolean cbind, boolean m2IsLast, int nextNCol);
}

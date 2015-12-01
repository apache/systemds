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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.WritableComparable;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.QuaternaryOperator;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.runtime.util.UtilFunctions;

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

		public int hashCode() {
			return UtilFunctions.longHashFunc((row << 16) + column);
		}
	
		public void set(int r, int c) {
			row = r;
			column = c;
		}
		public String toString()
		{
			return "("+row+","+column+")";
		}
	}

	public MatrixValue() {
	}

	public MatrixValue(int rl, int cl, boolean sp) {
	}

	public MatrixValue(MatrixValue that) {
		copy(that);
	}
	
	public MatrixValue(HashMap<CellIndex, Double> map) {
		
	}

	public abstract int getNumRows();

	public abstract int getNumColumns();

	public abstract int getMaxRow() throws DMLRuntimeException;;
	
	public abstract int getMaxColumn() throws DMLRuntimeException;;

	public abstract void setMaxRow(int _r) throws DMLRuntimeException;
	
	public abstract void setMaxColumn(int _c) throws DMLRuntimeException;;

	public abstract boolean isInSparseFormat();

	public abstract boolean isEmpty();
	
	public abstract void reset();

	public abstract void reset(int rl, int cl);

	public abstract void reset(int rl, int cl, boolean sp);
	
	public abstract void reset(int rl, int cl, boolean sp, long nnzs);

	public abstract void resetDenseWithValue(int rl, int cl, double v) throws DMLRuntimeException ;

	public abstract void copy(MatrixValue that);
	public abstract void copy(MatrixValue that, boolean sp);

	public abstract long getNonZeros();

	public abstract void setValue(int r, int c, double v);

	public abstract void setValue(CellIndex index, double v);

	public abstract void addValue(int r, int c, double v);

	public abstract double getValue(int r, int c);
	
	public abstract void getCellValues(Collection<Double> ret);
	
	public abstract void getCellValues(Map<Double, Integer> ret);

	public abstract MatrixValue scalarOperations(ScalarOperator op, MatrixValue result) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue binaryOperations(BinaryOperator op, MatrixValue thatValue, MatrixValue result) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract void binaryOperationsInPlace(BinaryOperator op, MatrixValue thatValue) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue reorgOperations(ReorgOperator op, MatrixValue result,
			int startRow, int startColumn, int length)
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	// tertiary where all three inputs are matrices
	public abstract void ternaryOperations(Operator op, MatrixValue that, MatrixValue that2, CTableMap resultMap, MatrixBlock resultBlock)
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	// tertiary where first two inputs are matrices, and third input is a scalar (double)
	public abstract void ternaryOperations(Operator op, MatrixValue that, double scalar_that2, boolean ignoreZeros, CTableMap resultMap, MatrixBlock resultBlock)
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	// tertiary where first input is a matrix, and second and third inputs are scalars (double)
	public abstract void ternaryOperations(Operator op, double scalar_that, double scalar_that2, CTableMap resultMap, MatrixBlock resultBlock)
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	// tertiary where first input is a matrix, and second and third inputs are scalars (double)
	public abstract void ternaryOperations(Operator op, MatrixIndexes ix1, double scalar_that, boolean left, int brlen, CTableMap resultMap, MatrixBlock resultBlock)
	throws DMLUnsupportedOperationException, DMLRuntimeException;

	// tertiary where first and third inputs are matrices and second is a scalar
	public abstract void ternaryOperations(Operator op, double scalarThat, MatrixValue that2, CTableMap ctableResult, MatrixBlock ctableResultBlock)
	throws DMLUnsupportedOperationException, DMLRuntimeException;

	public abstract MatrixValue quaternaryOperations(QuaternaryOperator qop, MatrixValue um, MatrixValue vm, MatrixValue wm, MatrixValue out)
		throws DMLUnsupportedOperationException, DMLRuntimeException;
		
	public abstract MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, 
			int brlen, int bclen, MatrixIndexes indexesIn) throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, 
			int blockingFactorRow, int blockingFactorCol, MatrixIndexes indexesIn, boolean inCP) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue aggregateBinaryOperations(MatrixValue m1Value, MatrixValue m2Value, 
			MatrixValue result, AggregateBinaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue aggregateBinaryOperations(MatrixIndexes m1Index, MatrixValue m1Value, MatrixIndexes m2Index, MatrixValue m2Value, 
			MatrixValue result, AggregateBinaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue unaryOperations(UnaryOperator op, MatrixValue result) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract void unaryOperationsInPlace(UnaryOperator op) throws DMLUnsupportedOperationException, DMLRuntimeException;

	public abstract void incrementalAggregate(AggregateOperator aggOp, MatrixValue correction, 
			MatrixValue newWithCorrection)	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract void incrementalAggregate(AggregateOperator aggOp, MatrixValue newWithCorrection)
	throws DMLUnsupportedOperationException, DMLRuntimeException;

	public abstract MatrixValue zeroOutOperations(MatrixValue result, IndexRange range, boolean complementary)
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract void sliceOperations(ArrayList<IndexedMatrixValue> outlist, IndexRange range, int rowCut, int colCut, 
			int blockRowFactor, int blockColFactor, int boundaryRlen, int boundaryClen)
	throws DMLUnsupportedOperationException, DMLRuntimeException;

	public abstract MatrixValue replaceOperations( MatrixValue result, double pattern, double replacement )
			throws DMLUnsupportedOperationException, DMLRuntimeException;

	public abstract void appendOperations(MatrixValue valueIn2, ArrayList<IndexedMatrixValue> outlist,
			int blockRowFactor, int blockColFactor, boolean cbind, boolean m2IsLast, int nextNCol)
			throws DMLUnsupportedOperationException, DMLRuntimeException ;
}

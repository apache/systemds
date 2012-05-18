package com.ibm.bi.dml.runtime.matrix.io;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.io.WritableComparable;

import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

public abstract class MatrixValue implements WritableComparable {

	static public class CellIndex {
		public int row;
		public int column;

		public CellIndex(int r, int c) {
			row = r;
			column = c;
		}

		
		public boolean equals(CellIndex that) {
			return (this.row == that.row && this.column == that.column);
		}

		public boolean equals(Object that) {
			if (that instanceof CellIndex)
				return equals((CellIndex) that);
			else
				return false;
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
		this.copy(that);
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

	public abstract void reset();

	public abstract void reset(int rl, int cl);

	public abstract void reset(int rl, int cl, boolean sp);

	public abstract void resetDenseWithValue(int rl, int cl, double v);

	public abstract void copy(MatrixValue that);

	public abstract int getNonZeros();

	public abstract void setValue(int r, int c, double v);

	public abstract void setValue(CellIndex index, double v);

	public abstract void addValue(int r, int c, double v);

	public abstract double getValue(int r, int c);
	
	public abstract void getCellValues(Collection<Double> ret);
	
	public abstract void getCellValues(Map<Double, Integer> ret);

/*	public abstract void sparseScalarOperationsInPlace(ScalarOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;

	public abstract void denseScalarOperationsInPlace(ScalarOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;

	public abstract void sparseUnaryOperationsInPlace(UnaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
*/	
	public abstract MatrixValue scalarOperations(ScalarOperator op, MatrixValue result) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract void scalarOperationsInPlace(ScalarOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue binaryOperations(BinaryOperator op, MatrixValue thatValue, MatrixValue result) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract void binaryOperationsInPlace(BinaryOperator op, MatrixValue thatValue) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue reorgOperations(ReorgOperator op, MatrixValue result,
			int startRow, int startColumn, int length)
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	// tertiary where all three inputs are matrices
	public abstract void tertiaryOperations(Operator op, MatrixValue that, MatrixValue that2, HashMap<CellIndex, Double> ctableResult)
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	// tertiary where first two inputs are matrices, and third input is a scalar (double)
	public abstract void tertiaryOperations(Operator op, MatrixValue that, double scalar_that2, HashMap<CellIndex, Double> ctableResult)
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	// tertiary where first input is a matrix, and second and third inputs are scalars (double)
	public abstract void tertiaryOperations(Operator op, double scalar_that, double scalar_that2, HashMap<CellIndex, Double> ctableResult)
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	// tertiary where first and third inputs are matrices and second is a scalar
	public abstract void tertiaryOperations(Operator op, double scalarThat, MatrixValue that2, HashMap<CellIndex, Double> ctableResult)
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, 
			int brlen, int bclen, MatrixIndexes indexesIn) throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, 
			int blockingFactorRow, int blockingFactorCol, MatrixIndexes indexesIn, boolean inCP) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue aggregateBinaryOperations(MatrixValue m1Value, MatrixValue m2Value, 
			MatrixValue result, AggregateBinaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract MatrixValue unaryOperations(UnaryOperator op, MatrixValue result) 
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract void unaryOperationsInPlace(UnaryOperator op) throws DMLUnsupportedOperationException, DMLRuntimeException;

	/*public abstract void combineOperations(MatrixValue thatValue, CollectMultipleConvertedOutputs multipleOutputs, 
			Reporter reporter, DoubleWritable keyBuff, IntWritable valueBuff, Vector<Integer> outputIndexes) 
	throws DMLUnsupportedOperationException, DMLRuntimeException, IOException;*/
	public abstract void incrementalAggregate(AggregateOperator aggOp, MatrixValue correction, 
			MatrixValue newWithCorrection)	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract void incrementalAggregate(AggregateOperator aggOp, MatrixValue newWithCorrection)
	throws DMLUnsupportedOperationException, DMLRuntimeException;

	public abstract MatrixValue zeroOutOperations(MatrixValue result, IndexRange range, boolean complementary)
	throws DMLUnsupportedOperationException, DMLRuntimeException;
	
	public abstract void slideOperations(ArrayList<IndexedMatrixValue> outlist, IndexRange range, int rowCut, int colCut, 
			int blockRowFactor, int blockColFactor, int boundaryRlen, int boundaryClen)
	throws DMLUnsupportedOperationException, DMLRuntimeException;

	protected CellIndex tempCellIndex=new CellIndex(0, 0);
	protected void updateCtable(double v1, double v2, double w, HashMap<CellIndex, Double> ctableResult) throws DMLRuntimeException {
		int _row, _col;
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v1) || Double.isNaN(v2) || Double.isNaN(w) ) {
			return;
		}
		else {
			_row = (int)v1;
			_col = (int)v2;
			
			if ( _row <= 0 || _col <= 0 ) {
				throw new DMLRuntimeException("Erroneous input while computing the contingency table (one of the value <= zero).");
			} 
			CellIndex temp=new CellIndex(_row, _col);
			Double oldw=ctableResult.get(temp);
			if(oldw==null)
				oldw=0.0;
			ctableResult.put(temp, oldw+w);
		}
	}
}

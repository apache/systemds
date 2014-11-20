/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.io.WritableComparable;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.CTable;
import com.ibm.bi.dml.runtime.functionobjects.ReduceDiag;
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

@SuppressWarnings("rawtypes")
public class MatrixCell extends MatrixValue implements WritableComparable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected double value;
	
	public MatrixCell(MatrixCell that)
	{
		this.value=that.value;
	}
	
	public MatrixCell()
	{
		value=0;
	}
	
	public MatrixCell(MatrixValue that) {
		if(that instanceof MatrixCell)
			this.value=((MatrixCell)that).value;
	}

	public MatrixCell(double v) {
		value=v;
	}

	private static MatrixCell checkType(MatrixValue cell) throws DMLUnsupportedOperationException
	{
		if( cell!=null && !(cell instanceof MatrixCell))
			throw new DMLUnsupportedOperationException("the Matrix Value is not MatrixCell!");
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
	public void copy(MatrixValue that){
		MatrixCell c2=MatrixCell.class.cast(that);
		if(c2==null)
			throw new RuntimeException(that+" is not of type MatrixCell");
		value=c2.getValue();
	}

	@Override
	public int getNonZeros() {
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
	
	public void reset(int rl, int cl, boolean sp, int nnzs) {		
		value=0;
	}

	@Override
	public void resetDenseWithValue(int rl, int cl, double v) {
		value=0;
	}

	@Override
	public void setValue(int r, int c, double v) {
		value=v;	
	}

	@Override
	public void setValue(CellIndex index, double v) {
		value=v;
	}

	@Override
	public void addValue(int r, int c, double v) {
		value += v;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(value);
	}

	public String toString()
	{
		return Double.toString(value);
	}

	@Override
	public MatrixValue aggregateBinaryOperations(MatrixValue value1,
			MatrixValue value2, MatrixValue result, AggregateBinaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		MatrixCell c1=checkType(value1);
		MatrixCell c2=checkType(value2);
		MatrixCell c3=checkType(result);
		if(c3==null)
			c3=new MatrixCell();
		c3.setValue(op.binaryFn.execute(c1.getValue(), c2.getValue()));
		return c3;
	}

	@Override
	public MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op,
			MatrixValue result, int brlen, int bclen,
			MatrixIndexes indexesIn) throws DMLUnsupportedOperationException {
		
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
			MatrixValue thatValue, MatrixValue result)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		MatrixCell c2=checkType(thatValue);
		MatrixCell c3=checkType(result);
		if(c3==null)
			c3=new MatrixCell();
		c3.setValue(op.fn.execute(this.getValue(), c2.getValue()));
		return c3;
	}

	@Override
	public void binaryOperationsInPlace(BinaryOperator op,
			MatrixValue thatValue)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		MatrixCell c2=checkType(thatValue);
		setValue(op.fn.execute(this.getValue(), c2.getValue()));
		
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
		
		MatrixCell c3=checkType(result);
		if(c3==null)
			c3=new MatrixCell();
		c3.setValue(getValue());
		return c3;
	}

	@Override
	public MatrixValue scalarOperations(ScalarOperator op, MatrixValue result) throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		MatrixCell c3=checkType(result);
		c3.setValue(op.fn.execute(value, op.getConstant()));
		return c3;
	}

	@Override
	public void scalarOperationsInPlace(ScalarOperator op)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		value=op.executeScalar(value);
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
		MatrixCell c3=checkType(result);
		c3.setValue(op.fn.execute(value));
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

	public int compareTo(MatrixCell o) {
		return Double.compare(this.value, o.value);
	}

	@Override
	public int compareTo(Object o) {
		if(o instanceof MatrixCell)
			return Double.compare(this.value, ((MatrixCell) o).value);
		else
			throw new RuntimeException("MatrixCell cannot compare with "+o.getClass());
	}

	@Override
	public int getMaxColumn() throws DMLRuntimeException {
		throw new DMLRuntimeException("getMaxColumn() can not be executed on cells");
	}

	@Override
	public int getMaxRow() throws DMLRuntimeException {
		throw new DMLRuntimeException("getMaxRow() can not be executed on cells");
	}

	@Override
	public void setMaxColumn(int c) throws DMLRuntimeException {
		throw new DMLRuntimeException("setMaxColumn() can not be executed on cells");
	}

	@Override
	public void setMaxRow(int r) throws DMLRuntimeException {
		throw new DMLRuntimeException("setMaxRow() can not be executed on cells");
	}

/*	public void combineOperations(MatrixValue thatValue, CollectMultipleConvertedOutputs multipleOutputs, 
			Reporter reporter, DoubleWritable keyBuff, IntWritable valueBuff, Vector<Integer> outputIndexes)
	throws DMLUnsupportedOperationException, DMLRuntimeException, IOException
	{
		MatrixCell c2=checkType(thatValue);
		keyBuff.set(this.value);
		valueBuff.set((int)c2.getValue());
		for(int i: outputIndexes)
			multipleOutputs.collectOutput(keyBuff, valueBuff, i, reporter);
	}*/

	@Override
	public void incrementalAggregate(AggregateOperator aggOp,
			MatrixValue correction, MatrixValue newWithCorrection)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("MatrixCell.incrementalAggregate should never be called");
		/*
		MatrixCell cor=checkType(correction);
		MatrixCell newWithCor=checkType(newWithCorrection);
		KahanObject buffer=new KahanObject(value, cor.value);
		buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.value);
		value=buffer._sum;
		cor.value=buffer._correction;*/
	}

	@Override
	public MatrixValue zeroOutOperations(MatrixValue result, IndexRange range, boolean complementary)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		if(range.rowStart!=0 || range.rowEnd!=0 || range.colStart!=0 || range.colEnd!=0)
			throw new DMLRuntimeException("wrong range: "+range+" for matrixCell");
		MatrixCell c3=checkType(result);
		c3.setValue(value);
		return c3;
	}

	@Override
	public void incrementalAggregate(AggregateOperator aggOp,
			MatrixValue newWithCorrection)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("MatrixCell.incrementalAggregate should never be called");
	}

	@Override
	public void tertiaryOperations(Operator op, MatrixValue that,
			MatrixValue that2, HashMap<MatrixIndexes, Double> ctableResult, MatrixBlock ctableResultBlock)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		MatrixCell c2=checkType(that);
		MatrixCell c3=checkType(that2);
		CTable ctable = CTable.getCTableFnObject();
		if ( ctableResult != null)
			ctable.execute(this.value, c2.value, c3.value, ctableResult);
		else
			ctable.execute(this.value, c2.value, c3.value, ctableResultBlock);
	}

	@Override
	public void tertiaryOperations(Operator op, MatrixValue that,
			double scalarThat2, HashMap<MatrixIndexes, Double> ctableResult, MatrixBlock ctableResultBlock)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		MatrixCell c2=checkType(that);
		CTable ctable = CTable.getCTableFnObject();
		if ( ctableResult != null)
			ctable.execute(this.value, c2.value, scalarThat2, ctableResult);		
		else
			ctable.execute(this.value, c2.value, scalarThat2, ctableResultBlock);		
	}

	@Override
	public void tertiaryOperations(Operator op, double scalarThat,
			double scalarThat2, HashMap<MatrixIndexes, Double> ctableResult, MatrixBlock ctableResultBlock)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		CTable ctable = CTable.getCTableFnObject();
		if ( ctableResult != null)
			ctable.execute(this.value, scalarThat, scalarThat2, ctableResult);		
		else
			ctable.execute(this.value, scalarThat, scalarThat2, ctableResultBlock);		
	}
	
	@Override
	public void tertiaryOperations(Operator op, MatrixIndexes ix1, double scalarThat, boolean left, int brlen,
			HashMap<MatrixIndexes, Double> ctableResult, MatrixBlock ctableResultBlock)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//ctable expand (column vector to ctable)
		CTable ctable = CTable.getCTableFnObject();
		if ( ctableResult != null ) {
			if( left )
				ctable.execute(ix1.getRowIndex(), this.value, scalarThat, ctableResult);		
			else
				ctable.execute(this.value, ix1.getRowIndex(), scalarThat, ctableResult);			
		} 
		else {
			if( left )
				ctable.execute(ix1.getRowIndex(), this.value, scalarThat, ctableResultBlock);		
			else
				ctable.execute(this.value, ix1.getRowIndex(), scalarThat, ctableResultBlock);			
		}
	}

	@Override
	public void tertiaryOperations(Operator op, double scalarThat,
			MatrixValue that2, HashMap<MatrixIndexes, Double> ctableResult, MatrixBlock ctableResultBlock)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		MatrixCell c3=checkType(that2);
		CTable ctable = CTable.getCTableFnObject();
		if ( ctableResult != null)
			ctable.execute(this.value, scalarThat, c3.value, ctableResult);
		else 
			ctable.execute(this.value, scalarThat, c3.value, ctableResultBlock);

	}

	@Override
	public void sliceOperations(ArrayList<IndexedMatrixValue> outlist,
			IndexRange range, int rowCut, int colCut, int blockRowFactor,
			int blockColFactor, int boundaryRlen, int boundaryClen)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		((MatrixCell)outlist.get(0).getValue()).setValue(this.value);
		
	}
	
	@Override
	public MatrixValue replaceOperations(MatrixValue result, double pattern, double replacement)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		MatrixCell out = checkType(result);		
		if( value == pattern || (Double.isNaN(pattern) && Double.isNaN(value)) )
			out.value = replacement;
		else
			out.value = value;
		return out;
	}

	@Override
	public MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op,
			MatrixValue result, int blockingFactorRow, int blockingFactorCol,
			MatrixIndexes indexesIn, boolean inCP)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		return aggregateUnaryOperations(op,	result, blockingFactorRow, blockingFactorCol,indexesIn);
	}

	@Override
	public MatrixValue aggregateBinaryOperations(MatrixIndexes m1Index,
			MatrixValue m1Value, MatrixIndexes m2Index, MatrixValue m2Value,
			MatrixValue result, AggregateBinaryOperator op, boolean partialMult)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("MatrixCell.aggregateBinaryOperations should never be called");
	}

	@Override
	public void appendOperations(MatrixValue valueIn2, ArrayList<IndexedMatrixValue> outlist,
			int blockRowFactor, int blockColFactor, boolean m2IsLast, int nextNCol) 
	throws DMLUnsupportedOperationException, DMLRuntimeException  {
		((MatrixCell)outlist.get(0).getValue()).setValue(this.value);
		MatrixCell c2=checkType(valueIn2);
		((MatrixCell)outlist.get(1).getValue()).setValue(c2.getValue());	
	}

}

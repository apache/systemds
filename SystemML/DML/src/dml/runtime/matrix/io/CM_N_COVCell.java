package dml.runtime.matrix.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.io.WritableComparable;
import dml.runtime.instructions.CPInstructions.CM_COV_Object;
import dml.runtime.instructions.MRInstructions.SelectInstruction.IndexRange;
import dml.runtime.matrix.operators.AggregateBinaryOperator;
import dml.runtime.matrix.operators.AggregateOperator;
import dml.runtime.matrix.operators.AggregateUnaryOperator;
import dml.runtime.matrix.operators.BinaryOperator;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.ReorgOperator;
import dml.runtime.matrix.operators.ScalarOperator;
import dml.runtime.matrix.operators.UnaryOperator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class CM_N_COVCell extends MatrixValue implements WritableComparable{

	private CM_COV_Object cm=new CM_COV_Object();
	
	public String toString()
	{
		return cm.toString();
	}
	
	@Override
	public void addValue(int r, int c, double v) {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public MatrixValue aggregateBinaryOperations(MatrixValue m1Value,
			MatrixValue m2Value, MatrixValue result, AggregateBinaryOperator op)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op,
			MatrixValue result, int brlen, int bclen, MatrixIndexes indexesIn)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public MatrixValue binaryOperations(BinaryOperator op,
			MatrixValue thatValue, MatrixValue result)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void binaryOperationsInPlace(BinaryOperator op, MatrixValue thatValue)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");		
	}

	@Override
	public void copy(MatrixValue that) {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void getCellValues(Collection<Double> ret) {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void getCellValues(Map<Double, Integer> ret) {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public int getMaxColumn() throws DMLRuntimeException {
		return 1;
	}

	@Override
	public int getMaxRow() throws DMLRuntimeException {
		return 1;
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
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void incrementalAggregate(AggregateOperator aggOp,
			MatrixValue correction, MatrixValue newWithCorrection)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}
	
	@Override
	public void incrementalAggregate(AggregateOperator aggOp,
			MatrixValue newWithCorrection)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public boolean isInSparseFormat() {
		return false;
	}

	@Override
	public MatrixValue reorgOperations(ReorgOperator op, MatrixValue result,
			int startRow, int startColumn, int length)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void reset() {
	}

	@Override
	public void reset(int rl, int cl) {
		
	}

	@Override
	public void reset(int rl, int cl, boolean sp) {
	}

	@Override
	public void resetDenseWithValue(int rl, int cl, double v) {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public MatrixValue scalarOperations(ScalarOperator op, MatrixValue result)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void scalarOperationsInPlace(ScalarOperator op)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void setMaxColumn(int c) throws DMLRuntimeException {
		
	}

	@Override
	public void setMaxRow(int r) throws DMLRuntimeException {
		
	}

	@Override
	public void setValue(int r, int c, double v) {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void setValue(CellIndex index, double v) {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public MatrixValue unaryOperations(UnaryOperator op, MatrixValue result)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void unaryOperationsInPlace(UnaryOperator op)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		cm.w=in.readDouble();
		cm.mean.read(in);
		cm.m2.read(in);
		cm.m3.read(in);
		cm.m4.read(in);
		cm.mean_v.read(in);
		cm.c2.read(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(cm.w);
		cm.mean.write(out);
		cm.m2.write(out);
		cm.m3.write(out);
		cm.m4.write(out);
		cm.mean_v.write(out);
		cm.c2.write(out);
	}

	@Override
	public int compareTo(Object o) {
		if(o instanceof CM_N_COVCell)
		{
			CM_N_COVCell that=(CM_N_COVCell)o;
			return cm.compareTo(that.cm);
		}
		return -1;
	}
	
	public CM_COV_Object getCM_N_COVObject()
	{
		return cm;
	}

	public void setCM_N_COVObject(double u, double v, double w)
	{
		cm.w=w;
		cm.mean.set(u,0);
		cm.mean_v.set(v, 0);
		cm.m2.set(0,0);
		cm.m3.set(0,0);
		cm.m4.set(0,0);
		cm.c2.set(0,0);
	}
	public void setCM_N_COVObject(CM_COV_Object that)
	{
		cm.set(that);
	}

	@Override
	public MatrixValue selectOperations(MatrixValue valueOut, IndexRange range)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void tertiaryOperations(Operator op, MatrixValue that,
			MatrixValue that2,
			HashMap<CellIndex, Double> ctableResult)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
		
	}

	@Override
	public void tertiaryOperations(Operator op, MatrixValue that,
			double scalarThat2,
			HashMap<CellIndex, Double> ctableResult)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void tertiaryOperations(Operator op, double scalarThat,
			double scalarThat2,
			HashMap<CellIndex, Double> ctableResult)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}

	@Override
	public void tertiaryOperations(Operator op, double scalarThat,
			MatrixValue that2,
			HashMap<CellIndex, Double> ctableResult)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new RuntimeException("operation not supported fro WeightedCell");
	}
}

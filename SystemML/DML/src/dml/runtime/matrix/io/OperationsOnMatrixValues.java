package dml.runtime.matrix.io;

import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import dml.runtime.functionobjects.Builtin;
import dml.lops.PartialAggregate.CorrectionLocationType;
import dml.runtime.instructions.MRInstructions.SelectInstruction;
import dml.runtime.matrix.io.MatrixValue.CellIndex;
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

public class OperationsOnMatrixValues {

	private static IndexedCell tempCell1=new IndexedCell();
	private static IndexedCell tempCell2=new IndexedCell();
	private static Log LOG=LogFactory.getLog(OperationsOnMatrixValues.class);
	//private static boolean sparseSafte=false;
	
	public static void performScalarIgnoreIndexes(MatrixValue value_in, MatrixValue value_out, ScalarOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_in.scalarOperations(op, value_out);
	}
	
	public static void performScalarIgnoreIndexesInPlace(MatrixValue value_in, ScalarOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_in.scalarOperationsInPlace(op);
	}
	
	public static void performUnaryIgnoreIndexes(MatrixValue value_in, MatrixValue value_out, UnaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_in.unaryOperations(op, value_out);
	}
	
	public static void performUnaryIgnoreIndexesInPlace(MatrixValue value_in, UnaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_in.unaryOperationsInPlace(op);
	}

/*	public static void performBuiltinIgnoreIndexes(MatrixValue value_in, double constant, 
			MatrixValue value_out, Builtin.SupportedOperation op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		boolean sparseSafe=false;
		if(op==Builtin.SupportedOperation.ABS || op==Builtin.SupportedOperation.SIN 
				|| op==Builtin.SupportedOperation.SQRT || op==Builtin.SupportedOperation.TAN)
			sparseSafe=true;
		value_in.builtinOperations(op, constant, value_out, sparseSafe);
	}*/

	
	public static void performReorg(MatrixIndexes indexes_in, MatrixValue value_in, 
			MatrixIndexes indexes_out, MatrixValue value_out, ReorgOperator op, 
			int startRow, int startColumn, int length) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operate on the value indexes first
		op.fn.execute(indexes_in, indexes_out);
		
		//operation on the cells inside the value
		value_out=value_in.reorgOperations(op, value_out, startRow, startColumn, length);
		
	}
	
	public static void performAppend(MatrixIndexes indexes_in, MatrixValue value_in, 
			MatrixIndexes indexes_out, MatrixValue value_out, ReorgOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operates only on indexes
		op.fn.execute(indexes_in, indexes_out);
		
		//operation on the cells inside the value
		value_out=value_in.reorgOperations(op, value_out, 0, 0, 0);
	}
	
	public static void performSelect(MatrixIndexes indexes_in, MatrixValue value_in, 
			MatrixIndexes indexes_out, MatrixValue value_out, SelectInstruction.IndexRange range) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_out=value_in.selectOperations(value_out, range);
	}
	
	// ------------- Tertiary Operations -------------
	// tertiary where all three inputs are matrices
	public static void performTertiary(MatrixIndexes indexes_in1, MatrixValue value_in1, MatrixIndexes indexes_in2, MatrixValue value_in2, 
			MatrixIndexes indexes_in3, MatrixValue value_in3, HashMap<CellIndex, Double> ctableResult, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		value_in1.tertiaryOperations(op, value_in2, value_in3, ctableResult);
	}
	
	// tertiary where first two inputs are matrices, and third input is a scalar (double)
	public static void performTertiary(MatrixIndexes indexes_in1, MatrixValue value_in1, MatrixIndexes indexes_in2, MatrixValue value_in2, 
			double scalar_in3, HashMap<CellIndex, Double> ctableResult, Operator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		value_in1.tertiaryOperations(op, value_in2, scalar_in3, ctableResult);
	}
	
	// tertiary where first input is a matrix, and second and third inputs are scalars (double)
	public static void performTertiary(MatrixIndexes indexes_in1, MatrixValue value_in1, double scalar_in2, 
			double scalar_in3, HashMap<CellIndex, Double> ctableResult, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		value_in1.tertiaryOperations(op, scalar_in2, scalar_in3, ctableResult);
	}
	
	// tertiary where first and third inputs are matrices, and second is a scalars (double)
	public static void performTertiary(MatrixIndexes indexes_in1, MatrixValue value_in1, double scalar_in2, 
			MatrixIndexes indexes_in3, MatrixValue value_in3, HashMap<CellIndex, Double> ctableResult, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		value_in1.tertiaryOperations(op, scalar_in2, value_in3, ctableResult);
	}
	// -----------------------------------------------------
	
	//binary operations are those that the indexes of both cells have to be matched
	public static void performBinaryIgnoreIndexes(MatrixValue value1, MatrixValue value2, 
			MatrixValue value_out, BinaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_out=value1.binaryOperations(op, value2, value_out);
	}
	
	public static void startAggregation(MatrixValue value_out, MatrixValue correction, AggregateOperator op, 
			int rlen, int clen, boolean sparseHint, boolean imbededCorrection)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		int outRow=0, outCol=0, corRow=0, corCol=0;
		if(op.correctionExists)
		{
			if(!imbededCorrection)
			{
				switch(op.correctionLocation)
				{
				case NONE:
					outRow=rlen;
					outCol=clen;
					corRow=rlen;
					corCol=clen;
					break;
				case LASTROW:
					outRow=rlen-1;
					outCol=clen;
					corRow=1;
					corCol=clen;
					break;
				case LASTCOLUMN:
					if(op.increOp.fn instanceof Builtin 
					   && ((Builtin)(op.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MAXINDEX ){
						outRow = rlen;
						outCol = 1;
						corRow = rlen;
						corCol = 1;
					}else{
						outRow=rlen;
						outCol=clen-1;
						corRow=rlen;
						corCol=1;
					}
					break;
				case LASTTWOROWS:
					outRow=rlen-2;
					outCol=clen;
					corRow=2;
					corCol=clen;
					break;
				case LASTTWOCOLUMNS:
					outRow=rlen;
					outCol=clen-2;
					corRow=rlen;
					corCol=2;
					break;
				default:
						throw new DMLRuntimeException("unrecognized correctionLocation: "+op.correctionLocation);
				}
			}else
			{
				outRow=rlen;
				outCol=clen;
				corRow=rlen;
				corCol=clen;
			}
			
	/*		if(op.correctionLocation==1)
			{
				outRow=rlen-1;
				outCol=clen;
				corRow=1;
				corCol=clen;
			}else if(op.correctionLocation==2)
			{
				outRow=rlen;
				outCol=clen-1;
				corRow=rlen;
				corCol=1;
			}else if(op.correctionLocation==0)
			{
				outRow=rlen;
				outCol=clen;
				corRow=rlen;
				corCol=clen;
			}
			else
				throw new DMLRuntimeException("unrecognized correctionLocation: "+op.correctionLocation);	*/
			
			value_out.reset(outRow, outCol, sparseHint);
			correction.reset(corRow, corCol, false);
			
		}else
		{
			if(op.initialValue==0)
				value_out.reset(rlen, clen, sparseHint);
			else
				value_out.resetDenseWithValue(rlen, clen, op.initialValue);
		}
	}
	
	public static void incrementalAggregation(MatrixValue value_agg, MatrixValue correction, MatrixValue value_add, 
			AggregateOperator op, boolean imbededCorrection) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(op.correctionExists)
		{
			if(!imbededCorrection || op.correctionLocation==CorrectionLocationType.NONE)
				value_agg.incrementalAggregate(op, correction, value_add);
			else
				value_agg.incrementalAggregate(op, value_add);
		}
		else
			value_agg.binaryOperationsInPlace(op.increOp, value_add);
	}
	
	public static void performRandUnary(MatrixIndexes indexes_in, MatrixValue value_in, 
			MatrixIndexes indexes_out, MatrixValue value_out, int brlen, int bclen)
	{
		indexes_out.setIndexes(indexes_in);
		value_out.copy(value_in);
	}
	
	public static void performAggregateUnary(MatrixIndexes indexes_in, MatrixValue value_in, 
			MatrixIndexes indexes_out, MatrixValue value_out, AggregateUnaryOperator op,
			int brlen, int bclen)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operate on the value indexes first
		op.indexFn.execute(indexes_in, indexes_out);
		
		//MatrixValue value_tmp = null ;
		//value_in.predicateOperations(indexes_in, value_tmp, op.getPredicateOperation(), brlen, bclen) ;
		
		//TODO: cannot handle trace
		//perform on the value
		value_out=value_in.aggregateUnaryOperations(op, value_out, brlen, bclen, indexes_in);
	}
	
	public static void performAggregateBinary(MatrixIndexes indexes1, MatrixValue value1, MatrixIndexes indexes2, MatrixValue value2, 
			MatrixIndexes indexes_out, MatrixValue value_out, AggregateBinaryOperator op)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operate on the value indexes first
		//tempCell1.setIndexes(indexes1);
		//tempCell2.setIndexes(indexes2);
		//CellOperations.performMMCJIndexOnly(tempCell1, tempCell2, tempCell1);
		indexes_out.setIndexes(indexes1.getRowIndex(), indexes2.getColumnIndex());
		
		//perform on the value
		value_out=value1.aggregateBinaryOperations(value1, value2, value_out, op);
	}
	
	//including scalar, reorg and aggregateUnary operations
/*	public static void performAllUnary(MatrixIndexes indexes_in, MatrixValue value_in, 
			MatrixIndexes indexes_out, MatrixValue value_out, Instruction ins) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(ins instanceof Scalar.InstructionType)
		{
			performScalarIgnoreIndexes(value_in, ((Scalar.InstructionType) ins).constant, 
					value_out, (Scalar.SupportedOperation)ins.operation);
			indexes_out.setIndexes(indexes_in);
		}
		else if(ins instanceof Reorg.InstructionType)
			performReorg(indexes_in, value_in, indexes_out, value_out, 
					(Reorg.SupportedOperation)ins.operation);
		else if(ins instanceof AggregateUnary.InstructionType)
			performAggregateUnary(indexes_in, value_in, indexes_out, value_out, 
					(AggregateUnary.SupportedOperation)ins.operation,
					numRowsInBlock, numColsInBlock);
		else
			throw new DMLUnsupportedOperationException("Operation unsupported");
	}*/

	public static void performAggregateBinaryIgnoreIndexes(
			MatrixValue value1, MatrixValue value2,
			MatrixValue value_out, AggregateBinaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException {
			
		//perform on the value
		value_out=value1.aggregateBinaryOperations(value1, value2, value_out, op);
	}
	
	public static void main(String[] args)
	{
/*		boolean sparse1=false, sparse23=true;
		
		MatrixIndexes ind1=new MatrixIndexes(1,2);
		MatrixBlock m1=new MatrixBlock(3, 2, sparse1);
		m1.setValue(0, 0, 1);
		m1.setValue(0, 1, 2);
		m1.setValue(1, 0, 3);
		m1.setValue(1, 1, 4);
		m1.setValue(2, 0, 5);
		m1.setValue(2, 1, 6);
		System.out.println("matrix m1: ");
		ind1.print();
		m1.print();
		
		MatrixIndexes ind2=new MatrixIndexes(1,2);
		MatrixBlock m2=new MatrixBlock(3, 2, sparse23);
		m2.setValue(0, 0, 6);
		m2.setValue(0, 1, 5);
		m2.setValue(1, 0, 4);
		m2.setValue(1, 1, 3);
		m2.setValue(2, 0, 2);
		m2.setValue(2, 1, 1);
		System.out.println("matrix m2: ");
		ind2.print();
		m2.print();
		
		
		MatrixIndexes ind3=new MatrixIndexes(2, 3);
		MatrixBlock m3=new MatrixBlock(2, 3, sparse23);
		m3.setValue(0, 0, 6);
		m3.setValue(0, 1, 5);
		m3.setValue(0, 2, 4);
		m3.setValue(1, 0, 3);
		m3.setValue(1, 1, 2);
		m3.setValue(1, 2, 1);
		System.out.println("matrix m3:");
		ind3.print();
		m3.print();
	
		MatrixBlock m4=new MatrixBlock();
		MatrixIndexes ind4=new MatrixIndexes();
		
		try {
			System.out.println("--------------------------------");
			System.out.println("m4=col_sum(m1)");
			//performAggregateUnary(ind1, m1, ind4, m4, AggregateUnary.SupportedOperation.AGU_COLUMN_SUM);
			ind4.print();
			m4.print();
			
			System.out.println("--------------------------------");
			System.out.println("m4=m1 %*% m3");
			performAggregateBinary(ind1, m1, ind3, m3, ind4, m4, AggregateBinary.SupportedOperation.AGB_MMULT);
			ind4.print();
			m4.print();
			
			System.out.println("--------------------------------");
			System.out.println("prepare m4 for product");
			m4.reset(3, 2, false);
			startAggregation(m4, Aggregate.SupportedOperation.AGG_PRODUCT, 3, 2, false);
			m4.print();
			
			System.out.println("--------------------------------");
			System.out.println("product");
			incrementalAggregation(m4, m1, Aggregate.SupportedOperation.AGG_PRODUCT);
			m4.print();
			
		} catch (Exception e) {
			e.printStackTrace();
		}*/
	}

	static int numRowsInBlock, numColsInBlock ;
	public static void setNumRowsInBlock(int i) {
		numRowsInBlock = i ;
	}
	public static void setNumColumnsInBlock(int i) {
		numColsInBlock = i;
	}
}

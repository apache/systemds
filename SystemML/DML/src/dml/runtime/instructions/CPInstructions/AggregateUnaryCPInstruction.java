package dml.runtime.instructions.CPInstructions;

import dml.runtime.functionobjects.Builtin;
import dml.lops.PartialAggregate.CorrectionLocationType;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.runtime.functionobjects.KahanPlus;
import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.Plus;
import dml.runtime.functionobjects.ReduceAll;
import dml.runtime.functionobjects.ReduceCol;
import dml.runtime.functionobjects.ReduceRow;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MatrixDimensionsMetaData;
import dml.runtime.matrix.operators.AggregateOperator;
import dml.runtime.matrix.operators.AggregateUnaryOperator;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.SimpleOperator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class AggregateUnaryCPInstruction extends UnaryCPInstruction{
	public AggregateUnaryCPInstruction(Operator op, CPOperand in, CPOperand out, String istr){
		super(op, in, out, istr);
		cptype = CPINSTRUCTION_TYPE.AggregateUnary;		
	}
	
	public static Instruction parseInstruction(String str)
		throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseUnaryInstruction(str, in, out);
		
		if(opcode.equalsIgnoreCase("nrow") || opcode.equalsIgnoreCase("ncol") || opcode.equalsIgnoreCase("length")){
			return new AggregateUnaryCPInstruction(new SimpleOperator(Builtin.getBuiltinFnObject(opcode)),
												   in, 
												   out, 
												   str);
		}
		else if ( opcode.equalsIgnoreCase("uak+") ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uark+") ) {
			// RowSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uack+") ) {
			// ColSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTROW);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("uamean") ) {
			// Mean
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTTWOCOLUMNS);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uarmean") ) {
			// RowMeans
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTTWOCOLUMNS);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uacmean") ) {
			// ColMeans
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTTWOROWS);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("ua+") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uar+") ) {
			// RowSums
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uac+") ) {
			// ColSums
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		}
		
		else if ( opcode.equalsIgnoreCase("ua*") ) {
			AggregateOperator agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uamax") ) {
			AggregateOperator agg = new AggregateOperator(Double.MIN_VALUE, Builtin.getBuiltinFnObject("max"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
			// return new AggregateUnaryCPInstruction(new BinaryOperator(Builtin.getBuiltinFnObject("max")), in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uamin") ) {
			AggregateOperator agg = new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject("min"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
			// return new AggregateUnaryCPInstruction(new BinaryOperator(Builtin.getBuiltinFnObject("min")), in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("uatrace") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			aggun.isTrace=true;
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("uaktrace") ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			aggun.isTrace=true;
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("rdiagM2V") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			aggun.isDiagM2V=true;
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uarmax") ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("max"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		
		else if (opcode.equalsIgnoreCase("uarimax") ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("maxindex"), true, CorrectionLocationType.LASTCOLUMN);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		}
		
		else if ( opcode.equalsIgnoreCase("uarmin") ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("min"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uacmax") ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("max"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("uacmin") ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("min"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
			return new AggregateUnaryCPInstruction(aggun, in, out, str);
		} 
		
		return null;
	}
	
	public Data processInstruction (ProgramBlock pb)
		throws DMLRuntimeException, DMLUnsupportedOperationException{
		String output_name = output.get_name();
		
		String opcode = InstructionUtils.getOpCode(instString);
		if(opcode.equalsIgnoreCase("nrow")){
			MatrixDimensionsMetaData dims = (MatrixDimensionsMetaData)(pb.getMetaData(input1.get_name()));
			ScalarObject ret = null;
			if ( output.get_valueType() == ValueType.INT ) {
				ret = new IntObject(output_name, (int) dims.getMatrixCharacteristics().get_rows());
			}
			else if ( output.get_valueType() == ValueType.DOUBLE ) {
				ret = new DoubleObject(output_name, dims.getMatrixCharacteristics().get_rows());
			}
			pb.setVariable(output_name, ret);
			return ret;
			
/*			if ( output.get_valueType() == ValueType.INT ) {
				IntObject ret = new IntObject(output_name, (int) dims.getMatrixCharacteristics().get_rows());
				pb.setVariable(output_name, ret);
				return ret;
			}
			else if ( output.get_valueType() == ValueType.DOUBLE ) {
				DoubleObject ret = new DoubleObject(output_name, dims.getMatrixCharacteristics().get_rows());
				pb.setVariable(output_name, ret);
				return ret;
			}
*/		}
		else if(opcode.equalsIgnoreCase("ncol")){
			MatrixDimensionsMetaData dims = (MatrixDimensionsMetaData)(pb.getMetaData(input1.get_name()));
			ScalarObject ret = null;
			if ( output.get_valueType() == ValueType.INT ) {
				ret = new IntObject(output_name, (int) dims.getMatrixCharacteristics().get_cols());
			}
			else if ( output.get_valueType() == ValueType.DOUBLE ) {
				ret = new DoubleObject(output_name, dims.getMatrixCharacteristics().get_cols());
			}
			pb.setVariable(output_name, ret);
			return ret;
/*			DoubleObject ret = new DoubleObject(output_name, dims.getMatrixCharacteristics().get_cols());
			pb.setVariable(output_name, ret);
			return ret;
*/		}
		else if(opcode.equalsIgnoreCase("length")){
			MatrixDimensionsMetaData dims = (MatrixDimensionsMetaData)(pb.getMetaData(input1.get_name()));
			ScalarObject ret = null;
			if ( output.get_valueType() == ValueType.INT ) {
				ret = new IntObject(output_name, (int) (dims.getMatrixCharacteristics().get_cols()
						 * dims.getMatrixCharacteristics().get_rows()));
			}
			else if ( output.get_valueType() == ValueType.DOUBLE ) {
				ret = new DoubleObject(output_name, dims.getMatrixCharacteristics().get_cols()
						 * dims.getMatrixCharacteristics().get_rows());
			}
			pb.setVariable(output_name, ret);
			return ret;
/*			DoubleObject ret = new DoubleObject(output_name, dims.getMatrixCharacteristics().get_cols()
															 * dims.getMatrixCharacteristics().get_rows());
			pb.setVariable(output_name, ret);
			return ret;
*/		}
		
		MatrixObject mat = pb.getMatrixVariable(input1.get_name());
		AggregateUnaryOperator au_op = (AggregateUnaryOperator) optr;
		
		if(output.get_dataType() == DataType.SCALAR){
			MatrixObject temp = new MatrixObject();
			temp.setMetaData(new MatrixDimensionsMetaData(new MatrixCharacteristics()));
			temp = mat.aggregateUnaryOperations(au_op, temp);
			DoubleObject ret = new DoubleObject(output_name, temp.getValue(0, 0));
			pb.setVariable(output_name, ret);
			return ret;
		}else{
			MatrixObject sores = mat.aggregateUnaryOperations(au_op, (MatrixObject)pb.getVariable(output_name));
			pb.setVariableAndWriteToHDFS(output_name, sores);
			return sores;
		}
	}
}

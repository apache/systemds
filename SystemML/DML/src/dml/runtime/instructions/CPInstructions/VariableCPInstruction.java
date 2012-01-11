package dml.runtime.instructions.CPInstructions;

import java.io.IOException;

import dml.parser.Expression.ValueType;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.InputInfoMetaData;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MetaData;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import dml.runtime.util.MapReduceTool;
import dml.runtime.util.UtilFunctions;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class VariableCPInstruction extends CPInstruction {

	/*
	 * Supported Operations
	 * --------------------
	 *	1) assignvar x:type y:type
	 *	    assign value of y to x (both types should match)
	 *	2) rmvar x
	 *	    remove variable x
	 *	3) mvvar x y
	 *	    rename x as y (same as assignvar followed by rmvar, types are not required)
	 *	4) rmfilevar x:type b:type
	 *	    remove variable x, and if b=true then the file object associated with x (b's type should be boolean)
	 *	5) assignvarwithfile FP x
	 *	    assign x with the first value in file FP
	 *	6) attachfiletovar FP x
	 *	    allocate a new file object with name FP, and associate it with variable x
	 */
	
	private enum VariableOperationCode {
		AssignVariable, RemoveVariable, RenameVariable, RemoveVariableAndFile, AssignVariableWithFirstValue, AttachFileToVariable, ValuePickCP, IQSize, SpearmanHelper, writeScalar, readScalar
	}
	
	VariableOperationCode opcode;
	private CPOperand input1;
	private CPOperand input2;
	private CPOperand output;
	int arity;
	
	private static VariableOperationCode getVariableOperationCode ( String str ) throws DMLUnsupportedOperationException {
		
		if ( str.equalsIgnoreCase("assignvar"))
			return VariableOperationCode.AssignVariable;
		
		else if ( str.equalsIgnoreCase("rmvar") ) 
			return VariableOperationCode.RemoveVariable;
		
		else if ( str.equalsIgnoreCase("mvvar"))
			return VariableOperationCode.RenameVariable;
		
		else if ( str.equalsIgnoreCase("rmfilevar") ) 
			return VariableOperationCode.RemoveVariableAndFile;
		
		else if ( str.equalsIgnoreCase("assignvarwithfile") ) 
			return VariableOperationCode.AssignVariableWithFirstValue;
		
		else if ( str.equalsIgnoreCase("attachfiletovar") ) 
			return VariableOperationCode.AttachFileToVariable;
		
		else if ( str.equalsIgnoreCase("valuepickCP") ) 
			return VariableOperationCode.ValuePickCP;
		
		else if ( str.equalsIgnoreCase("iqsize") ) 
			return VariableOperationCode.IQSize;
		
		else if ( str.equalsIgnoreCase("spearmanhelper") ) 
			return VariableOperationCode.SpearmanHelper;
		
		else if ( str.equalsIgnoreCase("writeScalar") ) 
			return VariableOperationCode.writeScalar;
		
		else if ( str.equalsIgnoreCase("readScalar") ) 
			return VariableOperationCode.readScalar;
		
		else
			throw new DMLUnsupportedOperationException("Invalid function: " + str);
	}
	
	private static String toString ( VariableOperationCode code ) {
		switch(code) {
		case AssignVariable:
			return "assignvar";
			
		case RemoveVariable: 
			return "rmvar";
			
		case RenameVariable:
			return "mvvar";
			
		case RemoveVariableAndFile:
			return "rmfilevar";
			
		case AssignVariableWithFirstValue:
			return "assignvarwithfile";
			
		case AttachFileToVariable:
			return "attachfiletovar";
		
		case ValuePickCP:
			return "valuepickCP";
		
		case IQSize:
			return "iqsize";
			
		case SpearmanHelper:
			return "spearmanhelper";
		
		case readScalar:
			return "readScalar";
		
		case writeScalar:
			return "writeScalar";
		
		default:
			return null;
		}
	}
	
	public VariableCPInstruction (VariableOperationCode op, CPOperand in1, CPOperand in2, int _arity, String istr )
	{
		super(null);
		cptype = CPINSTRUCTION_TYPE.Variable;
		opcode = op;
		input1 = in1;
		input2 = in2;
		arity = _arity;
		instString = istr;
	}

	public VariableCPInstruction (VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand out, int _arity, String istr )
	{
		super(null);
		cptype = CPINSTRUCTION_TYPE.Variable;
		opcode = op;
		input1 = in1;
		input2 = in2;
		output = out;
		arity = _arity;
		instString = istr;
	}

	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		String opcode = InstructionUtils.getOpCode(str);
		
		int _arity = 2;
		if ( opcode.equalsIgnoreCase("rmvar") )
			_arity = 1;
		else if ( opcode.equalsIgnoreCase("valuepickCP") || opcode.equalsIgnoreCase("iqsize"))
			_arity = 3;
		
		InstructionUtils.checkNumFields ( str, _arity ); // no output
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		
		CPOperand in1=null, in2=null, out=null;
		VariableOperationCode voc = getVariableOperationCode(opcode);
		
		switch (voc) {
		
		case AssignVariable:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			if ( in1.get_valueType() != in2.get_valueType() ) 
				throw new DMLRuntimeException("Value type mismatch while assigning variables.");
			break;
			
		case RemoveVariable:
			in1 = new CPOperand(parts[1], ValueType.UNKNOWN);
			in2 = null;
			break;
			
		case RenameVariable:
			// Value types are not given here
			in1 = new CPOperand(parts[1], ValueType.UNKNOWN);
			in2 = new CPOperand(parts[2], ValueType.UNKNOWN);
			break;
			
		case RemoveVariableAndFile:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			// second argument must be a boolean
			if ( in2.get_valueType() != ValueType.BOOLEAN)
				throw new DMLRuntimeException("Unexpected value type for second argument in: " + str);
			break;
			
		case AssignVariableWithFirstValue:
			in1 = new CPOperand(parts[1]); // first operand is a filename => string value type 
			in2 = new CPOperand(parts[2]); // second operand is a variable and is assumed to be double 
			break;
			
		case AttachFileToVariable:
			in1 = new CPOperand(parts[1], ValueType.STRING); // first operand is a filename => string value type 
			in2 = new CPOperand(parts[2], ValueType.UNKNOWN); 
			break;
		
		case ValuePickCP:
		case IQSize:
			in1 = new CPOperand(parts[1]); // first operand is a filename => string value type 
			in2 = new CPOperand(parts[2]); // second operand is a variable and is assumed to be double
			out = new CPOperand(parts[3]); // output variable name
			break;
		
		case SpearmanHelper:
			in1 = new CPOperand(parts[1]); // first operand is a filename => string value type 
			out = new CPOperand(parts[2]); // output variable name
			break;
			
		case writeScalar:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			out = null;
			break;
			
		case readScalar:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			out = null;
			break;
		}
		
		return new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, out, _arity, str);
		
	}

	@Override
	public ScalarObject processInstruction(ProgramBlock pb) throws DMLRuntimeException {
		
		switch ( opcode ) {
		case AssignVariable:
			// assign value of variable to the other
			pb.setVariable(input2.get_name(), pb.getVariable(input1.get_name(), input1.get_valueType()));			
			break;
			
		case RemoveVariable:
			// remove variable from the program block
			pb.removeVariable(input1.get_name());
			
			// if the variable corresponds to a file on HDFS, then delete it from program block's metadata structure
			if ( pb.getMetaData(input1.get_name()) != null ) {
				pb.removeMetaData(input1.get_name());
			}
			break;
			
		case RenameVariable:
			// assign value of input1 to input2 
			// we expect input1 to exist in the program block. 
			// We pass value type as UNKNOWN so that getVariable() fails if input1 is not in the program block.
			pb.setVariable(input2.get_name(), pb.getVariable(input1.get_name(), ValueType.UNKNOWN));
			
			// If input1 is a file, copy the corresponding metadata
			MetaData md = null;
			if ( (md = pb.getMetaData(input1.get_name())) != null ) {
				pb.setMetaData(input2.get_name(), md);
			}
			break;
			
		case RemoveVariableAndFile:
			 // Remove the variable from HashMap _variables, and possibly delete the data on disk. 
			boolean del = ( (BooleanObject) pb.getVariable(input2.get_name(), input2.get_valueType()) ).getBooleanValue();
			
			if ( del == true ) {
				// delete the file on disk
				try {
					FileObject fileObj = (FileObject) pb.getVariable( input1.get_name(), input1.get_valueType() );
					MapReduceTool.deleteFileIfExistOnHDFS( fileObj.getFilePath() );
					
					// delete metadata associated with the file
					pb.removeMetaData(input1.get_name());
					MapReduceTool.deleteFileIfExistOnHDFS( fileObj.getFilePath() + ".mtd" ); // delete the metadata file on hdfs
				}
				catch ( IOException e ) {
					throw new DMLRuntimeException(e);
				}
			}
			// remove the variable from the HashMap (_variables) in ProgramBlock.
			pb.removeVariable( input1.get_name() );
			break;
			
		case AssignVariableWithFirstValue:
			// Read the value from HDFS file, and assign it to a variable (used in CAST_TO_SCALAR)
			try {
				double value = MapReduceTool.readFirstNumberFromHDFSMatrix(input1.get_name());
				pb.setVariable(input2.get_name(), new DoubleObject(value));
			} catch (IOException e) {
				e.printStackTrace();
				pb.setVariable(input2.get_name(), new DoubleObject(0));
			}
			break;
			
		case AttachFileToVariable:
			pb.setVariable( input2.get_name(), new FileObject( input1.get_name()));
			break;
			
		case ValuePickCP:
			// example = valuepickCP:::temp3:DOUBLE:::0.5:DOUBLE:::Var0:DOUBLE
			// pick middle value from "temp3" and assign to Var0
			
			String fname = input1.get_name();
			MetaData mdata = pb.getMetaData(fname);
			ScalarObject pickindex = pb.getScalarVariable(input2.get_name(), input2.get_valueType());
			
			if ( mdata != null ) {
				try {
					double picked = MapReduceTool.pickValue(fname, (NumItemsByEachReducerMetaData) mdata, pickindex.getDoubleValue());
					ScalarObject result = (ScalarObject) new DoubleObject(picked);
					pb.setVariable(output.get_name(), result);
				} catch (Exception e ) {
					throw new DMLRuntimeException(e);
				}
			}
			else {
				throw new DMLRuntimeException("Unexpected error while executing ValuePickCP: otherMetaData for file (" + fname + ") not found." );
			}
			break;
			
			
		case IQSize:
			// example = iqsize:::temp3:DOUBLE:::0.25:DOUBLE:::Var0:DOUBLE
			// compute the number of records in interquartile
			
			// get otherMetadata associated with the matrix
			String iqs_fname = input1.get_name();
			MetaData iqs_md = pb.getMetaData(iqs_fname);
			
			// ge the range
			ScalarObject pickrange = pb.getScalarVariable(input2.get_name(), input2.get_valueType());
			
			if ( iqs_md != null ) {
				try {
					double rangesize = UtilFunctions.getLengthForInterQuantile((NumItemsByEachReducerMetaData) iqs_md, pickrange.getDoubleValue());
					ScalarObject result = (ScalarObject) new DoubleObject(rangesize);
					pb.setVariable(output.get_name(), result);
				} catch (Exception e ) {
					throw new DMLRuntimeException(e);
				}
			}
			else {
				throw new DMLRuntimeException("Unexpected error while executing IQSize: otherMetaData for file (" + iqs_fname + ") not found." );
			}
			break;
			
		case SpearmanHelper:
			// get otherMetadata associated with the matrix
			String sh_fname = input1.get_name();
			MetaData sh_md = pb.getMetaData(sh_fname);
			
			InputInfo ii = ((InputInfoMetaData)sh_md).getInputInfo();
			MatrixCharacteristics mc = ((InputInfoMetaData)sh_md).getMatrixCharacteristics();
			
			try {
				double[][] ctable = MapReduceTool.readMatrixFromHDFS(sh_fname, ii, mc.numRows, mc.numColumns, mc.numRowsPerBlock, mc.numColumnsPerBlock);
				
				double sp = spearmanHelperFunction(ctable, (int)mc.numRows, (int)mc.numColumns);
				
				ScalarObject result = (ScalarObject) new DoubleObject(sp);
				pb.setVariable(output.get_name(), result);
				
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			break;
		
		case readScalar:
			ScalarObject res = null;
			try {
				switch(input1.get_valueType()) {
				case DOUBLE:
					double d = MapReduceTool.readDoubleFromHDFSFile(input2.get_name());
					res = (ScalarObject) new DoubleObject(d);
					break;
				case INT:
					int i = MapReduceTool.readIntegerFromHDFSFile(input2.get_name());
					res = (ScalarObject) new IntObject(i);
					break;
				case BOOLEAN:
					boolean b = MapReduceTool.readBooleanFromHDFSFile(input2.get_name());
					res = (ScalarObject) new BooleanObject(b);
					break;
				case STRING:
					String s = MapReduceTool.readStringFromHDFSFile(input2.get_name());
					res = (ScalarObject) new StringObject(s);
					break;
					default:
						throw new DMLRuntimeException("Invalid value type (" + input1.get_valueType() + ") while processing readScalar instruction.");
				}
			} catch ( IOException e ) {
				throw new DMLRuntimeException(e);
			}
			pb.setVariable(input1.get_name(), res);
			
			break;
			
		case writeScalar:
			ScalarObject scalar = pb.getScalarVariable(input1.get_name(), input1.get_valueType());
			try {
				switch ( input1.get_valueType() ) {
				case DOUBLE:
					MapReduceTool.writeDoubleToHDFS(scalar.getDoubleValue(), input2.get_name());
					break;
				case INT:
					MapReduceTool.writeIntToHDFS(scalar.getIntValue(), input2.get_name());
					break;
				case BOOLEAN:
					MapReduceTool.writeBooleanToHDFS(scalar.getBooleanValue(), input2.get_name());
					break;
				case STRING:
					MapReduceTool.writeStringToHDFS(scalar.getStringValue(), input2.get_name());
					break;
				default:
					throw new DMLRuntimeException("Invalid value type (" + input1.get_valueType() + ") while processing writeScalar instruction.");
				}
			} catch ( IOException e ) {
				throw new DMLRuntimeException(e);
			}
			break;
			
		default:
			throw new DMLRuntimeException("Unknown opcode: " + opcode );
		}
		return null;
	}

	/*
	 * This is a helper function to compute Spearman correlation between two ordinal variables.
	 * It takes the contingency table constructed between ordinal variables as the input.
	 */
	double spearmanHelperFunction(double [][]ctable, int rows, int cols) {
		
		double [] rowSums = new double[rows];
		double [] colSums = new double[cols];
		double [] rowScores = new double[rows];
		double [] colScores = new double[cols];
		double totalWeight = 0.0;
		
		for ( int i=0; i < rows; i++ ) {
			rowSums[i] = rowScores[i] = 0.0;
		}
		for ( int j=0; j < cols; j++ ) {
			colSums[j] = colScores[j] = 0;
		}
		
		for ( int i=0; i < rows; i++ ) {
			for ( int j=0; j < cols; j++ ) {
				rowSums[i] += ctable[i][j];
				colSums[j] += ctable[i][j];
				totalWeight += ctable[i][j]; 
			}
		}
		
		double prefix_sum=0.0;
		for ( int i=0; i < rows; i++ ) {
			rowScores[i] = prefix_sum + (rowSums[i]+1)/2;
			prefix_sum += rowSums[i];
		}
		
		prefix_sum=0.0;
		for ( int j=0; j < cols; j++ ) {
			colScores[j] = prefix_sum + (colSums[j]+1)/2;
			prefix_sum += colSums[j];
		}
		
		double Rx = 0.0, Ry = 0.0;
		for ( int i=0; i < rows; i++ ) {
			Rx += rowSums[i]*rowScores[i];
		}
		for ( int j=0; j < cols; j++ ) {
			Ry += colSums[j]*colScores[j];
		}
		Rx = Rx/(double)totalWeight;
		Ry = Ry/(double)totalWeight;
		
		double VRx=0.0, VRy=0.0;
		for ( int i=0; i < rows; i++ ) {
			VRx += rowSums[i] * ((rowScores[i]-Rx)*(rowScores[i]-Rx));
		}
		VRx = VRx/(double)(totalWeight-1);
		
		for ( int j=0; j < cols; j++ ) {
			VRy += colSums[j] * ((colScores[j]-Ry)*(colScores[j]-Ry));
		}
		VRy = VRy/(double)(totalWeight-1);
		
		double CRxRy = 0.0;
		for ( int i=0; i < rows; i++ ) {
			for ( int j=0; j < cols; j++ ) {
				CRxRy = ctable[i][j] * (rowScores[i]-Rx) * (colScores[j]-Ry);
			}
		}
		CRxRy = CRxRy / (double)(totalWeight-1);
		
		double spearman = CRxRy/(Math.sqrt(VRx) * Math.sqrt(VRy));
		
		return spearman;
	}
	
}

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import java.io.IOException;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.MetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


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
	 *	5) assignvarwithfile FN x
	 *	    assign x with the first value from the file whose name=FN
	 *	6) attachfiletovar FP x
	 *	    allocate a new file object with name FP, and associate it with variable x
	 *     createvar x FP [dimensions] [formatinfo]
	 */
	
	private enum VariableOperationCode {
		CreateVariable, AssignVariable, RemoveVariable, RenameVariable, RemoveVariableAndFile, AssignVariableWithFirstValue, ValuePick, InMemValuePick, InMemIQM, IQSize, SpearmanHelper, Write, Read, SetFileName
	}
	
	VariableOperationCode opcode;
	private CPOperand input1;
	private CPOperand input2;
	private CPOperand input3;
	private CPOperand output;
	private MetaData metadata;
	int arity;
	
	private static VariableOperationCode getVariableOperationCode ( String str ) throws DMLUnsupportedOperationException {
		
		if ( str.equalsIgnoreCase("createvar"))
			return VariableOperationCode.CreateVariable;
		
		else if ( str.equalsIgnoreCase("assignvar"))
			return VariableOperationCode.AssignVariable;
		
		else if ( str.equalsIgnoreCase("rmvar") ) 
			return VariableOperationCode.RemoveVariable;
		
		else if ( str.equalsIgnoreCase("mvvar"))
			return VariableOperationCode.RenameVariable;
		
		else if ( str.equalsIgnoreCase("rmfilevar") ) 
			return VariableOperationCode.RemoveVariableAndFile;
		
		else if ( str.equalsIgnoreCase("assignvarwithfile") ) 
			return VariableOperationCode.AssignVariableWithFirstValue;
		
		else if ( str.equalsIgnoreCase("valuepick") ) 
			return VariableOperationCode.ValuePick;
		
		else if ( str.equalsIgnoreCase("inmem-valuepick") ) 
			return VariableOperationCode.InMemValuePick;
		
		else if ( str.equalsIgnoreCase("inmem-iqm") ) 
			return VariableOperationCode.InMemIQM;
		
		else if ( str.equalsIgnoreCase("iqsize") ) 
			return VariableOperationCode.IQSize;
		
		else if ( str.equalsIgnoreCase("spearmanhelper") ) 
			return VariableOperationCode.SpearmanHelper;
		
		else if ( str.equalsIgnoreCase("write") ) 
			return VariableOperationCode.Write;
		
		else if ( str.equalsIgnoreCase("read") ) 
			return VariableOperationCode.Read;
		
		else if ( str.equalsIgnoreCase("setfilename") ) 
			return VariableOperationCode.SetFileName;
		
		else
			throw new DMLUnsupportedOperationException("Invalid function: " + str);
	}
	
/*	private static String toString ( VariableOperationCode code ) {
		switch(code) {
		case CreateVariable:
			return "createvar";
		
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
			
		case ValuePick:
			return "valuepick";
		
		case InMemValuePick:
			return "inmem-valuepick";
		
		case InMemIQM:
			return "inmem-iqm";
		
		case IQSize:
			return "iqsize";
			
		case SpearmanHelper:
			return "spearmanhelper";
		
		case Read:
			return "read";
		
		case Write:
			return "write";
			
		case SetFileName:
			return "setfilename";
		
		default:
			return null;
		}
	}
*/	
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

	public VariableCPInstruction (VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, String istr )
	{
		super(null);
		cptype = CPINSTRUCTION_TYPE.Variable;
		opcode = op;
		input1 = in1;
		input2 = in2;
		input3 = in3;
		output = null;
		arity = 3;
		instString = istr;
	}

	public VariableCPInstruction (VariableOperationCode op, CPOperand in1, CPOperand in2, MetaData md, int _arity, String istr )
	{
		super(null);
		cptype = CPINSTRUCTION_TYPE.Variable;
		opcode = op;
		input1 = in1;
		input2 = in2;
		output = null;
		metadata = md;
		arity = _arity;
		instString = istr;
	}

	private static int getArity(VariableOperationCode op) {
		switch(op) {
		case RemoveVariable:
			return 1;
		case IQSize:
		case ValuePick:
		case InMemValuePick:
		case Write:
		case SetFileName:
			return 3;
		default:
			return 2;
		}
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		String opcode = InstructionUtils.getOpCode(str);
		VariableOperationCode voc = getVariableOperationCode(opcode);
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		int _arity = -1;
		
		if ( voc == VariableOperationCode.CreateVariable ){
			if ( parts.length != 3 && parts.length != 4 && parts.length != 8 && parts.length != 9 )
				throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
		}
		else {
			_arity = getArity(voc);
			InstructionUtils.checkNumFields ( str, _arity ); // no output
		}
		
		CPOperand in1=null, in2=null, in3=null, out=null;
		
		switch (voc) {
		
		case CreateVariable:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			if ( parts.length > 3 ) {
				MatrixCharacteristics mc = new MatrixCharacteristics();
				if ( parts.length == 4 ) {
					// the last operand is the OutputInfo
					OutputInfo oi = OutputInfo.stringToOutputInfo(parts[3]);
					InputInfo ii = OutputInfo.getMatchingInputInfo(oi);
					MatrixFormatMetaData iimd = new MatrixFormatMetaData(mc, oi, ii);
					return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, iimd, parts.length, str);
				}
				mc.setDimension(Long.parseLong(parts[3]), Long.parseLong(parts[4]));
				mc.setBlockSize(Integer.parseInt(parts[5]), Integer.parseInt(parts[6]));
				mc.setNonZeros(Long.parseLong(parts[7]));
				
				if (parts.length > 8) {
					OutputInfo oi = OutputInfo.stringToOutputInfo(parts[8]);
					InputInfo ii = OutputInfo.getMatchingInputInfo(oi);
					MatrixFormatMetaData iimd = new MatrixFormatMetaData(mc, oi, ii);
					return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, iimd, parts.length, str);
				}
				else {
					MatrixDimensionsMetaData mdmd = new MatrixDimensionsMetaData(mc);
					return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, mdmd, parts.length, str);
				}
			}
			break;
			
		case AssignVariable:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			if ( in1.get_valueType() != in2.get_valueType() ) 
				throw new DMLRuntimeException("Value type mismatch while assigning variables.");
			break;
			
		case RemoveVariable:
			in1 = new CPOperand(parts[1], ValueType.UNKNOWN, DataType.SCALAR);
			in2 = null;
			break;
			
		case RenameVariable:
			// Value types are not given here
			in1 = new CPOperand(parts[1], ValueType.UNKNOWN, DataType.UNKNOWN);
			in2 = new CPOperand(parts[2], ValueType.UNKNOWN, DataType.UNKNOWN);
			break;
			
		case RemoveVariableAndFile:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			// second argument must be a boolean
			if ( in2.get_valueType() != ValueType.BOOLEAN)
				throw new DMLRuntimeException("Unexpected value type for second argument in: " + str);
			break;
			
		case AssignVariableWithFirstValue:
			in1 = new CPOperand(parts[1]); // first operand is a variable name => string value type 
			in2 = new CPOperand(parts[2]); // second operand is a variable and is assumed to be double 
			break;
			
		case ValuePick:
		case InMemValuePick: 
		case IQSize:
			in1 = new CPOperand(parts[1]); // sorted data, which is input to Valuepick 
			in2 = new CPOperand(parts[2]); // second operand is a variable and is assumed to be double
			out = new CPOperand(parts[3]); // output variable name
			break;
		
		case InMemIQM:
			in1 = new CPOperand(parts[1]); // sorted data, which is input to IQM
			out = new CPOperand(parts[2]);
			break;
			
		case SpearmanHelper:
			in1 = new CPOperand(parts[1]); // first operand is a filename => string value type 
			out = new CPOperand(parts[2]); // output variable name
			break;
			
		case Write:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			in3 = new CPOperand(parts[3]);
			return new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, in3, str);
			
		case Read:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			out = null;
			break;
			
		case SetFileName:
			in1 = new CPOperand(parts[1]); // variable name
			in2 = new CPOperand(parts[2], ValueType.UNKNOWN, DataType.UNKNOWN); // file name
			in3 = new CPOperand(parts[3], ValueType.UNKNOWN, DataType.UNKNOWN); // option: remote or local
			return new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, in3, str);
		}
		return new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, out, _arity, str);
	}

	@Override
	public void processInstruction(ProgramBlock pb) throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		switch ( opcode ) { 
		case CreateVariable:
			
			if ( input1.get_dataType() == DataType.MATRIX ) {
				/*
				 * Following if condition is used for debugging, and to easily
				 * switch between OLD MatrixObject and NEW MatrixObject classes.
				 * Once MatrixObjectNew is completely tested, it can be removed. 
				 */
				
				//create new variable for symbol table and cache
				//(existing objects already cleared on 'rm vars')
				MatrixObjectNew mobj = new MatrixObjectNew(input1.get_valueType(), input2.get_name());
				mobj.setVarName(input1.get_name());
				mobj.setDataType(DataType.MATRIX);
				mobj.setMetaData(metadata);
				
				pb.setVariable(input1.get_name(), mobj);
			}
			else if ( input1.get_dataType() == DataType.SCALAR ){
				ScalarObject sobj = null;
				pb.setScalarOutput(input1.get_name(), sobj);
			}
			else {
				throw new DMLRuntimeException("Unexpected data type: " + input1.get_dataType());
			}
			break;
		
		case AssignVariable:
			// assign value of variable to the other
			pb.setScalarOutput(input2.get_name(), pb.getScalarInput(input1.get_name(), input1.get_valueType()));			
			break;
			
		case RemoveVariable:
			//remove matrix object from cache
			if ( input1.get_dataType() == DataType.MATRIX )
				((MatrixObjectNew)pb.getVariable(input1.get_name())).clearData();
			
			// remove variable from the program block
			pb.removeVariable(input1.get_name());
			break;
			
		case RenameVariable:
			Data dd = pb.getVariable(input1.get_name());
			pb.setVariable(input2.get_name(), dd);
			//pb.removeVariable(input1.get_name());
			break;
			
		case RemoveVariableAndFile:
			//remove matrix object from cache
			if ( input1.get_dataType() == DataType.MATRIX )
				((MatrixObjectNew)pb.getVariable(input1.get_name())).clearData();
			else if ( input1.get_dataType() == DataType.UNKNOWN || input1.get_dataType() == null )
			{
				Data dat = pb.getVariable(input1.get_name());
				if( dat instanceof MatrixObjectNew )
					((MatrixObjectNew)dat).clearData();
			}
			
			 // Remove the variable from HashMap _variables, and possibly delete the data on disk. 
			boolean del = ( (BooleanObject) pb.getScalarInput(input2.get_name(), input2.get_valueType()) ).getBooleanValue();
			
			if ( del == true ) {
				// delete the file on disk
				try {
					String fpath = ((MatrixObjectNew)pb.getVariable( input1.get_name())).getFileName();
					if ( fpath != null ) {
						MapReduceTool.deleteFileIfExistOnHDFS( fpath );
						// delete metadata associated with the file
						pb.removeMetaData(input1.get_name());
						MapReduceTool.deleteFileIfExistOnHDFS( fpath + ".mtd" ); // delete the metadata file on hdfs
					}
				}
				catch ( IOException e ) {
					throw new DMLRuntimeException(e);
				}
			}
			// remove the variable from the HashMap (_variables) in ProgramBlock.
			pb.removeVariable( input1.get_name() );
			break;
			
		case AssignVariableWithFirstValue:
			MatrixBlock mBlock = (MatrixBlock) pb.getMatrixInput(input1.get_name());
			double value = mBlock.getValue(0,0);
			//double value = MapReduceTool.readFirstNumberFromHDFSMatrix(mobj.getFileName());
			pb.releaseMatrixInput(input1.get_name());
			pb.setScalarOutput(input2.get_name(), new DoubleObject(value));
			break;
			
		case ValuePick:
			// example = valuepickCP:::temp3:DOUBLE:::0.5:DOUBLE:::Var0:DOUBLE
			// pick a value from "temp3" and assign to Var0
			
			MatrixObjectNew mat = (MatrixObjectNew)pb.getVariable(input1.get_name());
			String fname = mat.getFileName();
			MetaData mdata = mat.getMetaData();
			ScalarObject pickindex = pb.getScalarInput(input2.get_name(), input2.get_valueType());
			
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
			
		case InMemValuePick:
			MatrixBlock matBlock = (MatrixBlock) pb.getMatrixInput(input1.get_name());

			if ( input2.get_dataType() == DataType.SCALAR ) {
				ScalarObject quantile = pb.getScalarInput(input2.get_name(), input2.get_valueType());
				double picked = matBlock.pickValue(quantile.getDoubleValue());
				pb.setScalarOutput(output.get_name(), (ScalarObject) new DoubleObject(picked));
			} 
			else {
				MatrixBlock quantiles = (MatrixBlock) pb.getMatrixInput(input2.get_name());
				MatrixBlock resultBlock = (MatrixBlock) matBlock.pickValues(quantiles, new MatrixBlock());
				quantiles = null;
				pb.releaseMatrixInput(input2.get_name());
				pb.setMatrixOutput(output.get_name(), resultBlock);
			}
			matBlock = null;
			pb.releaseMatrixInput(input1.get_name());
			break;
			
		case InMemIQM:
			MatrixBlock matBlock1 = pb.getMatrixInput(input1.get_name());
			double iqm = matBlock1.interQuartileMean();
			
			matBlock1 = null;
			pb.releaseMatrixInput(input1.get_name());
			pb.setScalarOutput(output.get_name(), (ScalarObject) new DoubleObject(iqm));
			break;
			
		case IQSize:
			// example = iqsize:::temp3:DOUBLE:::0.25:DOUBLE:::Var0:DOUBLE
			// compute the number of records in interquartile
			
			// get otherMetadata associated with the matrix
			String iqs_fname = input1.get_name();
			MetaData iqs_md = pb.getMetaData(iqs_fname);
			
			// ge the range
			ScalarObject pickrange = pb.getScalarInput(input2.get_name(), input2.get_valueType());
			
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
			//MatrixObject mobj = pb.getMatrixVariable(input1.get_name());
			MatrixObjectNew mobj = (MatrixObjectNew)pb.getVariable(input1.get_name());
			String sh_fname = mobj.getFileName();
			MetaData sh_md = mobj.getMetaData();
			
			InputInfo ii = ((MatrixFormatMetaData)sh_md).getInputInfo();
			MatrixCharacteristics mc = ((MatrixFormatMetaData)sh_md).getMatrixCharacteristics();
			
			try {
				double[][] ctable = MapReduceTool.readMatrixFromHDFS(sh_fname, ii, mc.numRows, mc.numColumns, mc.numRowsPerBlock, mc.numColumnsPerBlock);
				
				double sp = spearmanHelperFunction(ctable, (int)mc.numRows, (int)mc.numColumns);
				
				ScalarObject result = (ScalarObject) new DoubleObject(sp);
				pb.setVariable(output.get_name(), result);
				
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			break;
		
		case Read:
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
			pb.setScalarOutput(input1.get_name(), res);
			
			break;
			
		case Write:
			if ( input1.get_dataType() == DataType.SCALAR ) {
				ScalarObject scalar = pb.getScalarInput(input1.get_name(), input1.get_valueType());
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
						throw new DMLRuntimeException("Invalid value type (" + input1.get_valueType() + ") in writeScalar instruction: " + instString);
					}
				  // write out .mtd file
				  MapReduceTool.writeScalarMetaDataFile(input2.get_name() +".mtd", input1.get_valueType());
				} catch ( IOException e ) {
					throw new DMLRuntimeException(e);
				}
			}
			else {
				MatrixObjectNew mo = (MatrixObjectNew)pb.getVariable(input1.get_name());
				mo.exportData(input2.get_name(), input3.get_name());
			}
			break;
			
		case SetFileName:
			Data data = pb.getVariable(input1.get_name());
			if ( data.getDataType() == DataType.MATRIX ) {
				if ( input3.get_name().equalsIgnoreCase("remote") ) {
					((MatrixObjectNew)data).setFileName(input2.get_name());
				}
				else {
					throw new DMLRuntimeException("Invalid location (" + input3.get_name() + ") in SetFileName instruction: " + instString);
				}
			} else{
				throw new DMLRuntimeException("Invalid data type (" + input1.get_dataType() + ") in SetFileName instruction: " + instString);
			}
			break;
			
		default:
			throw new DMLRuntimeException("Unknown opcode: " + opcode );
		}
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

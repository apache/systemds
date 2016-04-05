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

package org.apache.sysml.runtime.instructions.cp;

import java.io.IOException;

import org.apache.commons.lang.StringUtils;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.UnaryCP;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.io.WriterMatrixMarket;
import org.apache.sysml.runtime.io.WriterTextCSV;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.MetaData;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.utils.Statistics;


public class VariableCPInstruction extends CPInstruction 
{
	
	/*
	 * Supported Operations
	 * --------------------
	 *	1) assignvar x:type y:type
	 *	    assign value of y to x (both types should match)
	 *	2) rmvar x
	 *	    remove variable x
	 *	3) cpvar x y
	 *	    copy x to y (same as assignvar followed by rmvar, types are not required)
	 *	4) rmfilevar x:type b:type
	 *	    remove variable x, and if b=true then the file object associated with x (b's type should be boolean)
	 *	5) assignvarwithfile FN x
	 *	    assign x with the first value from the file whose name=FN
	 *	6) attachfiletovar FP x
	 *	    allocate a new file object with name FP, and associate it with variable x
	 *     createvar x FP [dimensions] [formatinfo]
	 */
	
	private enum VariableOperationCode 
	{
		CreateVariable, 
		AssignVariable, 
		CopyVariable,
		MoveVariable,
		RemoveVariable, 
		RemoveVariableAndFile,		 
		CastAsScalarVariable, 
		CastAsMatrixVariable, 
		CastAsFrameVariable,
		CastAsDoubleVariable,
		CastAsIntegerVariable,
		CastAsBooleanVariable,
		Write, 
		Read, 
		SetFileName, 
	}
	
	private static IDSequence _uniqueVarID;	
	private static final int CREATEVAR_FILE_NAME_VAR_POS=3;
	
	private VariableOperationCode opcode;
	private CPOperand input1;
	private CPOperand input2;
	private CPOperand input3;
	private CPOperand output;
	private MetaData metadata;
	private boolean updateInPlace;
	
	// CSV related members (used only in createvar instructions)
	private FileFormatProperties formatProperties;
	
	static {
		_uniqueVarID  = new IDSequence(true); 
	}
	
	private static VariableOperationCode getVariableOperationCode ( String str ) throws DMLRuntimeException {
		
		if ( str.equalsIgnoreCase("createvar"))
			return VariableOperationCode.CreateVariable;
		
		else if ( str.equalsIgnoreCase("assignvar"))
			return VariableOperationCode.AssignVariable;
		
		else if ( str.equalsIgnoreCase("cpvar"))
			return VariableOperationCode.CopyVariable;
		
		else if ( str.equalsIgnoreCase("mvvar"))
			return VariableOperationCode.MoveVariable;
		
		else if ( str.equalsIgnoreCase("rmvar") ) 
			return VariableOperationCode.RemoveVariable;
		
		else if ( str.equalsIgnoreCase("rmfilevar") ) 
			return VariableOperationCode.RemoveVariableAndFile;
		
		else if ( str.equalsIgnoreCase(UnaryCP.CAST_AS_SCALAR_OPCODE) ) 
			return VariableOperationCode.CastAsScalarVariable;
		
		else if ( str.equalsIgnoreCase(UnaryCP.CAST_AS_MATRIX_OPCODE) ) 
			return VariableOperationCode.CastAsMatrixVariable;
		
		else if ( str.equalsIgnoreCase(UnaryCP.CAST_AS_FRAME_OPCODE) ) 
			return VariableOperationCode.CastAsFrameVariable;
		
		else if ( str.equalsIgnoreCase(UnaryCP.CAST_AS_DOUBLE_OPCODE) ) 
			return VariableOperationCode.CastAsDoubleVariable;
		
		else if ( str.equalsIgnoreCase(UnaryCP.CAST_AS_INT_OPCODE) ) 
			return VariableOperationCode.CastAsIntegerVariable;
		
		else if ( str.equalsIgnoreCase(UnaryCP.CAST_AS_BOOLEAN_OPCODE) ) 
			return VariableOperationCode.CastAsBooleanVariable;
		
		else if ( str.equalsIgnoreCase("write") ) 
			return VariableOperationCode.Write;
		
		else if ( str.equalsIgnoreCase("read") ) 
			return VariableOperationCode.Read;
		
		else if ( str.equalsIgnoreCase("setfilename") ) 
			return VariableOperationCode.SetFileName;
		
		else
			throw new DMLRuntimeException("Invalid function: " + str);
	}
	
	// Checks if this instructon is a remove instruction for varName
	public boolean isRemoveVariable(String varName) {
		if ( opcode == VariableOperationCode.RemoveVariable || opcode == VariableOperationCode.RemoveVariableAndFile) {
			if ( input1.getName().equalsIgnoreCase(varName))
				return true;
		}
		return false;
	}
	
	public boolean isRemoveVariable() {
		if ( opcode == VariableOperationCode.RemoveVariable || opcode == VariableOperationCode.RemoveVariableAndFile) {
			return true;
		}
		return false;
	}
	
	public VariableCPInstruction (VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, int _arity, String sopcode, String istr )
	{
		super(sopcode, istr);
		_cptype = CPINSTRUCTION_TYPE.Variable;
		opcode = op;
		input1 = in1;
		input2 = in2;
		input3 = in3;
		output = out;
		
		formatProperties = null;
	}

	// This version of the constructor is used only in case of CreateVariable
	public VariableCPInstruction (VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, MetaData md, boolean updateInPlace, int _arity, String sopcode, String istr)
	{
		this(op, in1, in2, in3, (CPOperand)null, _arity, sopcode, istr);
		metadata = md;
		this.updateInPlace = updateInPlace;		
	}
	
	// This version of the constructor is used only in case of CreateVariable
	public VariableCPInstruction (VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, MetaData md, boolean updateInPlace, int _arity, FileFormatProperties formatProperties, String sopcode, String istr)
	{
		this(op, in1, in2, in3, (CPOperand)null, _arity, sopcode, istr);
		metadata = md;
		this.updateInPlace = updateInPlace;  
		this.formatProperties = formatProperties;
	}
	
	public FileFormatProperties getFormatProperties() {
		return formatProperties;
	}
	
	public void setFormatProperties(FileFormatProperties prop) {
		formatProperties = prop;
	}
	
	public CPOperand getInput1() {
		return input1;
	}
	
	public CPOperand getInput2() {
		return input2;
	}
	
	public CPOperand getInput3() {
		return input3;
	}
	
	public String getOutputVariableName(){
		String ret = null;
		if( output != null )
			ret = output.getName();
		return ret;
	}

	private static int getArity(VariableOperationCode op) {
		switch(op) {
		case RemoveVariable:
			return 1;
		case Write:
		case SetFileName:
			return 3;
		default:
			return 2;
		}
	}
	
	public static VariableCPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{	
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		String opcode = parts[0];
		VariableOperationCode voc = getVariableOperationCode(opcode);
		int _arity = -1;
		
		if ( voc == VariableOperationCode.CreateVariable ){
			if ( parts.length < 5 )  //&& parts.length != 10 )
				throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
		}
		else if ( voc == VariableOperationCode.MoveVariable) {
			// mvvar tempA A; or mvvar mvar5 "data/out.mtx" "binary"
			if ( parts.length !=3 && parts.length != 4)
				throw new DMLRuntimeException("Invalid number of operands in mvvar instruction: " + str);
		}
		else if ( voc == VariableOperationCode.Write ) {
			// All write instructions have 3 parameters, except in case of delimited/csv file.
			// Write instructions for csv files also include three additional parameters (hasHeader, delimiter, sparse)
			if ( parts.length != 4 && parts.length != 7 )
				throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
		}
		else {
			_arity = getArity(voc);
			InstructionUtils.checkNumFields ( parts, _arity ); // no output
		}
		
		CPOperand in1=null, in2=null, in3=null, out=null;
		
		switch (voc) {
		
		case CreateVariable:
			// variable name (only supports Matrices, and only w/ double value type)
			in1 = new CPOperand(parts[1], ValueType.DOUBLE, DataType.MATRIX);
			// file name
			in2 = new CPOperand(parts[2], ValueType.STRING, DataType.SCALAR);
			// file name override flag
			in3 = new CPOperand(parts[3], ValueType.BOOLEAN, DataType.SCALAR);
			
			// format 
			String fmt = parts[4];
			if ( fmt.equalsIgnoreCase("csv") ) {
				/*
				 * Cretevar instructions for CSV format either has 13 or 14 inputs.
				 * 13 inputs: createvar corresponding to WRITE -- includes properties hasHeader, delim, and sparse
				 * 14 inputs: createvar corresponding to READ -- includes properties hasHeader, delim, fill, and fillValue
				 */
				if ( parts.length < 14 || parts.length > 16 )
					throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
			}
			else {
				if ( parts.length != 5 && parts.length != 11 )
					throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
			}
			OutputInfo oi = OutputInfo.stringToOutputInfo(fmt);
			InputInfo ii = OutputInfo.getMatchingInputInfo(oi);
			
			MatrixCharacteristics mc = new MatrixCharacteristics();
			if ( parts.length == 5 ) {
				// do nothing
				;
			}
			else if ( parts.length >= 10 ) {
				// matrix characteristics
				mc.setDimension(Long.parseLong(parts[5]), Long.parseLong(parts[6]));
				mc.setBlockSize(Integer.parseInt(parts[7]), Integer.parseInt(parts[8]));
				mc.setNonZeros(Long.parseLong(parts[9]));
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
			}
			MatrixFormatMetaData iimd = new MatrixFormatMetaData(mc, oi, ii);
			boolean updateInPlace = false;
			if ( parts.length >= 11 )
				updateInPlace = Boolean.parseBoolean(parts[10]);
			
			if ( fmt.equalsIgnoreCase("csv") ) {
				/*
				 * Cretevar instructions for CSV format either has 13 or 14 inputs.
				 * 13 inputs: createvar corresponding to WRITE -- includes properties hasHeader, delim, and sparse
				 * 14 inputs: createvar corresponding to READ -- includes properties hasHeader, delim, fill, and fillValue
				 */
				FileFormatProperties fmtProperties = null;
				if ( parts.length == 14 ) {
					boolean hasHeader = Boolean.parseBoolean(parts[11]);
					String delim = parts[12];
					boolean sparse = Boolean.parseBoolean(parts[13]);
					fmtProperties = new CSVFileFormatProperties(hasHeader, delim, sparse) ;
				}
				else {
					boolean hasHeader = Boolean.parseBoolean(parts[11]);
					String delim = parts[12];
					boolean fill = Boolean.parseBoolean(parts[13]);
					double fillValue = UtilFunctions.parseToDouble(parts[14]);
					String naStrings = null;
					if ( parts.length == 16 )
						naStrings = parts[15];
					fmtProperties = new CSVFileFormatProperties(hasHeader, delim, fill, fillValue, naStrings) ;
				}
				return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, in3, iimd, updateInPlace, parts.length, fmtProperties, opcode, str);
			}
			else {
				return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, in3, iimd, updateInPlace, parts.length, opcode, str);
			}
		case AssignVariable:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			//if ( in1.getValueType() != in2.getValueType() ) 
			//	throw new DMLRuntimeException("Value type mismatch while assigning variables ("+in1.getValueType()+", "+in2.getValueType()+").");
			break;
			
		case CopyVariable:
			// Value types are not given here
			in1 = new CPOperand(parts[1], ValueType.UNKNOWN, DataType.UNKNOWN);
			in2 = new CPOperand(parts[2], ValueType.UNKNOWN, DataType.UNKNOWN);
			break;
			
		case MoveVariable:
			in1 = new CPOperand(parts[1], ValueType.UNKNOWN, DataType.UNKNOWN);
			in2 = new CPOperand(parts[2], ValueType.UNKNOWN, DataType.UNKNOWN);
			if(parts.length > 3)
				in3 = new CPOperand(parts[3], ValueType.UNKNOWN, DataType.UNKNOWN);
			break;
			
		case RemoveVariable:
			in1 = new CPOperand(parts[1], ValueType.UNKNOWN, DataType.SCALAR);
			break;
			
		case RemoveVariableAndFile:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			// second argument must be a boolean
			if ( in2.getValueType() != ValueType.BOOLEAN)
				throw new DMLRuntimeException("Unexpected value type for second argument in: " + str);
			break;
			
		case CastAsScalarVariable:
		case CastAsMatrixVariable:
		case CastAsFrameVariable:
		case CastAsDoubleVariable:
		case CastAsIntegerVariable:
		case CastAsBooleanVariable:			
			in1 = new CPOperand(parts[1]); // first operand is a variable name => string value type 
			out = new CPOperand(parts[2]); // output variable name 
			break;
	
		case Write:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			in3 = new CPOperand(parts[3]);
			
			VariableCPInstruction inst = new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, in3, out, _arity, opcode, str); 
			
			if ( in3.getName().equalsIgnoreCase("csv") ) {
				boolean hasHeader = Boolean.parseBoolean(parts[4]);
				String delim = parts[5];
				boolean sparse = Boolean.parseBoolean(parts[6]);
				FileFormatProperties formatProperties = new CSVFileFormatProperties(hasHeader, delim, sparse);
				inst.setFormatProperties(formatProperties);
			}
			return inst;
			
		case Read:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			out = null;
			break;
			
		case SetFileName:
			in1 = new CPOperand(parts[1]); // variable name
			in2 = new CPOperand(parts[2], ValueType.UNKNOWN, DataType.UNKNOWN); // file name
			in3 = new CPOperand(parts[3], ValueType.UNKNOWN, DataType.UNKNOWN); // option: remote or local
			//return new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, in3, str);
			break;
		
		}
		return new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, in3, out, _arity, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{	
		switch ( opcode ) 
		{ 
		case CreateVariable:
			
			if ( input1.getDataType() == DataType.MATRIX ) {
				//create new variable for symbol table and cache
				//(existing objects gets cleared through rmvar instructions)
				String fname = input2.getName();
				
				// check if unique filename needs to be generated
				boolean overrideFileName = ((BooleanObject) ec.getScalarInput(input3.getName(), input3.getValueType(), true)).getBooleanValue();; //!(input1.getName().startsWith("p")); //    
				if ( overrideFileName ) {
					fname = fname + "_" + _uniqueVarID.getNextID();
				}
				
				MatrixObject mobj = new MatrixObject(input1.getValueType(), fname );
				mobj.setVarName(input1.getName());
				mobj.setDataType(DataType.MATRIX);
				//clone meta data because it is updated on copy-on-write, otherwise there
				//is potential for hidden side effects between variables.
				mobj.setMetaData((MetaData)metadata.clone());
				mobj.setFileFormatProperties(formatProperties);
				mobj.enableUpdateInPlace(updateInPlace);
				ec.setVariable(input1.getName(), mobj);
				if(DMLScript.STATISTICS && updateInPlace)
					Statistics.incrementTotalUIPVar();
			}
			else if ( input1.getDataType() == DataType.SCALAR ){
				ScalarObject sobj = null;
				ec.setScalarOutput(input1.getName(), sobj);
			}
			else {
				throw new DMLRuntimeException("Unexpected data type: " + input1.getDataType());
			}
			break;
		
		case AssignVariable:
			// assign value of variable to the other
			ec.setScalarOutput(input2.getName(), ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral()));			
			break;
			
		case CopyVariable:
			processCopyInstruction(ec);
			break;
			
		case MoveVariable:
			processMoveInstruction(ec);
			break;
			
		case RemoveVariable:
			processRemoveVariableInstruction(ec, input1.getName());
			break;
			
		case RemoveVariableAndFile:
			 // Remove the variable from HashMap _variables, and possibly delete the data on disk. 
			boolean del = ( (BooleanObject) ec.getScalarInput(input2.getName(), input2.getValueType(), true) ).getBooleanValue();
			MatrixObject m = (MatrixObject) ec.removeVariable(input1.getName());
			
			if ( !del ) {
				// HDFS file should be retailed after clearData(), 
				// therefore data must be exported if dirty flag is set
				if ( m.isDirty() )
					m.exportData();
			}
			else {
				//throw new DMLRuntimeException("rmfilevar w/ true is not expected! " + instString);
				//cleanDataOnHDFS(pb, input1.getName());
				cleanDataOnHDFS( m );
			}
			
			// check if in-memory object can be cleaned up
			if ( !ec.getVariables().hasReferences(m) ) {
				// no other variable in the symbol table points to the same Data object as that of input1.getName()
				
				//remove matrix object from cache
				m.clearData();
			}

			break;
			
		case CastAsScalarVariable: //castAsScalarVariable
			MatrixBlock mBlock = ec.getMatrixInput(input1.getName());
			if( mBlock.getNumRows()!=1 || mBlock.getNumColumns()!=1 )
				throw new DMLRuntimeException("Dimension mismatch - unable to cast matrix '"+input1.getName()+"' of dimension ("+mBlock.getNumRows()+" x "+mBlock.getNumColumns()+") to scalar.");
			double value = mBlock.getValue(0,0);
			ec.releaseMatrixInput(input1.getName());
			ec.setScalarOutput(output.getName(), new DoubleObject(value));
			break;
		case CastAsMatrixVariable:{
			MatrixBlock out = null;
			if( input1.getDataType()==DataType.SCALAR ) {
				ScalarObject scalarInput = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
				out = new MatrixBlock(1,1,false);
				out.quickSetValue(0, 0, scalarInput.getDoubleValue());	
			}
			else { //DataType.FRAME
				FrameBlock fin = ec.getFrameInput(input1.getName());
				out = DataConverter.convertToMatrixBlock(fin);
				ec.releaseFrameInput(input1.getName());
			}
			ec.setMatrixOutput(output.getName(), out);
			break;
		}
		case CastAsFrameVariable:{
			FrameBlock out = null;
			if( input1.getDataType()==DataType.SCALAR ) {
				ScalarObject scalarInput = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
				out = new FrameBlock(1, input1.getValueType());
				out.set(0, 0, scalarInput.getStringValue());	
			}
			else { //DataType.FRAME
				MatrixBlock min = ec.getMatrixInput(input1.getName());
				out = DataConverter.convertToFrameBlock(min);
				ec.releaseMatrixInput(input1.getName());
			}
			ec.setFrameOutput(output.getName(), out);
			break;
		}
		case CastAsDoubleVariable:{ 
			ScalarObject scalarInput = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
			ec.setScalarOutput(output.getName(), new DoubleObject(scalarInput.getDoubleValue()));
			break;
		}
		case CastAsIntegerVariable:{ 
			ScalarObject scalarInput = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
			ec.setScalarOutput(output.getName(), new IntObject(scalarInput.getLongValue()));
			break;
		}
		case CastAsBooleanVariable:{ 
			ScalarObject scalarInput = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
			ec.setScalarOutput(output.getName(), new BooleanObject(scalarInput.getBooleanValue()));
			break;
		}
			
		case Read:
			ScalarObject res = null;
			try {
				switch(input1.getValueType()) {
				case DOUBLE:
					double d = MapReduceTool.readDoubleFromHDFSFile(input2.getName());
					res = (ScalarObject) new DoubleObject(d);
					break;
				case INT:
					long i = MapReduceTool.readIntegerFromHDFSFile(input2.getName());
					res = (ScalarObject) new IntObject(i);
					break;
				case BOOLEAN:
					boolean b = MapReduceTool.readBooleanFromHDFSFile(input2.getName());
					res = (ScalarObject) new BooleanObject(b);
					break;
				case STRING:
					String s = MapReduceTool.readStringFromHDFSFile(input2.getName());
					res = (ScalarObject) new StringObject(s);
					break;
					default:
						throw new DMLRuntimeException("Invalid value type (" + input1.getValueType() + ") while processing readScalar instruction.");
				}
			} catch ( IOException e ) {
				throw new DMLRuntimeException(e);
			}
			ec.setScalarOutput(input1.getName(), res);
			
			break;
			
		case Write:
			processWriteInstruction(ec);
			break;
			
		case SetFileName:
			Data data = ec.getVariable(input1.getName());
			if ( data.getDataType() == DataType.MATRIX ) {
				if ( input3.getName().equalsIgnoreCase("remote") ) {
					((MatrixObject)data).setFileName(input2.getName());
				}
				else {
					throw new DMLRuntimeException("Invalid location (" + input3.getName() + ") in SetFileName instruction: " + instString);
				}
			} else{
				throw new DMLRuntimeException("Invalid data type (" + input1.getDataType() + ") in SetFileName instruction: " + instString);
			}
			break;
	
		default:
			throw new DMLRuntimeException("Unknown opcode: " + opcode );
		}
	}	
	
	/**
	 * Handler for mvvar instructions.
	 * Example: mvvar <srcvar> <destFile> <format>
	 * Move the file pointed by srcvar to destFile. 
	 * Currently, applicable only when format=binaryblock.
	 *  
	 * @param ec
	 * @throws DMLRuntimeException
	 */
	private void processMoveInstruction(ExecutionContext ec) throws DMLRuntimeException {
		
		if ( input3 == null ) {
			// example: mvvar tempA A
			
			// get source variable 
			Data srcData = ec.getVariable(input1.getName());		
				
			if ( srcData == null ) 
				throw new DMLRuntimeException("Unexpected error: could not find a data object for variable name:" + input1.getName() + ", while processing instruction " +this.toString());
				
			// remove existing variable bound to target name
			Data tgt = ec.removeVariable(input2.getName());
				
			//cleanup matrix data on fs/hdfs (if necessary)
			if ( tgt != null && tgt instanceof MatrixObject ) {
				ec.cleanupMatrixObject((MatrixObject) tgt);
			}
			
			// do the actual move
			ec.setVariable(input2.getName(), srcData);
			ec.removeVariable(input1.getName());
		}
		else {
			// example instruction: mvvar <srcVar> <destFile> <format>
			if ( ec.getVariable(input1.getName()) == null ) 
				throw new DMLRuntimeException("Unexpected error: could not find a data object for variable name:" + input1.getName() + ", while processing instruction " +this.toString());
			
			MatrixObject mo = (MatrixObject) ec.getVariable(input1.getName());
			if ( input3.getName().equalsIgnoreCase("binaryblock") ) {
				boolean success = mo.moveData(input2.getName(), input3.getName());
				if (!success) {
					throw new DMLRuntimeException("Failed to move var " + input1.getName() + " to file " + input2.getName() + ".");
				}
			}
			else 
				throw new DMLRuntimeException("Unexpected formats while copying: from blocks [" 
							+ mo.getNumRowsPerBlock() + "," + mo.getNumColumnsPerBlock() + "] to " + input3.getName());
		}
	}
	
	/**
	 * Handler for cpvar instructions.
	 * Example: cpvar <srcvar> <destvar>
	 * 
	 * @param ec
	 * @throws DMLRuntimeException 
	 */
	private void processCopyInstruction(ExecutionContext ec) throws DMLRuntimeException {
		// get source variable 
		Data dd = ec.getVariable(input1.getName());		
			
		if ( dd == null ) 
			throw new DMLRuntimeException("Unexpected error: could not find a data object for variable name:" + input1.getName() + ", while processing instruction " +this.toString());
			
		// remove existing variable bound to target name
		Data input2_data = ec.removeVariable(input2.getName());
			
		//cleanup matrix data on fs/hdfs (if necessary)
		if ( input2_data != null && input2_data instanceof MatrixObject ) {
			ec.cleanupMatrixObject((MatrixObject) input2_data);
		}
		
		// do the actual copy!
		ec.setVariable(input2.getName(), dd);
	}
	
	/**
	 * Handler for write instructions.
	 * 
	 * Non-native formats like MM and CSV are handled through specialized helper functions.
	 * The default behavior is to write out the specified matrix from the instruction, in 
	 * the format given by the corresponding symbol table entry.
	 * 
	 * @throws DMLRuntimeException 
	 */
	private void processWriteInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{
		//get filename (literal or variable expression)
		String fname = ec.getScalarInput(input2.getName(), ValueType.STRING, input2.isLiteral()).getStringValue();
		
		if ( input1.getDataType() == DataType.SCALAR ) {
			writeScalarToHDFS(ec, fname);
		}
		else 
		{
			String outFmt = input3.getName();
			
			if (outFmt.equalsIgnoreCase("matrixmarket")) {
				writeMMFile(ec, fname);
			}
			else if (outFmt.equalsIgnoreCase("csv") ) {
				writeCSVFile(ec, fname);
			}
			else {
				// Default behavior
				MatrixObject mo = (MatrixObject)ec.getVariable(input1.getName());
				mo.exportData(fname, outFmt);
			}
		}
	}
	
	/**
	 * Remove variable instruction externalized as a static function in order to allow various 
	 * cleanup procedures to use the same codepath as the actual rmVar instruction
	 * 
	 * @param ec
	 * @param varname
	 * @throws DMLRuntimeException
	 */
	public static void processRemoveVariableInstruction( ExecutionContext ec, String varname ) 
		throws DMLRuntimeException
	{
		// remove variable from symbol table
		Data input1_data = ec.removeVariable(varname);
		
		if ( input1_data == null )
			throw new DMLRuntimeException("Unexpected error: could not find a data object for variable name:" + varname + ", while processing rmvar instruction.");

		//cleanup matrix data on fs/hdfs (if necessary)
		if ( input1_data instanceof MatrixObject ) {
			ec.cleanupMatrixObject( (MatrixObject) input1_data );
		}
	}
	
	/**
	 * Helper function to write CSV files to HDFS.
	 * 
	 * @param ec
	 * @throws DMLRuntimeException
	 */
	private void writeCSVFile(ExecutionContext ec, String fname) 
		throws DMLRuntimeException 
	{
		MatrixObject mo = (MatrixObject)ec.getVariable(input1.getName());
		String outFmt = "csv";
		
		if(mo.isDirty()) {
			// there exist data computed in CP that is not backed up on HDFS
			// i.e., it is either in-memory or in evicted space
			mo.exportData(fname, outFmt, formatProperties);
		}
		else {
			try {
				OutputInfo oi = ((MatrixFormatMetaData)mo.getMetaData()).getOutputInfo();
				MatrixCharacteristics mc = ((MatrixFormatMetaData)mo.getMetaData()).getMatrixCharacteristics();
				if(oi == OutputInfo.CSVOutputInfo) {
					WriterTextCSV writer = new WriterTextCSV((CSVFileFormatProperties)formatProperties);
					writer.addHeaderToCSV(mo.getFileName(), fname, mc.getRows(), mc.getCols());
				}
				else if ( oi == OutputInfo.BinaryBlockOutputInfo || oi == OutputInfo.TextCellOutputInfo ) {
					mo.exportData(fname, outFmt, formatProperties);
				}
				else {
					throw new DMLRuntimeException("Unexpected data format (" + OutputInfo.outputInfoToString(oi) + "): can not export into CSV format.");
				}
				
				// Write Metadata file
				MapReduceTool.writeMetaDataFile (fname + ".mtd", mo.getValueType(), mc, OutputInfo.CSVOutputInfo, formatProperties);
			} catch (IOException e) {
				throw new DMLRuntimeException(e);
			}
		}
	}
	
	/**
	 * Helper function to write MM files to HDFS.
	 * @param ec
	 * @throws DMLRuntimeException
	 */
	private void writeMMFile(ExecutionContext ec, String fname) 
		throws DMLRuntimeException 
	{
		MatrixObject mo = (MatrixObject)ec.getVariable(input1.getName());
		String outFmt = "matrixmarket";
		if(mo.isDirty()) {
			// there exist data computed in CP that is not backed up on HDFS
			// i.e., it is either in-memory or in evicted space
			mo.exportData(fname, outFmt);
		}
		else {
			OutputInfo oi = ((MatrixFormatMetaData)mo.getMetaData()).getOutputInfo();
			MatrixCharacteristics mc = mo.getMatrixCharacteristics();
			if(oi == OutputInfo.TextCellOutputInfo) {
				try {
					WriterMatrixMarket writer = new WriterMatrixMarket();
					writer.mergeTextcellToMatrixMarket(mo.getFileName(), fname, mc.getRows(), mc.getCols(), mc.getNonZeros());
				} catch (IOException e) {
					throw new DMLRuntimeException(e);
				}
			}
			else if ( oi == OutputInfo.BinaryBlockOutputInfo) {
				mo.exportData(fname, outFmt);
			}
			else {
				throw new DMLRuntimeException("Unexpected data format (" + OutputInfo.outputInfoToString(oi) + "): can not export into MatrixMarket format.");
			}
		}
	}
	/**
	 * Helper function to write scalars to HDFS based on its value type.
	 * @throws DMLRuntimeException 
	 */
	private void writeScalarToHDFS(ExecutionContext ec, String fname) 
		throws DMLRuntimeException 
	{
		ScalarObject scalar = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
		try {
			switch ( input1.getValueType() ) {
			case DOUBLE:
				MapReduceTool.writeDoubleToHDFS(scalar.getDoubleValue(), fname);
				break;
			case INT:
				MapReduceTool.writeIntToHDFS(scalar.getLongValue(), fname);
				break;
			case BOOLEAN:
				MapReduceTool.writeBooleanToHDFS(scalar.getBooleanValue(), fname);
				break;
			case STRING:
				MapReduceTool.writeStringToHDFS(scalar.getStringValue(), fname);
				break;
			default:
				throw new DMLRuntimeException("Invalid value type (" + input1.getValueType() + ") in writeScalar instruction: " + instString);
			}
		  // write out .mtd file
		  MapReduceTool.writeScalarMetaDataFile(fname +".mtd", input1.getValueType());
		} catch ( IOException e ) {
			throw new DMLRuntimeException(e);
		}
	}
	
	private static void cleanDataOnHDFS(MatrixObject mo) 
		throws DMLRuntimeException 
	{
		try {
			String fpath = mo.getFileName();
			if (fpath != null) {
				MapReduceTool.deleteFileIfExistOnHDFS(fpath);
				MapReduceTool.deleteFileIfExistOnHDFS(fpath + ".mtd");
			}
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}
	
	public static Instruction prepareRemoveInstruction(String varName) throws DMLRuntimeException {
		StringBuilder sb = new StringBuilder();
		sb.append("CP");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append("rmvar");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(varName);
		String str = sb.toString();

		return parseInstruction(str);
	}
	
	public static Instruction prepareCopyInstruction(String srcVar, String destVar) throws DMLRuntimeException {
		StringBuilder sb = new StringBuilder();
		sb.append("CP");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append("cpvar");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(srcVar);
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(destVar);
		String str = sb.toString();

		return parseInstruction(str);
	}
	
	public static Instruction prepareMoveInstruction(String srcVar, String destFileName, String format) throws DMLRuntimeException {
		StringBuilder sb = new StringBuilder();
		sb.append("CP");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append("mvvar");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(srcVar);
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(destFileName);
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(format);
		String str = sb.toString();

		return parseInstruction(str);
	}
	
	public static Instruction prepareMoveInstruction(String srcVar, String destVar) throws DMLRuntimeException {
		// example: mvvar tempA A 
		// (instead of two instructions -- cpvar tempA A; rmvar tempA)
		StringBuilder sb = new StringBuilder();
		sb.append("CP");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append("mvvar");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(srcVar);
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(destVar);
		String str = sb.toString();

		return parseInstruction(str);
	}
	
	private static String getBasicCreateVarString(String varName, String fileName, boolean fNameOverride, String format) {
		StringBuilder sb = new StringBuilder();
		sb.append("CP");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append("createvar");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(varName); 
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(fileName);		// Constant CREATEVAR_FILE_NAME_VAR_POS is used to find a position of filename within a string generated through this function.
									// If this position of filename within this string changes then constant CREATEVAR_FILE_NAME_VAR_POS to be updated.
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(fNameOverride);
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(format);
		return sb.toString();
	}
	
	public static Instruction prepareCreateVariableInstruction(String varName, String fileName, boolean fNameOverride, String format) throws DMLRuntimeException {
		return parseInstruction(getBasicCreateVarString(varName, fileName, fNameOverride, format));
	}
	
	public static Instruction prepareCreateVariableInstruction(String varName, String fileName, boolean fNameOverride, String format, MatrixCharacteristics mc) throws DMLRuntimeException {
		StringBuilder sb = new StringBuilder();
		sb.append(getBasicCreateVarString(varName, fileName, fNameOverride, format));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getRows());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getCols());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getRowsPerBlock());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getColsPerBlock());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getNonZeros());
		
		String str = sb.toString();

		return parseInstruction(str);
	}	
	
	public static Instruction prepareCreateVariableInstruction(String varName, String fileName, boolean fNameOverride, String format, MatrixCharacteristics mc, boolean updateInPlace) throws DMLRuntimeException {
		StringBuilder sb = new StringBuilder();
		sb.append(getBasicCreateVarString(varName, fileName, fNameOverride, format));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getRows());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getCols());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getRowsPerBlock());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getColsPerBlock());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getNonZeros());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(updateInPlace);
		
		String str = sb.toString();

		return parseInstruction(str);
	}	
	
	public static Instruction prepareCreateVariableInstruction(String varName, String fileName, boolean fNameOverride, String format, MatrixCharacteristics mc, boolean updateInPlace, boolean hasHeader, String delim, boolean sparse) throws DMLRuntimeException {
		StringBuilder sb = new StringBuilder();
		sb.append(getBasicCreateVarString(varName, fileName, fNameOverride, format));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getRows());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getCols());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getRowsPerBlock());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getColsPerBlock());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getNonZeros());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(updateInPlace);
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(hasHeader);
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(delim);
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(sparse);
		
		String str = sb.toString();

		return parseInstruction(str);
	}	
	
	@Override
	public void updateInstructionThreadID(String pattern, String replace)
		throws DMLRuntimeException
	{
		if(    opcode == VariableOperationCode.CreateVariable
			|| opcode == VariableOperationCode.SetFileName )
		{
			//replace in-memory instruction
			input2.setName(input2.getName().replaceAll(pattern, replace));

			// Find a start position of file name string.
			int iPos = StringUtils.ordinalIndexOf(instString, Lop.OPERAND_DELIMITOR, CREATEVAR_FILE_NAME_VAR_POS); 
			// Find a end position of file name string.
			int iPos2 = StringUtils.indexOf(instString, Lop.OPERAND_DELIMITOR, iPos+1);
			
			StringBuilder sb = new StringBuilder();
			sb.append(instString.substring(0,iPos+1));			// It takes first part before file name.
			// This will replace 'pattern' with 'replace' string from file name.
			sb.append(ProgramConverter.saveReplaceFilenameThreadID(instString.substring(iPos+1, iPos2+1), pattern, replace));  
			sb.append(instString.substring(iPos2+1));			// It takes last part after file name.
			
			instString = sb.toString();
		}
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isVariableCastInstruction()
	{
		return ( opcode == VariableOperationCode.CastAsScalarVariable  ||
				 opcode == VariableOperationCode.CastAsMatrixVariable  ||
				 opcode == VariableOperationCode.CastAsIntegerVariable ||
				 opcode == VariableOperationCode.CastAsDoubleVariable  ||
				 opcode == VariableOperationCode.CastAsBooleanVariable );
	}
}

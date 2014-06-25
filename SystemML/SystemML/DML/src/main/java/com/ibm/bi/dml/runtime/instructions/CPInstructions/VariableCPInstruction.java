/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import java.io.IOException;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.MetaData;
import com.ibm.bi.dml.runtime.matrix.io.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.io.FileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


public class VariableCPInstruction extends CPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		RemoveVariable, 
		CopyVariable, 
		RemoveVariableAndFile,		 
		CastAsScalarVariable, 
		CastAsMatrixVariable,
		CastAsDoubleVariable,
		CastAsIntegerVariable,
		CastAsBooleanVariable,
		ValuePick, 
		InMemValuePick, 
		InMemIQM, 
		IQSize, 
		Write, 
		Read, 
		SetFileName, 
		SequenceIncrement,
	}
	
	private static IDSequence _uniqueVarID;	
	
	private VariableOperationCode opcode;
	private CPOperand input1;
	private CPOperand input2;
	private CPOperand input3;
	private CPOperand output;
	private MetaData metadata;
	
	// CSV related members (used only in createvar instructions)
	private FileFormatProperties formatProperties;
	
	static {
		_uniqueVarID  = new IDSequence(true); 
	}
	
	private static VariableOperationCode getVariableOperationCode ( String str ) throws DMLUnsupportedOperationException {
		
		if ( str.equalsIgnoreCase("createvar"))
			return VariableOperationCode.CreateVariable;
		
		else if ( str.equalsIgnoreCase("assignvar"))
			return VariableOperationCode.AssignVariable;
		
		else if ( str.equalsIgnoreCase("rmvar") ) 
			return VariableOperationCode.RemoveVariable;
		
		else if ( str.equalsIgnoreCase("cpvar"))
			return VariableOperationCode.CopyVariable;
		
		else if ( str.equalsIgnoreCase("rmfilevar") ) 
			return VariableOperationCode.RemoveVariableAndFile;
		
		else if ( str.equalsIgnoreCase(UnaryCP.CAST_AS_SCALAR_OPCODE) ) 
			return VariableOperationCode.CastAsScalarVariable;
		
		else if ( str.equalsIgnoreCase(UnaryCP.CAST_AS_MATRIX_OPCODE) ) 
			return VariableOperationCode.CastAsMatrixVariable;
		
		else if ( str.equalsIgnoreCase(UnaryCP.CAST_AS_DOUBLE_OPCODE) ) 
			return VariableOperationCode.CastAsDoubleVariable;
		
		else if ( str.equalsIgnoreCase(UnaryCP.CAST_AS_INT_OPCODE) ) 
			return VariableOperationCode.CastAsIntegerVariable;
		
		else if ( str.equalsIgnoreCase(UnaryCP.CAST_AS_BOOLEAN_OPCODE) ) 
			return VariableOperationCode.CastAsBooleanVariable;
		
		else if ( str.equalsIgnoreCase("valuepick") ) 
			return VariableOperationCode.ValuePick;
		
		else if ( str.equalsIgnoreCase("inmem-valuepick") ) 
			return VariableOperationCode.InMemValuePick;
		
		else if ( str.equalsIgnoreCase("inmem-iqm") ) 
			return VariableOperationCode.InMemIQM;
		
		else if ( str.equalsIgnoreCase("iqsize") ) 
			return VariableOperationCode.IQSize;
		
		else if ( str.equalsIgnoreCase("write") ) 
			return VariableOperationCode.Write;
		
		else if ( str.equalsIgnoreCase("read") ) 
			return VariableOperationCode.Read;
		
		else if ( str.equalsIgnoreCase("setfilename") ) 
			return VariableOperationCode.SetFileName;
		
		else if ( str.equalsIgnoreCase("seqincr") ) 
			return VariableOperationCode.SequenceIncrement;
		
		else
			throw new DMLUnsupportedOperationException("Invalid function: " + str);
	}
	
	// Checks if this instructon is a remove instruction for varName
	public boolean isRemoveVariable(String varName) {
		if ( opcode == VariableOperationCode.RemoveVariable || opcode == VariableOperationCode.RemoveVariableAndFile) {
			if ( input1.get_name().equalsIgnoreCase(varName))
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
	
	public VariableCPInstruction()
	{
	
	}
	
	public VariableCPInstruction (VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, int _arity, String istr )
	{
		super();
		cptype = CPINSTRUCTION_TYPE.Variable;
		opcode = op;
		input1 = in1;
		input2 = in2;
		input3 = in3;
		output = out;
		instString = istr;
		
		formatProperties = null;
	}

	// This version of the constructor is used only in case of CreateVariable
	public VariableCPInstruction (VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, MetaData md, int _arity, String istr)
	{
		this(op, in1, in2, in3, (CPOperand)null, _arity, istr);
		metadata = md;
	}
	
	// This version of the constructor is used only in case of CreateVariable
	public VariableCPInstruction (VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, MetaData md, int _arity, FileFormatProperties formatProperties, String istr)
	{
		this(op, in1, in2, in3, (CPOperand)null, _arity, istr);
		metadata = md;
		this.formatProperties = formatProperties;
	}
	
	public void setFormatProperties(FileFormatProperties prop) {
		formatProperties = prop;
	}
	
	public String getOutputVariableName(){
		String ret = null;
		if( output != null )
			ret = output.get_name();
		return ret;
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
		case SequenceIncrement:
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
			if ( parts.length < 5 )  //&& parts.length != 10 )
				throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
		}
		else if ( voc == VariableOperationCode.Write ) {
			// All write instructions have 3 parameters, except in case of delimited/csv file.
			// Write instructions for csv files also include three additional parameters (hasHeader, delimiter, sparse)
			if ( parts.length != 4 && parts.length != 7 )
				throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
		}
		else {
			_arity = getArity(voc);
			InstructionUtils.checkNumFields ( str, _arity ); // no output
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
				if ( parts.length != 13 && parts.length != 14 )
					throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
			}
			else {
				if ( parts.length != 5 && parts.length != 10 )
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
			
			if ( fmt.equalsIgnoreCase("csv") ) {
				/*
				 * Cretevar instructions for CSV format either has 13 or 14 inputs.
				 * 13 inputs: createvar corresponding to WRITE -- includes properties hasHeader, delim, and sparse
				 * 14 inputs: createvar corresponding to READ -- includes properties hasHeader, delim, fill, and fillValue
				 */
				FileFormatProperties fmtProperties = null;
				if ( parts.length == 13 ) {
					boolean hasHeader = Boolean.parseBoolean(parts[10]);
					String delim = parts[11];
					boolean sparse = Boolean.parseBoolean(parts[12]);
					fmtProperties = new CSVFileFormatProperties(hasHeader, delim, sparse) ;
				}
				else {
					boolean hasHeader = Boolean.parseBoolean(parts[10]);
					String delim = parts[11];
					boolean fill = Boolean.parseBoolean(parts[12]);
					double fillValue = Double.parseDouble(parts[13]);
					fmtProperties = new CSVFileFormatProperties(hasHeader, delim, fill, fillValue) ;
				}
				return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, in3, iimd, parts.length, fmtProperties, str);
			}
			else {
				return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, in3, iimd, parts.length, str);
			}
		case AssignVariable:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			//if ( in1.get_valueType() != in2.get_valueType() ) 
			//	throw new DMLRuntimeException("Value type mismatch while assigning variables ("+in1.get_valueType()+", "+in2.get_valueType()+").");
			break;
			
		case RemoveVariable:
			in1 = new CPOperand(parts[1], ValueType.UNKNOWN, DataType.SCALAR);
			break;
			
		case CopyVariable:
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
			
		case CastAsScalarVariable:
		case CastAsMatrixVariable:
		case CastAsDoubleVariable:
		case CastAsIntegerVariable:
		case CastAsBooleanVariable:			
			in1 = new CPOperand(parts[1]); // first operand is a variable name => string value type 
			out = new CPOperand(parts[2]); // output variable name 
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
			
		case Write:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			in3 = new CPOperand(parts[3]);
			
			VariableCPInstruction inst = new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, in3, out, _arity, str); 
			
			if ( in3.get_name().equalsIgnoreCase("csv") ) {
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
		
		case SequenceIncrement:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			out = new CPOperand(parts[3]);
			break;
				
		}
		return new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, in3, out, _arity, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{	
		switch ( opcode ) 
		{ 
		case CreateVariable:
			
			if ( input1.get_dataType() == DataType.MATRIX ) {
				//create new variable for symbol table and cache
				//(existing objects gets cleared through rmvar instructions)
				String fname = input2.get_name();
				
				// check if unique filename needs to be generated
				boolean overrideFileName = ((BooleanObject) ec.getScalarInput(input3.get_name(), input3.get_valueType(), true)).getBooleanValue();; //!(input1.get_name().startsWith("p")); //    
				if ( overrideFileName ) {
					fname = fname + "_" + _uniqueVarID.getNextID();
				}
				
				MatrixObject mobj = new MatrixObject(input1.get_valueType(), fname );
				mobj.setVarName(input1.get_name());
				mobj.setDataType(DataType.MATRIX);
				mobj.setMetaData(metadata);
				mobj.setFileFormatProperties(formatProperties);
				
				ec.setVariable(input1.get_name(), mobj);
			}
			else if ( input1.get_dataType() == DataType.SCALAR ){
				ScalarObject sobj = null;
				ec.setScalarOutput(input1.get_name(), sobj);
			}
			else {
				throw new DMLRuntimeException("Unexpected data type: " + input1.get_dataType());
			}
			break;
		
		case AssignVariable:
			// assign value of variable to the other
			ec.setScalarOutput(input2.get_name(), ec.getScalarInput(input1.get_name(), input1.get_valueType(), input1.isLiteral()));			
			break;
			
		case RemoveVariable:
			processRemoveVariableInstruction(ec, input1.get_name());
			
			break;
			
		case CopyVariable:
			// example instruction: cpvar <srcVar> <destVar>
			
			Data dd = ec.getVariable(input1.get_name());		
			
			if ( dd == null ) {
				throw new DMLRuntimeException("Unexpected error: could not find a data object for variable name:" + input1.get_name() + ", while processing instruction " +this.toString());
			}
			
			// check if <destVar> has any existing references
			Data input2_data = ec.getVariable(input2.get_name());
			int destRefCount = ec.getVariables().getNumReferences( input2_data, true );
			
			if ( destRefCount == 1 ) {
				// input2.get_name() currently refers to a Data object.
				// make sure to call clearData(), if it is a matrix object 
				
				//System.out.println("  " + this.instString + " ... clearing input2");
				if( input2_data instanceof MatrixObject )
				{
					MatrixObject mo = (MatrixObject) input2_data;
					mo.clearData();
					if ( mo.isFileExists() && mo.isCleanupEnabled() )
						cleanDataOnHDFS( mo );
				}
			
			} /*else if ( destRefCount > 1) {
				System.err.println("  --- " + this.instString + " ... refCount for input2 > 1");
			}*/
			
			// do the actual copy!
			ec.setVariable(input2.get_name(), dd);
			
			break;
			
		case RemoveVariableAndFile:
			 // Remove the variable from HashMap _variables, and possibly delete the data on disk. 
			boolean del = ( (BooleanObject) ec.getScalarInput(input2.get_name(), input2.get_valueType(), true) ).getBooleanValue();
			
			MatrixObject m = (MatrixObject) ec.getVariable(input1.get_name());
			if ( !del ) {
				// HDFS file should be retailed after clearData(), 
				// therefore data must be exported if dirty flag is set
				if ( m.isDirty() )
					m.exportData();
			}
			else {
				//throw new DMLRuntimeException("rmfilevar w/ true is not expected! " + instString);
				//cleanDataOnHDFS(pb, input1.get_name());
				cleanDataOnHDFS( m );
			}
			
			// check if in-memory object can be cleaned up
			int refCnt = ec.getVariables().getNumReferences(m, true);
			if ( refCnt== 1 ) {
				// no other variable in the symbol table points to the same Data object as that of input1.get_name()
				
				//remove matrix object from cache
				m.clearData();
			}
			else if ( refCnt == 0 ) 
				throw new DMLRuntimeException("  " + this.toString() + " -- refCount=0 is unexpected!");

			// remove the variable from the HashMap (_variables) in ProgramBlock.
			ec.removeVariable( input1.get_name() );
			break;
			
		case CastAsScalarVariable: //castAsScalarVariable
			MatrixBlock mBlock = ec.getMatrixInput(input1.get_name());
			if( mBlock.getNumRows()!=1 || mBlock.getNumColumns()!=1 )
				throw new DMLRuntimeException("Dimension mismatch - unable to cast matrix of dimension ("+mBlock.getNumRows()+" x "+mBlock.getNumColumns()+") to scalar.");
			double value = mBlock.getValue(0,0);
			ec.releaseMatrixInput(input1.get_name());
			ec.setScalarOutput(output.get_name(), new DoubleObject(value));
			break;
		case CastAsMatrixVariable:{
			ScalarObject scalarInput = ec.getScalarInput(input1.get_name(), input1.get_valueType(), input1.isLiteral());
			MatrixBlock out = new MatrixBlock(1,1,false);
			out.quickSetValue(0, 0, scalarInput.getDoubleValue());
			ec.setMatrixOutput(output.get_name(), out);
			break;
		}
		case CastAsDoubleVariable:{ 
			ScalarObject scalarInput = ec.getScalarInput(input1.get_name(), input1.get_valueType(), input1.isLiteral());
			ec.setScalarOutput(output.get_name(), new DoubleObject(scalarInput.getDoubleValue()));
			break;
		}
		case CastAsIntegerVariable:{ 
			ScalarObject scalarInput = ec.getScalarInput(input1.get_name(), input1.get_valueType(), input1.isLiteral());
			ec.setScalarOutput(output.get_name(), new IntObject(scalarInput.getLongValue()));
			break;
		}
		case CastAsBooleanVariable:{ 
			ScalarObject scalarInput = ec.getScalarInput(input1.get_name(), input1.get_valueType(), input1.isLiteral());
			ec.setScalarOutput(output.get_name(), new BooleanObject(scalarInput.getBooleanValue()));
			break;
		}
		
		case ValuePick:
			// example = valuepickCP:::temp3:DOUBLE:::0.5:DOUBLE:::Var0:DOUBLE
			// pick a value from "temp3" and assign to Var0
			
			MatrixObject mat = (MatrixObject)ec.getVariable(input1.get_name());
			String fname = mat.getFileName();
			MetaData mdata = mat.getMetaData();
			ScalarObject pickindex = ec.getScalarInput(input2.get_name(), input2.get_valueType(), input2.isLiteral());
			
			if ( mdata != null ) {
				try {
					double picked = MapReduceTool.pickValue(fname, (NumItemsByEachReducerMetaData) mdata, pickindex.getDoubleValue());
					ScalarObject result = (ScalarObject) new DoubleObject(picked);
					ec.setVariable(output.get_name(), result);
				} catch (Exception e ) {
					throw new DMLRuntimeException(e);
				}
			}
			else {
				throw new DMLRuntimeException("Unexpected error while executing ValuePickCP: otherMetaData for file (" + fname + ") not found." );
			}
			break;
			
		case InMemValuePick:
			MatrixBlock matBlock = ec.getMatrixInput(input1.get_name());

			if ( input2.get_dataType() == DataType.SCALAR ) {
				ScalarObject quantile = ec.getScalarInput(input2.get_name(), input2.get_valueType(), input2.isLiteral());
				double picked = matBlock.pickValue(quantile.getDoubleValue());
				ec.setScalarOutput(output.get_name(), (ScalarObject) new DoubleObject(picked));
			} 
			else {
				MatrixBlock quantiles = ec.getMatrixInput(input2.get_name());
				MatrixBlock resultBlock = (MatrixBlock) matBlock.pickValues(quantiles, new MatrixBlock());
				quantiles = null;
				ec.releaseMatrixInput(input2.get_name());
				ec.setMatrixOutput(output.get_name(), resultBlock);
			}
			matBlock = null;
			ec.releaseMatrixInput(input1.get_name());
			break;
			
		case InMemIQM:
			MatrixBlock matBlock1 = ec.getMatrixInput(input1.get_name());
			double iqm = matBlock1.interQuartileMean();
			
			matBlock1 = null;
			ec.releaseMatrixInput(input1.get_name());
			ec.setScalarOutput(output.get_name(), (ScalarObject) new DoubleObject(iqm));
			break;
			
		case IQSize:
			// example = iqsize:::temp3:DOUBLE:::0.25:DOUBLE:::Var0:DOUBLE
			// compute the number of records in interquartile
			
			// get otherMetadata associated with the matrix
			String iqs_fname = input1.get_name();
			MetaData iqs_md = ec.getMetaData(iqs_fname);
			
			// ge the range
			ScalarObject pickrange = ec.getScalarInput(input2.get_name(), input2.get_valueType(), input2.isLiteral());
			
			if ( iqs_md != null ) {
				try {
					double rangesize = UtilFunctions.getLengthForInterQuantile((NumItemsByEachReducerMetaData) iqs_md, pickrange.getDoubleValue());
					ScalarObject result = (ScalarObject) new DoubleObject(rangesize);
					ec.setVariable(output.get_name(), result);
				} catch (Exception e ) {
					throw new DMLRuntimeException(e);
				}
			}
			else {
				throw new DMLRuntimeException("Unexpected error while executing IQSize: otherMetaData for file (" + iqs_fname + ") not found." );
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
					long i = MapReduceTool.readIntegerFromHDFSFile(input2.get_name());
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
			ec.setScalarOutput(input1.get_name(), res);
			
			break;
			
		case Write:
			processWriteInstruction(ec);
			break;
			
		case SetFileName:
			Data data = ec.getVariable(input1.get_name());
			if ( data.getDataType() == DataType.MATRIX ) {
				if ( input3.get_name().equalsIgnoreCase("remote") ) {
					((MatrixObject)data).setFileName(input2.get_name());
				}
				else {
					throw new DMLRuntimeException("Invalid location (" + input3.get_name() + ") in SetFileName instruction: " + instString);
				}
			} else{
				throw new DMLRuntimeException("Invalid data type (" + input1.get_dataType() + ") in SetFileName instruction: " + instString);
			}
			break;
			
		case SequenceIncrement:
			ScalarObject fromObj = ec.getScalarInput(input1.get_name(), input1.get_valueType(), input1.isLiteral());
			ScalarObject toObj = ec.getScalarInput(input2.get_name(), input2.get_valueType(), input2.isLiteral());
			double ret = Double.NaN;
			if ( fromObj.getDoubleValue() >= toObj.getDoubleValue() )
				ret = -1.0;
			else
				ret = 1.0;
			ScalarObject incrObj = (ScalarObject) new DoubleObject(ret);
			ec.setVariable(output.get_name(), incrObj);

			break;
			
		default:
			throw new DMLRuntimeException("Unknown opcode: " + opcode );
		}
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
	private void processWriteInstruction(ExecutionContext ec) throws DMLRuntimeException {
		if ( input1.get_dataType() == DataType.SCALAR ) {
			writeScalarToHDFS(ec);
		}
		else {
			String outFmt = input3.get_name();
			
			if (outFmt.equalsIgnoreCase("matrixmarket")) {
				writeMMFile(ec);
			}
			else if (outFmt.equalsIgnoreCase("csv") ) {
				writeCSVFile(ec);
			}
			else {
				// Default behavior
				MatrixObject mo = (MatrixObject)ec.getVariable(input1.get_name());
				mo.exportData(input2.get_name(), outFmt);
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
		Data input1_data = ec.getVariable(varname);
		
		if ( input1_data == null ) {
			throw new DMLRuntimeException("Unexpected error: could not find a data object for variable name:" + varname + ", while processing rmVar instruction.");
		}

		// check if any other variable refers to the same Data object
		int refCount = ec.getVariables().getNumReferences(input1_data, true);
		if ( refCount == 1 ) {
			// no other variable in the symbol table points to the same Data object as that of input1.get_name()
			
			if ( input1_data instanceof MatrixObject ) {
				// clean in-memory object
				MatrixObject mo = (MatrixObject) input1_data;
				mo.clearData();
				
				if ( mo.isFileExists() && mo.isCleanupEnabled() )
					// clean data on hdfs, if exists
					cleanDataOnHDFS( mo );
			}
		}
		else if ( refCount == 0 ) 
			throw new DMLRuntimeException("Error while processing rmVar instruction: refCount=0 is unexpected!");

		// remove variable from the program block
		ec.removeVariable(varname);
	}
	
	/**
	 * Helper function to write CSV files to HDFS.
	 * 
	 * @param ec
	 * @throws DMLRuntimeException
	 */
	private void writeCSVFile(ExecutionContext ec) throws DMLRuntimeException {
		MatrixObject mo = (MatrixObject)ec.getVariable(input1.get_name());
		String outFmt = "csv";
		
		if(mo.isDirty()) {
			// there exist data computed in CP that is not backed up on HDFS
			// i.e., it is either in-memory or in evicted space
			mo.exportData(input2.get_name(), outFmt, formatProperties);
		}
		else {
			try {
				OutputInfo oi = ((MatrixFormatMetaData)mo.getMetaData()).getOutputInfo();
				MatrixCharacteristics mc = ((MatrixFormatMetaData)mo.getMetaData()).getMatrixCharacteristics();
				if(oi == OutputInfo.CSVOutputInfo) {
						DataConverter.mergeCSVPartFiles(mo.getFileName(), input2.get_name(), (CSVFileFormatProperties)formatProperties, mc.get_rows(), mc.get_cols());
				}
				else if ( oi == OutputInfo.BinaryBlockOutputInfo || oi == OutputInfo.TextCellOutputInfo ) {
					mo.exportData(input2.get_name(), outFmt, formatProperties);
				}
				else {
					throw new DMLRuntimeException("Unexpected data format (" + OutputInfo.outputInfoToString(oi) + "): can not export into CSV format.");
				}
				
				// Write Metadata file
				MapReduceTool.writeMetaDataFile (input2.get_name() + ".mtd", mo.getValueType(), mc, OutputInfo.CSVOutputInfo, formatProperties);
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
	private void writeMMFile(ExecutionContext ec) throws DMLRuntimeException {
		MatrixObject mo = (MatrixObject)ec.getVariable(input1.get_name());
		String outFmt = "matrixmarket";
		if(mo.isDirty()) {
			// there exist data computed in CP that is not backed up on HDFS
			// i.e., it is either in-memory or in evicted space
			mo.exportData(input2.get_name(), outFmt);
		}
		else {
			OutputInfo oi = ((MatrixFormatMetaData)mo.getMetaData()).getOutputInfo();
			MatrixCharacteristics mc = ((MatrixFormatMetaData)mo.getMetaData()).getMatrixCharacteristics();
			if(oi == OutputInfo.TextCellOutputInfo) {
				try {
					DataConverter.mergeTextcellToMatrixMarket(mo.getFileName(), input2.get_name(), mc.get_rows(), mc.get_cols(), mc.getNonZeros());
				} catch (IOException e) {
					throw new DMLRuntimeException(e);
				}
			}
			else if ( oi == OutputInfo.BinaryBlockOutputInfo) {
				mo.exportData(input2.get_name(), outFmt);
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
	private void writeScalarToHDFS(ExecutionContext ec) throws DMLRuntimeException {
		ScalarObject scalar = ec.getScalarInput(input1.get_name(), input1.get_valueType(), input1.isLiteral());
		try {
			switch ( input1.get_valueType() ) {
			case DOUBLE:
				MapReduceTool.writeDoubleToHDFS(scalar.getDoubleValue(), input2.get_name());
				break;
			case INT:
				MapReduceTool.writeIntToHDFS(scalar.getLongValue(), input2.get_name());
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
	
	public static Instruction prepareRemoveInstruction(String varName) throws DMLRuntimeException, DMLUnsupportedOperationException {
		// (example
		// "CP+Lops.OPERAND_DELIMITOR+rmvar+Lops.OPERAND_DELIMITOR+Var7")
		StringBuffer sb = new StringBuffer();
		sb.append("CP");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append("rmvar");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(varName);
		String str = sb.toString();

		return parseInstruction(str);
	}
	
	public static Instruction prepareCopyInstruction(String srcVar, String destVar) throws DMLRuntimeException, DMLUnsupportedOperationException {
		// (example
		// "CP+Lops.OPERAND_DELIMITOR+cpvar+Lops.OPERAND_DELIMITOR+mVar7+Lops.OPERAND_DELIMITOR+mVar8")
		StringBuffer sb = new StringBuffer();
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
	
	private static String getBasicCreateVarString(String varName, String fileName, boolean fNameOverride, String format) {
		StringBuffer sb = new StringBuffer();
		sb.append("CP");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append("createvar");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(varName); 
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(fileName);
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(fNameOverride);
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(format);
		return sb.toString();
	}
	
	public static Instruction prepareCreateVariableInstruction(String varName, String fileName, boolean fNameOverride, String format) throws DMLRuntimeException, DMLUnsupportedOperationException {
		return parseInstruction(getBasicCreateVarString(varName, fileName, fNameOverride, format));
	}
	
	public static Instruction prepareCreateVariableInstruction(String varName, String fileName, boolean fNameOverride, String format, MatrixCharacteristics mc) throws DMLRuntimeException, DMLUnsupportedOperationException {
		StringBuffer sb = new StringBuffer();
		sb.append(getBasicCreateVarString(varName, fileName, fNameOverride, format));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.get_rows());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.get_cols());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.get_rows_per_block());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.get_cols_per_block());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(mc.getNonZeros());
		
		String str = sb.toString();

		return parseInstruction(str);
	}	
	
	@Override
	public void updateInstructionThreadID(String pattern, String replace)
	{
		if(    opcode == VariableOperationCode.CreateVariable
			|| opcode == VariableOperationCode.SetFileName )
		{
			//replace in-memory instruction
			input2.set_name(input2.get_name().replaceAll(pattern, replace));
			int pos = 3;
		
			String[] parts = instString.split(Lop.OPERAND_DELIMITOR);
			StringBuilder sb = new StringBuilder();
			for( int i=0; i<parts.length; i++ )
			{
				if( i>0 ) sb.append(Lop.OPERAND_DELIMITOR);
				
				if( i==pos )
					sb.append(ProgramConverter.saveReplaceFilenameThreadID(parts[i], pattern, replace));
				else
					sb.append(parts[i]);
			}
			instString = sb.toString();
		}
	}
}

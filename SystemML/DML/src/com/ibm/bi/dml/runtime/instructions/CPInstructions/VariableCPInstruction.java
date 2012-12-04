package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import java.io.IOException;

import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.MetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.CacheException;
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
	
	protected static IDSequence _uniqueVarID;
	static {
		_uniqueVarID  = new IDSequence(true); 
	}
	private enum VariableOperationCode {
		CreateVariable, AssignVariable, RemoveVariable, CopyVariable, RemoveVariableAndFile, AssignVariableWithFirstValue, ValuePick, InMemValuePick, InMemIQM, IQSize, Write, Read, SetFileName
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
		
		else if ( str.equalsIgnoreCase("cpvar"))
			return VariableOperationCode.CopyVariable;
		
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
		
		else if ( str.equalsIgnoreCase("write") ) 
			return VariableOperationCode.Write;
		
		else if ( str.equalsIgnoreCase("read") ) 
			return VariableOperationCode.Read;
		
		else if ( str.equalsIgnoreCase("setfilename") ) 
			return VariableOperationCode.SetFileName;
		
		else
			throw new DMLUnsupportedOperationException("Invalid function: " + str);
	}
	
	/*public VariableOperationCode getVariableOpCode() {
		return opcode;
	}*/
	
	// Checks if this instructon is a remove instruction for varName
	public boolean isRemoveVariable(String varName) {
		if ( opcode == VariableOperationCode.RemoveVariable || opcode == VariableOperationCode.RemoveVariableAndFile) {
			if ( input1.get_name().equalsIgnoreCase(varName))
				return true;
		}
		return false;
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
		arity = _arity;
		instString = istr;
	}

	// This version of the constructor is used only in case of CreateVariable
	public VariableCPInstruction (VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, MetaData md, int _arity, String istr)
	{
		this(op, in1, in2, in3, (CPOperand)null, _arity, istr);
		metadata = md;
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
			if ( parts.length != 5 && parts.length != 10 )
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
			OutputInfo oi = OutputInfo.stringToOutputInfo(fmt);
			InputInfo ii = OutputInfo.getMatchingInputInfo(oi);
			
			MatrixCharacteristics mc = new MatrixCharacteristics();
			if ( parts.length == 5 ) {
				// do nothing
				;
			}
			else if ( parts.length == 10 ) {
				// matrix characteristics
				mc.setDimension(Long.parseLong(parts[5]), Long.parseLong(parts[6]));
				mc.setBlockSize(Integer.parseInt(parts[7]), Integer.parseInt(parts[8]));
				mc.setNonZeros(Long.parseLong(parts[9]));
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
			}
			MatrixFormatMetaData iimd = new MatrixFormatMetaData(mc, oi, ii);
			return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, in3, iimd, parts.length, str);
			
			/*if ( parts.length > 3 ) {
				MatrixCharacteristics mc = new MatrixCharacteristics();
				if ( parts.length == 4 ) {
					// the last operand is the OutputInfo
					OutputInfo oi = OutputInfo.stringToOutputInfo(parts[3]);
					InputInfo ii = OutputInfo.getMatchingInputInfo(oi);
					MatrixFormatMetaData iimd = new MatrixFormatMetaData(mc, oi, ii);
					return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, parts.length, str, iimd);
				}
				mc.setDimension(Long.parseLong(parts[3]), Long.parseLong(parts[4]));
				mc.setBlockSize(Integer.parseInt(parts[5]), Integer.parseInt(parts[6]));
				mc.setNonZeros(Long.parseLong(parts[7]));
				
				if (parts.length > 8) {
					OutputInfo oi = OutputInfo.stringToOutputInfo(parts[8]);
					InputInfo ii = OutputInfo.getMatchingInputInfo(oi);
					MatrixFormatMetaData iimd = new MatrixFormatMetaData(mc, oi, ii);
					return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, parts.length, str, iimd);
				}
				else {
					MatrixDimensionsMetaData mdmd = new MatrixDimensionsMetaData(mc);
					return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, parts.length, str, mdmd);
				}
			}
			break;*/
			
		case AssignVariable:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			if ( in1.get_valueType() != in2.get_valueType() ) 
				throw new DMLRuntimeException("Value type mismatch while assigning variables.");
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
			
		case AssignVariableWithFirstValue:
			in1 = new CPOperand(parts[1]); // first operand is a variable name => string value type 
			out = new CPOperand(parts[2]); // second operand is a variable and is assumed to be double 
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
			//return new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, in3, str);
			
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
		}
		return new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, in3, out, _arity, str);
	}

	//private int getRefCount(ProgramBlock pb, String var, boolean breakIfMultipleRefs) {
	private int getRefCount(ProgramBlock pb, Data d, boolean breakIfMultipleRefs) {
		
		//Data d = pb.getVariable(var);
		
		if ( d == null )
			return 0;
		
		int refCount = 0;
		//int hash = d.hashCode();
		for( String key : pb.getVariables().keySet() ) {
			//System.out.println("    probing for " + key + ": " + pb.getVariable(key));
			if ( pb.getVariable(key).equals(d) ) {
				refCount++;
				
				//if ( !key.equalsIgnoreCase(var) && !instString.contains("rmvar"))
				//	System.err.println("   " + instString + " .. var " + var + ", key " + key);
				// early exit if specified by "breakIfMultipleRefs"
				if ( breakIfMultipleRefs && refCount > 1)
					return refCount;
			}
		}
		return refCount;
	}
	
	@Override
	public void processInstruction(ProgramBlock pb) throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		switch ( opcode ) { 
		case CreateVariable:
			
			if ( input1.get_dataType() == DataType.MATRIX ) {
				//create new variable for symbol table and cache
				//(existing objects gets cleared through rmvar instructions)
				String fname = input2.get_name();
				
				// check if unique filename needs to be generated
				boolean overrideFileName = ((BooleanObject) pb.getScalarInput(input3.get_name(), input3.get_valueType())).getBooleanValue();; //!(input1.get_name().startsWith("p")); //    
				if ( overrideFileName ) {
					fname = fname + "_" + _uniqueVarID.getNextID();
				}
				
				MatrixObject mobj = new MatrixObject(input1.get_valueType(), fname );
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
			
			Data input1_data = pb.getVariable(input1.get_name());
			
			if ( input1_data == null ) {
				throw new DMLRuntimeException("Unexpected error: could not find a data object for variable name:" + input1.get_name() + ", while processing instruction " +this.toString());
			}

			// check if any other variable refers to the same Data object
			int refCount = getRefCount(pb, input1_data, true);
			if ( refCount == 1 ) {
				// no other variable in the symbol table points to the same Data object as that of input1.get_name()
				
				if ( input1_data instanceof MatrixObject ) {
					// clean in-memory object
					clearCachedMatrixObject(pb, input1_data);
					
					if ( ((MatrixObject) input1_data).isFileExists() && ((MatrixObject) input1_data).isCleanupEnabled() )
						// clean data on hdfs, if exists
						cleanDataOnHDFS(pb, input1_data);
				}
			}
			else if ( refCount == 0 ) 
				throw new DMLRuntimeException("  " + this.toString() + " -- refCount=0 is unexpected!");

			// remove variable from the program block
			pb.removeVariable(input1.get_name());

			break;
			
		case CopyVariable:
			// example instruction: cpvar <srcVar> <destVar>
			
			Data dd = pb.getVariable(input1.get_name());		
			
			if ( dd == null ) {
				throw new DMLRuntimeException("Unexpected error: could not find a data object for variable name:" + input1.get_name() + ", while processing instruction " +this.toString());
			}
			
			// check if <destVar> has any existing references
			int destRefCount = getRefCount(pb, pb.getVariable(input2.get_name()), false);
			
			if ( destRefCount == 1 ) {
				// input2.get_name() currently refers to a Data object.
				// make sure to call clearData(), if it is a matrix object 
				
				//System.out.println("  " + this.instString + " ... clearing input2");
				clearCachedMatrixObject(pb, pb.getVariable(input2.get_name()) );
			
			} /*else if ( destRefCount > 1) {
				System.err.println("  --- " + this.instString + " ... refCount for input2 > 1");
			}*/
			
			// do the actual copy!
			pb.setVariable(input2.get_name(), dd);
			
			break;
			
		case RemoveVariableAndFile:
			 // Remove the variable from HashMap _variables, and possibly delete the data on disk. 
			boolean del = ( (BooleanObject) pb.getScalarInput(input2.get_name(), input2.get_valueType()) ).getBooleanValue();
			
			MatrixObject m = (MatrixObject) pb.getVariable(input1.get_name());
			if ( !del ) {
				// HDFS file should be retailed after clearData(), 
				// therefore data must be exported if dirty flag is set
				if ( m.isDirty() )
					m.exportData();
			}
			else {
				throw new DMLRuntimeException("rmfilevar w/ true is not expected! " + instString);
				//cleanDataOnHDFS(pb, input1.get_name());
			}
			
			// check if in-memory object can be cleaned up
			int refCnt = getRefCount(pb, pb.getVariable(input1.get_name()), true);
			if ( refCnt== 1 ) {
				// no other variable in the symbol table points to the same Data object as that of input1.get_name()
				
				//remove matrix object from cache
				clearCachedMatrixObject( pb, m);
			}
			else if ( refCnt == 0 ) 
				throw new DMLRuntimeException("  " + this.toString() + " -- refCount=0 is unexpected!");

			// remove the variable from the HashMap (_variables) in ProgramBlock.
			pb.removeVariable( input1.get_name() );
			break;
			
		case AssignVariableWithFirstValue:
			MatrixBlock mBlock = (MatrixBlock) pb.getMatrixInput(input1.get_name());
			double value = mBlock.getValue(0,0);
			//double value = MapReduceTool.readFirstNumberFromHDFSMatrix(mobj.getFileName());
			pb.releaseMatrixInput(input1.get_name());
			pb.setScalarOutput(output.get_name(), new DoubleObject(value));
			break;
			
		case ValuePick:
			// example = valuepickCP:::temp3:DOUBLE:::0.5:DOUBLE:::Var0:DOUBLE
			// pick a value from "temp3" and assign to Var0
			
			MatrixObject mat = (MatrixObject)pb.getVariable(input1.get_name());
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
				MatrixObject mo = (MatrixObject)pb.getVariable(input1.get_name());
				mo.exportData(input2.get_name(), input3.get_name());
			}
			break;
			
		case SetFileName:
			Data data = pb.getVariable(input1.get_name());
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
			
		default:
			throw new DMLRuntimeException("Unknown opcode: " + opcode );
		}
	}	

	/**
	 * 
	 * @param pb
	 * @param op
	 * @throws CacheException 
	 */
	public void clearCachedMatrixObject( ProgramBlock pb, Data d) 
		throws CacheException 
	{
		if ( d instanceof MatrixObject ) {
			((MatrixObject)d).clearData();
		}
	}
	
	private void cleanDataOnHDFS(ProgramBlock pb, Data d) 
			throws DMLRuntimeException {
		if (d instanceof MatrixObject ) {
			MatrixObject m = (MatrixObject) d;
			try {
				String fpath = m.getFileName();
				if (fpath != null) {
					MapReduceTool.deleteFileIfExistOnHDFS(fpath);
					// removeMetaData(); // delete in-memory metadata
					MapReduceTool.deleteFileIfExistOnHDFS(fpath + ".mtd");
				}
			} catch (IOException e) {
				throw new DMLRuntimeException(e);
			}
		}
	}
	
	public static Instruction prepareRemoveInstruction(String varName) throws DMLRuntimeException, DMLUnsupportedOperationException {
		// (example
		// "CP+Lops.OPERAND_DELIMITOR+rmvar+Lops.OPERAND_DELIMITOR+Var7")
		StringBuffer sb = new StringBuffer();
		sb.append("CP");
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append("rmvar");
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(varName);
		String str = sb.toString();

		return parseInstruction(str);
	}
	
	public static Instruction prepareCopyInstruction(String srcVar, String destVar) throws DMLRuntimeException, DMLUnsupportedOperationException {
		// (example
		// "CP+Lops.OPERAND_DELIMITOR+cpvar+Lops.OPERAND_DELIMITOR+mVar7+Lops.OPERAND_DELIMITOR+mVar8")
		StringBuffer sb = new StringBuffer();
		sb.append("CP");
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append("cpvar");
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(srcVar);
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(destVar);
		String str = sb.toString();

		return parseInstruction(str);
	}
	
	private static String getBasicCreateVarString(String varName, String fileName, boolean fNameOverride, String format) {
		StringBuffer sb = new StringBuffer();
		sb.append("CP");
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append("createvar");
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(varName); 
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(fileName);
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(fNameOverride);
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(format);
		return sb.toString();
	}
	
	public static Instruction prepareCreateVariableInstruction(String varName, String fileName, boolean fNameOverride, String format) throws DMLRuntimeException, DMLUnsupportedOperationException {
		return parseInstruction(getBasicCreateVarString(varName, fileName, fNameOverride, format));
	}
	
	public static Instruction prepareCreateVariableInstruction(String varName, String fileName, boolean fNameOverride, String format, MatrixCharacteristics mc) throws DMLRuntimeException, DMLUnsupportedOperationException {
		StringBuffer sb = new StringBuffer();
		sb.append(getBasicCreateVarString(varName, fileName, fNameOverride, format));
		
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(mc.get_rows());
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(mc.get_cols());
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(mc.get_rows_per_block());
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(mc.get_cols_per_block());
		sb.append(Lops.OPERAND_DELIMITOR);
		sb.append(mc.getNonZeros());
		
		String str = sb.toString();

		return parseInstruction(str);
	}
	
}

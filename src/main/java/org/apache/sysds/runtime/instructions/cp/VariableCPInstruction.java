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

package org.apache.sysds.runtime.instructions.cp;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.controlprogram.caching.TensorObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.io.*;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.lineage.LineageTraceable;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaData;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.meta.TensorCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.ProgramConverter;
import org.apache.sysds.utils.Statistics;

/*
 * Supported Operations
 * --------------------
 *  1) assignvar x:type y:type
 *      assign value of y to x (both types should match)
 *  2) rmvar x
 *      remove variable x
 *  3) cpvar x y
 *      copy x to y (same as assignvar followed by rmvar, types are not required)
 *  4) rmfilevar x:type b:type
 *      remove variable x, and if b=true then the file object associated with x (b's type should be boolean)
 *  5) assignvarwithfile FN x
 *      assign x with the first value from the file whose name=FN
 *  6) attachfiletovar FP x
 *      allocate a new file object with name FP, and associate it with variable x
 *      createvar x FP [dimensions] [formatinfo]
 */
public class VariableCPInstruction extends CPInstruction implements LineageTraceable {

	public enum VariableOperationCode {
		CreateVariable,
		AssignVariable,
		CopyVariable,
		MoveVariable,
		RemoveVariable,
		RemoveVariableAndFile,
		CastAsScalarVariable,
		CastAsMatrixVariable,
		CastAsFrameVariable,
		CastAsListVariable,
		CastAsDoubleVariable,
		CastAsIntegerVariable,
		CastAsBooleanVariable,
		Write,
		Read,
		SetFileName;

		public boolean isCast() {
			switch(this) {
				case CastAsScalarVariable:
				case CastAsMatrixVariable:
				case CastAsFrameVariable:
				case CastAsListVariable:
				case CastAsDoubleVariable:
				case CastAsIntegerVariable:
				case CastAsBooleanVariable:
					return true;
				default:
					return false;
			}
		}
	}

	private static final IDSequence _uniqueVarID = new IDSequence(true);
	private static final int CREATEVAR_FILE_NAME_VAR_POS=3;

	private final VariableOperationCode opcode;
	private final List<CPOperand> inputs;
	private final CPOperand output;
	private final MetaData metadata;
	private final UpdateType _updateType;
	private final boolean _containsPreadPrefix;

	// Frame related members
	private final String _schema;

	// parallelization degree for non IO related operations
	private final int k;

	// CSV and LIBSVM related members (used only in createvar instructions)
	private final FileFormatProperties _formatProperties;

	private VariableCPInstruction(VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
			MetaData meta, FileFormatProperties fprops, String schema, UpdateType utype, String sopcode, String istr, int k) {
		super(CPType.Variable, sopcode, istr);
		opcode = op;
		inputs = new ArrayList<>();
		addInput(in1);
		addInput(in2);
		addInput(in3);
		output = out;
		metadata = meta;
		_formatProperties = fprops;
		_schema = schema;
		_updateType = utype;
		_containsPreadPrefix = in1 != null && in1.getName()
			.contains(org.apache.sysds.lops.Data.PREAD_PREFIX);
		this.k = k;
	}


	private VariableCPInstruction(VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
			MetaData meta, FileFormatProperties fprops, String schema, UpdateType utype, String sopcode, String istr) {
		this(op ,in1,in2,in3,out,meta, fprops, schema, utype, sopcode, istr, 1);
	}

	private VariableCPInstruction(VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		String sopcode, String istr) {
		this(op, in1, in2, in3, out, null, null, null, null, sopcode, istr, 1);
	}

	private VariableCPInstruction(VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		String sopcode, String istr, int k) {
		this(op, in1, in2, in3, out, null, null, null, null, sopcode, istr, k);
	}

	// This version of the constructor is used only in case of CreateVariable
	private VariableCPInstruction(VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, MetaData md,
			UpdateType updateType, String schema, String sopcode, String istr) {
		this(op, in1, in2, in3, null, md, null, schema, updateType, sopcode, istr);
	}

	// This version of the constructor is used only in case of CreateVariable
	private VariableCPInstruction(VariableOperationCode op, CPOperand in1, CPOperand in2, CPOperand in3, MetaData md,
			UpdateType updateType, FileFormatProperties formatProperties, String schema, String sopcode,
			String istr) {
		this(op, in1, in2, in3, null, md, formatProperties, schema, updateType, sopcode, istr);
	}

	private static VariableOperationCode getVariableOperationCode ( String str ) {
		if ( str.equalsIgnoreCase(Opcodes.CREATEVAR.toString()))
			return VariableOperationCode.CreateVariable;
		else if ( str.equalsIgnoreCase(Opcodes.ASSIGNVAR.toString()))
			return VariableOperationCode.AssignVariable;
		else if ( str.equalsIgnoreCase(Opcodes.CPVAR.toString()))
			return VariableOperationCode.CopyVariable;
		else if ( str.equalsIgnoreCase(Opcodes.MVVAR.toString()))
			return VariableOperationCode.MoveVariable;
		else if ( str.equalsIgnoreCase(Opcodes.RMVAR.toString()) )
			return VariableOperationCode.RemoveVariable;
		else if ( str.equalsIgnoreCase(Opcodes.RMFILEVAR.toString()) )
			return VariableOperationCode.RemoveVariableAndFile;
		else if ( str.equalsIgnoreCase(Opcodes.CAST_AS_SCALAR.toString()) )
			return VariableOperationCode.CastAsScalarVariable;
		else if ( str.equalsIgnoreCase(Opcodes.CAST_AS_MATRIX.toString()) )
			return VariableOperationCode.CastAsMatrixVariable;
		else if ( str.equalsIgnoreCase(Opcodes.CAST_AS_FRAME.toString())
			|| str.equalsIgnoreCase(Opcodes.CAST_AS_FRAME_VAR.toString()))
			return VariableOperationCode.CastAsFrameVariable;
		else if ( str.equalsIgnoreCase(Opcodes.CAST_AS_LIST.toString()) )
			return VariableOperationCode.CastAsListVariable;
		else if ( str.equalsIgnoreCase(Opcodes.CAST_AS_DOUBLE.toString()) )
			return VariableOperationCode.CastAsDoubleVariable;
		else if ( str.equalsIgnoreCase(Opcodes.CAST_AS_INT.toString()) )
			return VariableOperationCode.CastAsIntegerVariable;
		else if ( str.equalsIgnoreCase(Opcodes.CAST_AS_BOOLEAN.toString()) )
			return VariableOperationCode.CastAsBooleanVariable;
		else if ( str.equalsIgnoreCase(Opcodes.WRITE.toString()) )
			return VariableOperationCode.Write;
		else if ( str.equalsIgnoreCase(Opcodes.READ.toString()) )
			return VariableOperationCode.Read;
		else if ( str.equalsIgnoreCase("setfilename") )
			return VariableOperationCode.SetFileName;
		else
			throw new DMLRuntimeException("Invalid function: " + str);
	}

	/**
	 * Checks if this instruction is a remove instruction for varName
	 *
	 * @param varName variable name
	 * @return true if rmvar instruction including varName
	 */
	public boolean isRemoveVariable(String varName) {
		if( isRemoveVariable() ) {
			for( CPOperand input : inputs )
				if(input.getName().equalsIgnoreCase(varName))
					return true;
		}
		return false;
	}

	public boolean isRemoveVariableNoFile() {
		return opcode == VariableOperationCode.RemoveVariable;
	}

	public boolean isRemoveVariable() {
		return opcode == VariableOperationCode.RemoveVariable
			|| opcode == VariableOperationCode.RemoveVariableAndFile;
	}
	
	public boolean isMoveVariable() {
		return opcode == VariableOperationCode.MoveVariable;
	}

	public boolean isAssignVariable() {
		return opcode == VariableOperationCode.AssignVariable;
	}

	public boolean isAssignOrCopyVariable() {
		return opcode == VariableOperationCode.AssignVariable
			|| opcode == VariableOperationCode.CopyVariable;
	}

	public boolean isCreateVariable() {
		return opcode == VariableOperationCode.CreateVariable;
	}

	public VariableOperationCode getVariableOpcode() {
		return opcode;
	}

	public FileFormatProperties getFormatProperties() {
		return _formatProperties;
	}

	public List<CPOperand> getInputs() {
		return inputs;
	}

	public CPOperand getInput1() {
		return getInput(0);
	}

	public CPOperand getInput2() {
		return getInput(1);
	}

	public CPOperand getInput3() {
		return getInput(2);
	}

	public CPOperand getInput4() {
		return getInput(3);
	}

	public CPOperand getInput(int index) {
		if( inputs.size() <= index )
			return null;
		return inputs.get(index);
	}

	public void addInput(CPOperand input) {
		if( input != null )
			inputs.add(input);
	}

	public String getOutputVariableName(){
		String ret = null;
		if( output != null )
			ret = output.getName();
		return ret;
	}

	public CPOperand getOutput(){
		return output;
	}

	private static int getArity(VariableOperationCode op) {
		if(op.isCast())
			return 3;
		switch(op) {
			case Write:
			case SetFileName:
				return 3;
			default:
				return 2;
		}
	}

	public static VariableCPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		String opcode = parts[0];
		VariableOperationCode voc = getVariableOperationCode(opcode);
	
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
			// All write instructions have 3 parameters, except in case of delimited/csv/libsvm file.
			// Write instructions for csv files also include three additional parameters (hasHeader, delimiter, sparse)
			// Write instructions for libsvm files also include one additional parameters (sparse)
			// TODO - replace hardcoded numbers with more sophisticated code
			if ( parts.length != 6 && parts.length != 7 && parts.length != 9 )
				throw new DMLRuntimeException("Invalid number of operands in write instruction: " + str);
		}
		else if(voc == VariableOperationCode.CastAsFrameVariable){
			InstructionUtils.checkNumFields(parts, 3, 4, 5);
		}
		else {
			try{
				if( voc != VariableOperationCode.RemoveVariable )
					InstructionUtils.checkNumFields ( parts, getArity(voc) ); // no output
			}
			catch(Exception e){
				throw new DMLRuntimeException("Invalid number of fields with operation code: " + voc, e);
			}
		}

		CPOperand in1=null, in2=null, in3=null, in4=null, out=null;
		int k = 1;

		switch (voc) {

		case CreateVariable:
			// variable name
			DataType dt = DataType.valueOf(parts[4]);
			//TODO choose correct value type for tensor
			ValueType vt = dt==DataType.MATRIX ? ValueType.FP64 : ValueType.STRING;
			int extSchema = (dt==DataType.FRAME && parts.length>=12) ? 1 : 0;
			in1 = new CPOperand(parts[1], vt, dt);
			// file name
			in2 = new CPOperand(parts[2], ValueType.STRING, DataType.SCALAR);
			// file name override flag (always literal)
			in3 = new CPOperand(parts[3], ValueType.BOOLEAN, DataType.SCALAR);

			// format
			String fmt = parts[5];
			if ( fmt.equalsIgnoreCase("csv") ) {
				// Cretevar instructions for CSV format either has 13 or 14 inputs.
				// 13 inputs: createvar corresponding to WRITE -- includes properties hasHeader, delim, and sparse
				// 14 inputs: createvar corresponding to READ -- includes properties hasHeader, delim, fill, and fillValue
				if ( parts.length < 14+extSchema || parts.length > 16+extSchema )
					throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
			}
			else if(fmt.equalsIgnoreCase("libsvm")) {
				// 13 inputs: createvar corresponding to WRITE -- includes properties delim, index delim, and sparse
				// 12 inputs: createvar corresponding to READ -- includes properties delim, index delim, and sparse

				if(parts.length < 12 + extSchema)
					throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
			}
			else if(fmt.equalsIgnoreCase("hdf5")) {
				// 11 inputs: createvar corresponding to WRITE/READ -- includes properties dataset name
				if(parts.length < 11 + extSchema)
					throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
			}
			else {
				if ( parts.length != 6 && parts.length != 11+extSchema )
					throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
			}

			MetaDataFormat iimd = null;
			if (dt == DataType.MATRIX || dt == DataType.FRAME || dt == DataType.LIST) {
				DataCharacteristics mc = new MatrixCharacteristics();
				if (parts.length == 6) {
					// do nothing
				}
				else if (parts.length >= 10) {
					// matrix characteristics
					mc.setDimension(Long.parseLong(parts[6]), Long.parseLong(parts[7]));
					mc.setBlocksize(Integer.parseInt(parts[8]));
					mc.setNonZeros(Long.parseLong(parts[9]));
				}
				else {
					throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
				}
				iimd = new MetaDataFormat(mc, FileFormat.safeValueOf(fmt));
			}
			else if (dt == DataType.TENSOR) {
				TensorCharacteristics tc = new TensorCharacteristics(new long[]{1, 1}, 0);
				if (parts.length == 6) {
					// do nothing
				}
				else if (parts.length >= 10) {
					// TODO correct sizes
					tc.setDim(0, Long.parseLong(parts[6]));
					tc.setDim(1, Long.parseLong(parts[7]));
					tc.setBlocksize(Integer.parseInt(parts[8]));
				}
				else {
					throw new DMLRuntimeException("Invalid number of operands in createvar instruction: " + str);
				}
				iimd = new MetaDataFormat(tc, FileFormat.safeValueOf(fmt));
			}
			UpdateType updateType = UpdateType.COPY;
			if ( parts.length >= 11 )
				updateType = UpdateType.valueOf(parts[10].toUpperCase());

			//handle frame schema
			String schema = (dt==DataType.FRAME && parts.length>=12) ? parts[parts.length-1] : null;

			if ( fmt.equalsIgnoreCase("csv") ) {
				// Cretevar instructions for CSV format either has 13 or 14 inputs.
				// 13 inputs: createvar corresponding to WRITE -- includes properties hasHeader, delim, and sparse
				// 14 inputs: createvar corresponding to READ -- includes properties hasHeader, delim, fill, and fillValue
				FileFormatProperties fmtProperties = null;
				int curPos = 11;
				if ( parts.length == 14+extSchema ) {
					boolean hasHeader = Boolean.parseBoolean(parts[curPos]);
					String delim = parts[curPos+1];
					boolean sparse = Boolean.parseBoolean(parts[curPos+2]);
					fmtProperties = new FileFormatPropertiesCSV(hasHeader, delim, sparse) ;
				}
				else {
					boolean hasHeader = Boolean.parseBoolean(parts[curPos]);
					String delim = parts[curPos+1];
					boolean fill = Boolean.parseBoolean(parts[curPos+2]);
					double fillValue = Double.parseDouble(parts[curPos+3]);
					String naStrings = null;
					if ( parts.length == 16+extSchema )
						naStrings = parts[curPos+4];
					fmtProperties = new FileFormatPropertiesCSV(hasHeader, delim, fill, fillValue, naStrings) ;
				}
				return new VariableCPInstruction(VariableOperationCode.CreateVariable,
					in1, in2, in3, iimd, updateType, fmtProperties, schema, opcode, str);
			}
			else if(fmt.equalsIgnoreCase("libsvm")) {
				// Cretevar instructions for LIBSVM format has 13.
				// 13 inputs: createvar corresponding to WRITE -- includes properties delim, index delim and sparse
				// 12 inputs: createvar corresponding to READ -- includes properties delim, index delim, and sparse
				FileFormatProperties fmtProperties = null;
				int curPos = 11;
				if(parts.length == 12 + extSchema) {
					String delim = parts[curPos];
					String indexDelim = parts[curPos + 1];
					fmtProperties = new FileFormatPropertiesLIBSVM(delim, indexDelim);
				}
				else {
					String delim = parts[curPos];
					String indexDelim = parts[curPos + 1];
					boolean sparse = Boolean.parseBoolean(parts[curPos + 2]);
					fmtProperties = new FileFormatPropertiesLIBSVM(delim, indexDelim, sparse);
				}
	
				return new VariableCPInstruction(VariableOperationCode.CreateVariable,
					in1, in2, in3, iimd, updateType, fmtProperties, schema, opcode, str);
			}
			else if(fmt.equalsIgnoreCase("hdf5")) {
				// Cretevar instructions for HDF5 format has 13.
				// 11 inputs: createvar corresponding to WRITE/READ -- includes properties dataset name
				int curPos = 11;
				String datasetName = parts[curPos];
				FileFormatProperties fmtProperties = new FileFormatPropertiesHDF5(datasetName);

				return new VariableCPInstruction(VariableOperationCode.CreateVariable,
					in1, in2, in3, iimd, updateType, fmtProperties, schema, opcode, str);
			}
			else {
				return new VariableCPInstruction(VariableOperationCode.CreateVariable, in1, in2, in3, iimd, updateType, schema, opcode, str);
			}
			
		case AssignVariable:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			break;

		case CopyVariable:
			// Value types are not given here
			boolean withTypes = parts[1].split(VALUETYPE_PREFIX).length > 2 && parts[2].split(VALUETYPE_PREFIX).length > 2;
			in1 = withTypes ? new CPOperand(parts[1]) : new CPOperand(parts[1], ValueType.UNKNOWN, DataType.UNKNOWN);
			in2 = withTypes ? new CPOperand(parts[2]) : new CPOperand(parts[2], ValueType.UNKNOWN, DataType.UNKNOWN);
			break;

		case MoveVariable:
			in1 = new CPOperand(parts[1], ValueType.UNKNOWN, DataType.UNKNOWN);
			in2 = new CPOperand(parts[2], ValueType.UNKNOWN, DataType.UNKNOWN);
			if(parts.length > 3)
				in3 = new CPOperand(parts[3], ValueType.UNKNOWN, DataType.UNKNOWN);
			break;

		case RemoveVariable:
			VariableCPInstruction rminst = new VariableCPInstruction(
				getVariableOperationCode(opcode), null, null, null, out, opcode, str);
			for( int i=1; i<parts.length; i++ )
				rminst.addInput(new CPOperand(parts[i], ValueType.UNKNOWN, DataType.SCALAR));
			return rminst;

		case RemoveVariableAndFile:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			// second argument must be a boolean
			if ( in2.getValueType() != ValueType.BOOLEAN)
				throw new DMLRuntimeException("Unexpected value type for second argument in: " + str);
			break;

		case CastAsFrameVariable:
			if(parts.length==5){
				in1 = new CPOperand(parts[1]); // input to cast
				in2 = new CPOperand(parts[2]); // list of column names
				out = new CPOperand(parts[3]); // output
				k = Integer.parseInt(parts[4]);
				break;
			}
		case CastAsScalarVariable:
		case CastAsMatrixVariable:
		case CastAsListVariable:
		case CastAsDoubleVariable:
		case CastAsIntegerVariable:
		case CastAsBooleanVariable:
			in1 = new CPOperand(parts[1]); // first operand is a variable name => string value type
			out = new CPOperand(parts[2]); // output variable name
			k = Integer.parseInt(parts[3]); // thread count
			break;

		case Write:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			in3 = new CPOperand(parts[3]);

			FileFormatProperties fprops = null;
			if ( in3.getName().equalsIgnoreCase("csv") ) {
				boolean hasHeader = Boolean.parseBoolean(parts[4]);
				String delim = parts[5];
				boolean sparse = Boolean.parseBoolean(parts[6]);
				fprops = new FileFormatPropertiesCSV(hasHeader, delim, sparse);
				in4 = new CPOperand(parts[7]); // description
			}
			else if ( in3.getName().equalsIgnoreCase("libsvm") ) {
				String delim = parts[4];
				String indexDelim = parts[5];
				boolean sparse = Boolean.parseBoolean(parts[6]);
				fprops = new FileFormatPropertiesLIBSVM(delim, indexDelim, sparse);
			}
			else if(in3.getName().equalsIgnoreCase("hdf5") ){
				String datasetName = parts[4];
				fprops = new FileFormatPropertiesHDF5(datasetName);
			}
			else {
				fprops = new FileFormatProperties();
				in4 = new CPOperand(parts[5]); // blocksize in empty description
			}
			VariableCPInstruction inst = new VariableCPInstruction(
				getVariableOperationCode(opcode), in1, in2, in3, out, null, fprops, null, null, opcode, str);
			inst.addInput(in4);

			return inst;

		case Read:
			in1 = new CPOperand(parts[1]);
			in2 = new CPOperand(parts[2]);
			break;

		case SetFileName:
			in1 = new CPOperand(parts[1]); // variable name
			in2 = new CPOperand(parts[2], ValueType.UNKNOWN, DataType.UNKNOWN); // file name
			in3 = new CPOperand(parts[3], ValueType.UNKNOWN, DataType.UNKNOWN); // option: remote or local
			break;

		}
		return new VariableCPInstruction(getVariableOperationCode(opcode), in1, in2, in3, out, opcode, str, k);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		switch ( opcode )
		{
		case CreateVariable:
			processCreateVariableInstruction(ec);
			break;

		case AssignVariable:
			// assign value of variable to the other
			ec.setScalarOutput(getInput2().getName(), ec.getScalarInput(getInput1()));
			break;

		case CopyVariable:
			processCopyInstruction(ec);
			break;

		case MoveVariable:
			processMoveInstruction(ec);
			break;

		case RemoveVariable:
			for( CPOperand input : inputs )
				processRmvarInstruction(ec, input.getName());
			break;

		case RemoveVariableAndFile:
			 processRemoveVariableAndFileInstruction(ec);
			break;

		case CastAsScalarVariable: //castAsScalarVariable
			processCastAsScalarVariableInstruction(ec);
			break;

		case CastAsMatrixVariable:
			processCastAsMatrixVariableInstruction(ec);
			break;

		case CastAsFrameVariable:
			processCastAsFrameVariableInstruction(ec);
			break;

		case CastAsListVariable:
			ListObject lobj = ec.getListObject(getInput1());
			if( lobj.getLength() != 1 || !(lobj.getData(0) instanceof ListObject) )
				ec.setVariable(output.getName(), lobj);
//				throw new RuntimeException("as.list() expects a list input with one nested list: "
//					+ "length(list)="+lobj.getLength()+", dt(list[0])="+lobj.getData(0).getDataType() );
			else ec.setVariable(output.getName(), lobj.getData(0));
			break;

		case CastAsDoubleVariable:
			ScalarObject scalarDoubleInput = ec.getScalarInput(getInput1());
			ec.setScalarOutput(output.getName(), ScalarObjectFactory.castToDouble(scalarDoubleInput));
			break;

		case CastAsIntegerVariable:
			ScalarObject scalarLongInput = ec.getScalarInput(getInput1());
			ec.setScalarOutput(output.getName(), ScalarObjectFactory.castToLong(scalarLongInput));
			break;

		case CastAsBooleanVariable:
			ScalarObject scalarBooleanInput = ec.getScalarInput(getInput1());
			ec.setScalarOutput(output.getName(), new BooleanObject(scalarBooleanInput.getBooleanValue()));
			break;

		case Read:
			processReadInstruction(ec);
			break;

		case Write:
			processWriteInstruction(ec);
			break;

		case SetFileName:
			processSetFileNameInstruction(ec);
			break;

		default:
			throw new DMLRuntimeException("Unknown opcode: " + opcode );
		}
	}

	/**
	 * Handler for processInstruction "CreateVariable" case
	 *
	 * @param ec execution context of the instruction
	 */
	private void processCreateVariableInstruction(ExecutionContext ec){
		//PRE: for robustness we cleanup existing variables, because a setVariable
		//would  cause a buffer pool memory leak as these objects would never be removed
		if(ec.containsVariable(getInput1()))
			processRmvarInstruction(ec, getInput1().getName());

		switch(getInput1().getDataType()) {
			case MATRIX: {
				String fname = createUniqueFilename();
				MatrixObject obj = new MatrixObject(getInput1().getValueType(), fname);
				setCacheableDataFields(obj, getInput1().getName());
				obj.setUpdateType(_updateType);
				obj.setMarkForLinCache(true);
				ec.setVariable(getInput1().getName(), obj);
				if(DMLScript.STATISTICS && _updateType.isInPlace())
					Statistics.incrementTotalUIPVar();
				break;
			}
			case TENSOR: {
				String fname = createUniqueFilename();
				TensorObject obj = new TensorObject(getInput1().getValueType(), fname);
				setCacheableDataFields(obj, getInput1().getName());
				ec.setVariable(getInput1().getName(), obj);
				break;
			}
			case FRAME: {
				String fname = createUniqueFilename();
				FrameObject fobj = new FrameObject(fname);
				setCacheableDataFields(fobj, getInput1().getName());
				if( _schema != null )
					fobj.setSchema(_schema); //after metadata
				ec.setVariable(getInput1().getName(), fobj);
				break;
			}
			case LIST: {
				ListObject lo = ListReader.readListFromHDFS(getInput2().getName(),
					((MetaDataFormat)metadata).getFileFormat().name(), _formatProperties);
				ec.setVariable(getInput1().getName(), lo);
				break;
			}
			case SCALAR: {
				//created variable not called for scalars
				ec.setScalarOutput(getInput1().getName(), null);
				break;
			}
			default:
				throw new DMLRuntimeException("Unexpected data type: " + getInput1().getDataType());
		}
	}

	private String createUniqueFilename(){
		//create new variable for symbol table and cache
		//(existing objects gets cleared through rmvar instructions)
		String fname = getInput2().getName();
		// check if unique filename needs to be generated
		if( Boolean.parseBoolean(getInput3().getName()) ) {
			fname = getUniqueFileName(fname);
		}
		return fname;
	}

	private void setCacheableDataFields(CacheableData<?> obj, String varname){
		//clone metadata because it is updated on copy-on-write, otherwise there
		//is potential for hidden side effects between variables.
		obj.setMetaData((MetaData)metadata.clone());
		obj.enableCleanup(!getInput1().getName()
			.startsWith(org.apache.sysds.lops.Data.PREAD_PREFIX));
		obj.setFileFormatProperties(_formatProperties);
		obj.setPersistentRead(varname.startsWith(org.apache.sysds.lops.Data.PREAD_PREFIX));
	}

	/**
	 * Handler for mvvar instructions.
	 * Example: mvvar &lt;srcvar&gt; &lt;destFile&gt; &lt;format&gt;
	 * Move the file pointed by srcvar to destFile.
	 * Currently, applicable only when format=binaryblock.
	 *
	 * @param ec execution context
	 */
	@SuppressWarnings("rawtypes")
	private void processMoveInstruction(ExecutionContext ec) {

		if ( getInput3() == null ) {
			// example: mvvar tempA A (note that mvvar does not carry the data types)

			// get and remove source variable
			Data srcData = ec.removeVariable(getInput1().getName());

			if ( srcData == null ) {
				throw new DMLRuntimeException("Unexpected error: could not find a data object "
					+ "for variable name: " + getInput1().getName() + ", while processing instruction ");
			}

			// remove existing variable bound to target name and
			// cleanup matrix/frame/list data if necessary
			if( srcData.getDataType().isMatrix() || srcData.getDataType().isFrame() ) {
				Data tgtData = ec.removeVariable(getInput2().getName());
				if( tgtData != null && srcData != tgtData )
					ec.cleanupDataObject(tgtData);
			}

			// do the actual move
			ec.setVariable(getInput2().getName(), srcData);
		}
		else {
			// example instruction: mvvar <srcVar> <destFile> <format>
			if ( ec.getVariable(getInput1().getName()) == null )
				throw new DMLRuntimeException("Unexpected error: could not find a data object for variable name:" + getInput1().getName() + ", while processing instruction " +this.toString());

			Data object = ec.getVariable(getInput1().getName());

			if ( getInput3().getName().equalsIgnoreCase("binaryblock") ) {
				boolean success = false;
				success = ((CacheableData)object).moveData(getInput2().getName(), getInput3().getName());
				if (!success) {
					throw new DMLRuntimeException("Failed to move var " + getInput1().getName() + " to file " + getInput2().getName() + ".");
				}
			}
			else
				if(object instanceof MatrixObject)
					throw new DMLRuntimeException("Unexpected formats while copying: from matrix blocks ["
							+ ((MatrixObject)object).getBlocksize() + "] to " + getInput3().getName());
				else if (object instanceof FrameObject)
					throw new DMLRuntimeException("Unexpected formats while copying: from fram object ["
							+ ((FrameObject)object).getNumColumns() + "," + ((FrameObject)object).getNumColumns() + "] to " + getInput3().getName());
		}
	}

	/**
	 * Handler for RemoveVariableAndFile instruction
	 *
	 * @param ec execution context
	 */
	private void processRemoveVariableAndFileInstruction(ExecutionContext ec){
		// Remove the variable from HashMap _variables, and possibly delete the data on disk.
		boolean del = ( (BooleanObject) ec.getScalarInput(getInput2().getName(), getInput2().getValueType(), true) ).getBooleanValue();
		MatrixObject m = (MatrixObject) ec.removeVariable(getInput1().getName());

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
			m.clearData(ec.getTID());
		}
	}

	/**
	 * Process CastAsScalarVariable instruction.
   *
	 * @param ec execution context
	 */
	private void processCastAsScalarVariableInstruction(ExecutionContext ec){

		switch( getInput1().getDataType() ) {
			case MATRIX: {
				MatrixBlock mBlock = ec.getMatrixInput(getInput1().getName());
				if( mBlock.getNumRows()!=1 || mBlock.getNumColumns()!=1 )
					throw new DMLRuntimeException("Dimension mismatch - unable to cast matrix '"+getInput1().getName()+"' of dimension ("+mBlock.getNumRows()+" x "+mBlock.getNumColumns()+") to scalar. ");
				double value = mBlock.get(0, 0);
				ec.releaseMatrixInput(getInput1().getName());
				ec.setScalarOutput(output.getName(), new DoubleObject(value));
				break;
			}
			case FRAME: {
				FrameBlock fBlock = ec.getFrameInput(getInput1().getName());
				if( fBlock.getNumRows()!=1 || fBlock.getNumColumns()!=1 )
					throw new DMLRuntimeException("Dimension mismatch - unable to cast frame '"+getInput1().getName()+"' of dimension ("+fBlock.getNumRows()+" x "+fBlock.getNumColumns()+") to scalar.");
				Object value = fBlock.get(0,0);
				ec.releaseFrameInput(getInput1().getName());
				ec.setScalarOutput(output.getName(),
					ScalarObjectFactory.createScalarObject(fBlock.getSchema()[0], value));
				break;
			}
			case TENSOR: {
				TensorBlock tBlock = ec.getTensorInput(getInput1().getName());
				if (tBlock.getNumDims() != 2 || tBlock.getNumRows() != 1 || tBlock.getNumColumns() != 1)
					throw new DMLRuntimeException("Dimension mismatch - unable to cast tensor '" + getInput1().getName() + "' to scalar.");
				ValueType vt = !tBlock.isBasic() ? tBlock.getSchema()[0] : tBlock.getValueType();
				ec.setScalarOutput(output.getName(), ScalarObjectFactory
					.createScalarObject(vt, tBlock.get(new int[] {0, 0})));
				ec.releaseTensorInput(getInput1().getName());
				break;
			}
			case LIST: {
				//TODO handling of cleanup status, potentially new object
				ListObject list = (ListObject)ec.getVariable(getInput1().getName());
				ec.setVariable(output.getName(), list.slice(0));
				break;
			}
			case SCALAR: {
				//for robustness in case rewrites added unnecessary as.scalars
				ec.setScalarOutput(output.getName(), ec.getScalarInput(getInput1()));
				break;
			}
			default:
				throw new DMLRuntimeException("Unsupported data type "
					+ "in as.scalar(): "+getInput1().getDataType().name());
		}
	}

	/**
	 * Handler for CastAsMatrixVariable instruction
	 *
	 * @param ec execution context
	 */
	private void processCastAsMatrixVariableInstruction(ExecutionContext ec) {
		switch( getInput1().getDataType() ) {
			case FRAME: {
				FrameBlock fin = ec.getFrameInput(getInput1().getName());
				MatrixBlock out = DataConverter.convertToMatrixBlock(fin);
				ec.releaseFrameInput(getInput1().getName());
				ec.setMatrixOutput(output.getName(), out);
				break;
			}
			case SCALAR: {
				ScalarObject scalarInput = ec.getScalarInput(getInput1());
				MatrixBlock out = new MatrixBlock(scalarInput.getDoubleValue());
				ec.setMatrixOutput(output.getName(), out);
				break;
			}
			case LIST: {
				//TODO handling of cleanup status, potentially new object
				ListObject list = (ListObject)ec.getVariable(getInput1().getName());
				if( list.getLength() > 1 ) {
					if( !list.checkAllDataTypes(DataType.SCALAR) )
						throw new DMLRuntimeException("as.matrix over multi-entry list only allows scalars.");
					MatrixBlock out = new MatrixBlock(list.getLength(), 1, false);
					for( int i=0; i<list.getLength(); i++ )
						out.set(i, 0, ((ScalarObject)list.slice(i)).getDoubleValue());
					ec.setMatrixOutput(output.getName(), out);
				}
				else {
					//pass through matrix input or create 1x1 matrix for scalar
					Data tmp = list.slice(0);
					if( tmp instanceof ScalarObject && tmp.getValueType()!=ValueType.STRING ) {
						MatrixBlock out = new MatrixBlock(((ScalarObject)tmp).getDoubleValue());
						ec.setMatrixOutput(output.getName(), out);
					}
					else {
						ec.setVariable(output.getName(), tmp);
					}
				}
				break;
			}
			default:
				throw new DMLRuntimeException("Unsupported data type "
					+ "in as.matrix(): "+getInput1().getDataType().name());
		}
	}

	/**
	 * Handler for CastAsFrameVariable instruction
	 *
	 * @param ec execution context
	 */
	private void processCastAsFrameVariableInstruction(ExecutionContext ec){
		FrameBlock out;
		if( getInput1().getDataType()==DataType.SCALAR ) {
			ScalarObject scalarInput = ec.getScalarInput(getInput1());
			out = new FrameBlock(1, getInput1().getValueType());
			out.ensureAllocatedColumns(1);
			out.set(0, 0, scalarInput.getStringValue());
			setColumnNames(ec, out);
			ec.setFrameOutput(output.getName(), out);
		}
		else if(getInput1().getDataType()==DataType.MATRIX) { //DataType.FRAME
			MatrixBlock min = ec.getMatrixInput(getInput1().getName());
			out = DataConverter.convertToFrameBlock(min, k);
			ec.releaseMatrixInput(getInput1().getName());
			setColumnNames(ec, out);
			ec.setFrameOutput(output.getName(), out);
		}
		else { //convert list
			ListObject list = (ListObject)ec.getVariable(getInput1().getName());
			Data tmp = list.slice(0);
			if(getInput2() != null){
				throw new RuntimeException("List does not support as.frame column names arguments");
			}
			ec.setVariable(output.getName(), tmp);
		}
	}

	private void setColumnNames(ExecutionContext ec, FrameBlock out){
		if(getInput2() != null){
			ListObject colNames = (ListObject)ec.getVariable(getInput2().getName());
			String[] names = new String[out.getNumColumns()];
			List<Data> dat = colNames.getData();
			for(int i = 0; i < out.getNumColumns();i++)
				names[i] = ((StringObject)dat.get(i)).getStringValue();
			out.setColumnNames(names);
		}
	}

	/**
	 * Handler for Read instruction
   *
	 * @param ec execution context
	 */
	private void processReadInstruction(ExecutionContext ec){
		ec.setScalarOutput(getInput1().getName(),
			HDFSTool.readScalarObjectFromHDFSFile(getInput2().getName(), getInput1().getValueType()));
	}

	/**
	 * Handler for cpvar instructions.
	 * Example: cpvar &lt;srcvar&gt; &lt;destvar&gt;
	 *
	 * @param ec execution context
	 */
	private void processCopyInstruction(ExecutionContext ec) {
		
		// get source variable
		Data dd = ec.getVariable(getInput1().getName());

		if ( dd == null )
			throw new DMLRuntimeException("Unexpected error: could not find a data object for variable name:" + getInput1().getName() + ", while processing instruction " +this.toString());

		// remove existing variable bound to target name
		Data input2_data = ec.removeVariable(getInput2().getName());

		//cleanup matrix data on fs/hdfs (if necessary)
		if( input2_data != null )
			ec.cleanupDataObject(input2_data);

		// do the actual copy!
		ec.setVariable(getInput2().getName(), dd);
	}

	/**
	 * Handler for write instructions.
	 *
	 * Non-native formats like MM and CSV are handled through specialized helper functions.
	 * The default behavior is to write out the specified matrix from the instruction, in
	 * the format given by the corresponding symbol table entry.
	 *
	 * @param ec execution context
	 */
	private void processWriteInstruction(ExecutionContext ec) {
		//get filename (literal or variable expression)
		String fname = ec.getScalarInput(getInput2()).getStringValue();
		String fmtStr = ec.getScalarInput(getInput3()).getStringValue();
		FileFormat fmt = FileFormat.safeValueOf(fmtStr);
		if( fmt != FileFormat.LIBSVM  && fmt != FileFormat.HDF5) {
			String desc = ec.getScalarInput(getInput4().getName(), ValueType.STRING, getInput4().isLiteral()).getStringValue();
			_formatProperties.setDescription(desc);
		}

		if( getInput1().getDataType() == DataType.SCALAR ) {
			HDFSTool.writeScalarToHDFS(ec.getScalarInput(getInput1()), fname);
		}
		else if( getInput1().getDataType() == DataType.MATRIX ) {
			MatrixObject mo = ec.getMatrixObject(getInput1().getName());
			int blen = Integer.parseInt(getInput4().getName());
			LocalTaskQueue<IndexedMatrixValue> stream = mo.getStreamHandle();

			if (stream != null) {

				try {
					MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(fmt);
                    long nrows = mo.getNumRows();
                    long ncols = mo.getNumColumns();

					long totalNnz = writer.writeMatrixFromStream(fname, stream, nrows, ncols, blen);
					MatrixCharacteristics mc = new MatrixCharacteristics(nrows, ncols, blen, totalNnz);
                    HDFSTool.writeMetaDataFile(fname + ".mtd", mo.getValueType(), mc, fmt);

				}
				catch(Exception ex) {
					throw new DMLRuntimeException("Failed to write OOC stream to " + fname, ex);
				}
			}
			if( fmt == FileFormat.MM )
				writeMMFile(ec, fname);
			else if( fmt == FileFormat.CSV )
				writeCSVFile(ec, fname);
			else if(fmt == FileFormat.LIBSVM)
				writeLIBSVMFile(ec, fname);
			else if(fmt == FileFormat.HDF5)
				writeHDF5File(ec, fname);
			else {
				// Default behavior (text, binary)
//				MatrixObject mo = ec.getMatrixObject(getInput1().getName());
//				int blen = Integer.parseInt(getInput4().getName());
				mo.exportData(fname, fmtStr, new FileFormatProperties(blen));
			}
		}
		else if( getInput1().getDataType() == DataType.FRAME ) {
			FrameObject mo = ec.getFrameObject(getInput1().getName());
			mo.exportData(fname, fmtStr, _formatProperties);
		}
		else if( getInput1().getDataType() == DataType.TENSOR ) {
			// TODO write tensor
			TensorObject to = ec.getTensorObject(getInput1().getName());
			to.exportData(fname, fmtStr, _formatProperties);
		}
		else if( getInput1().getDataType() == DataType.LIST ) {
			ListObject lo = ec.getListObject(getInput1().getName());
			int blen = Integer.parseInt(getInput4().getName());
			ListWriter.writeListToHDFS(lo, fname, fmtStr, new FileFormatProperties(blen));
		}
	}

	/**
	 * Handler for SetFileName instruction
	 * @param ec execution context
	 */
	private void processSetFileNameInstruction(ExecutionContext ec){
		Data data = ec.getVariable(getInput1().getName());
		if ( data.getDataType() == DataType.MATRIX ) {
			if ( getInput3().getName().equalsIgnoreCase("remote") )
				((MatrixObject)data).setFileName(getInput2().getName());
			else
				throw new DMLRuntimeException(
					"Invalid location (" + getInput3().getName() + ") in SetFileName instruction: " + instString);
		}
		else
			throw new DMLRuntimeException("Invalid data type (" + getInput1().getDataType() + ") in SetFileName instruction: " + instString);
	}

	/**
	 * Remove variable instruction externalized as a static function in order to allow various
	 * cleanup procedures to use the same codepath as the actual rmVar instruction
	 *
	 * @param ec execution context
	 * @param varname variable name
	 */
	public static void processRmvarInstruction( ExecutionContext ec, String varname ) {
		// remove variable from symbol table
		Data dat = ec.removeVariable(varname);
		//cleanup matrix data on fs/hdfs (if necessary)
		if( dat != null )
			ec.cleanupDataObject(dat);
	}

	/**
	 * Helper function to write CSV files to HDFS.
	 *
	 * @param ec execution context
	 * @param fname file name
	 */
	private void writeCSVFile(ExecutionContext ec, String fname) {
		MatrixObject mo = ec.getMatrixObject(getInput1().getName());
		String outFmt = "csv";
		FileFormatProperties fprop = (_formatProperties instanceof FileFormatPropertiesCSV) ?
			_formatProperties : new FileFormatPropertiesCSV(); //for dynamic format strings
		
		if(mo.isDirty()) {
			// there exist data computed in CP that is not backed up on HDFS
			// i.e., it is either in-memory or in evicted space
			mo.exportData(fname, outFmt, fprop);
		}
		else {
			try {
				FileFormat fmt = ((MetaDataFormat)mo.getMetaData()).getFileFormat();
				DataCharacteristics dc = (mo.getMetaData()).getDataCharacteristics();
				if( fmt == FileFormat.CSV && !mo.isPersistentRead() ) {
					WriterTextCSV writer = new WriterTextCSV((FileFormatPropertiesCSV)fprop);
					writer.addHeaderToCSV(mo.getFileName(), fname, dc.getRows(), dc.getCols());
				}
				else {
					mo.exportData(fname, outFmt, fprop);
				}
				HDFSTool.writeMetaDataFile(fname + ".mtd",
					mo.getValueType(), dc, FileFormat.CSV, fprop);
			}
			catch(IOException e) {
				throw new DMLRuntimeException(e);
			}
		}
	}

	/**
	 * Helper function to write LIBSVM files to HDFS.
	 *
	 * @param ec	execution context
	 * @param fname file name
	 */
	private void writeLIBSVMFile(ExecutionContext ec, String fname) {
		MatrixObject mo = ec.getMatrixObject(getInput1().getName());
		String outFmt = "libsvm";

		if(mo.isDirty()) {
			// there exist data computed in CP that is not backed up on HDFS
			// i.e., it is either in-memory or in evicted space
			mo.exportData(fname, outFmt, _formatProperties);
		}
		else {
			try {
				mo.exportData(fname, outFmt, _formatProperties);
				HDFSTool.writeMetaDataFile(fname + ".mtd", mo.getValueType(),
					mo.getMetaData().getDataCharacteristics(), FileFormat.LIBSVM, _formatProperties);
			}
			catch (IOException e) {
				throw new DMLRuntimeException(e);
			}
		}
	}

	/**
	 * Helper function to write HDF5 files to HDFS.
	 *
	 * @param ec    execution context
	 * @param fname file name
	 */
	private void writeHDF5File(ExecutionContext ec, String fname) {
		MatrixObject mo = ec.getMatrixObject(getInput1().getName());
		String outFmt = "hdf5";

		if(mo.isDirty()) {
			// there exist data computed in CP that is not backed up on HDFS
			// i.e., it is either in-memory or in evicted space
			mo.exportData(fname, outFmt, _formatProperties);
		}
		else {
			try {
				FileFormat fmt = ((MetaDataFormat) mo.getMetaData()).getFileFormat();
				DataCharacteristics dc = (mo.getMetaData()).getDataCharacteristics();
				if(fmt == FileFormat.HDF5 && !getInput1().getName().startsWith(org.apache.sysds.lops.Data.PREAD_PREFIX)) {
					//FIXME why is this writer never used?
					@SuppressWarnings("unused")
					WriterHDF5 writer = new WriterHDF5((FileFormatPropertiesHDF5) _formatProperties);
				}
				else {
					mo.exportData(fname, outFmt, _formatProperties);
				}
				HDFSTool.writeMetaDataFile(fname + ".mtd", mo.getValueType(), dc, FileFormat.HDF5, _formatProperties);
			}
			catch (IOException e) {
				throw new DMLRuntimeException(e);
			}
		}
	}

	/**
	 * Helper function to write MM files to HDFS.
	 *
	 * @param ec execution context
	 * @param fname file name
	 */
	private void writeMMFile(ExecutionContext ec, String fname) {
		MatrixObject mo = ec.getMatrixObject(getInput1().getName());
		String outFmt = FileFormat.MM.toString();
		if(mo.isDirty()) {
			// there exist data computed in CP that is not backed up on HDFS
			// i.e., it is either in-memory or in evicted space
			mo.exportData(fname, outFmt);
		}
		else {
			try {
				FileFormat fmt = ((MetaDataFormat)mo.getMetaData()).getFileFormat();
				DataCharacteristics dc = mo.getDataCharacteristics();
				if( fmt == FileFormat.TEXT
					&& !getInput1().getName().startsWith(org.apache.sysds.lops.Data.PREAD_PREFIX) )
				{
					WriterMatrixMarket.mergeTextcellToMatrixMarket(mo.getFileName(),
						fname, dc.getRows(), dc.getCols(), dc.getNonZeros());
				}
				else {
					mo.exportData(fname, outFmt);
				}
			}
			catch (IOException e) {
				throw new DMLRuntimeException(e);
			}
		}
	}

	private static void cleanDataOnHDFS(MatrixObject mo) {
		try {
			String fpath = mo.getFileName();
			if (fpath != null) {
				HDFSTool.deleteFileIfExistOnHDFS(fpath);
				HDFSTool.deleteFileIfExistOnHDFS(fpath + ".mtd");
			}
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}

	public static Instruction prepareRemoveInstruction(long... varName) {
		String[] tmp = new String[varName.length];
		Arrays.setAll(tmp, i -> String.valueOf(varName[i]));
		return prepareRemoveInstruction(tmp);
	}

	public static Instruction prepareRemoveInstruction(String... varNames) {
		StringBuilder sb = InstructionUtils.getStringBuilder();
		sb.append("CP");
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(Opcodes.RMVAR);
		for( String varName : varNames ) {
			sb.append(Lop.OPERAND_DELIMITOR);
			sb.append(varName);
		}
		return parseInstruction(sb.toString());
	}

	public static Instruction prepareCopyInstruction(String srcVar, String destVar) {
		return parseInstruction(
			InstructionUtils.concatOperands("CP", Opcodes.CPVAR.toString(), srcVar, destVar));
	}

	public static Instruction prepMoveInstruction(String srcVar, String destFileName, String format) {
		return parseInstruction(
			InstructionUtils.concatOperands("CP", Opcodes.MVVAR.toString(), srcVar, destFileName, format));
	}

	public static Instruction prepMoveInstruction(String srcVar, String destVar) {
		return parseInstruction(
			InstructionUtils.concatOperands("CP", Opcodes.MVVAR.toString(), srcVar, destVar));
	}

	private static String getBasicCreatevarString(String varName, String fileName, boolean fNameOverride, DataType dt, String format) {
		//note: the filename override property leads to concatenation of unique ids in order to
		//ensure conflicting filenames for objects that originate from the same instruction
		boolean lfNameOverride = fNameOverride && !ConfigurationManager
			.getCompilerConfigFlag(ConfigType.IGNORE_TEMPORARY_FILENAMES);

		// Constant CREATEVAR_FILE_NAME_VAR_POS is used to find a position of filename within a string generated through this function.
		// If this position of filename within this string changes then constant CREATEVAR_FILE_NAME_VAR_POS to be updated.
		return InstructionUtils.concatOperands(
			"CP", Opcodes.CREATEVAR.toString(), varName, fileName, String.valueOf(lfNameOverride), dt.toString(), format);
	}

	public static Instruction prepCreatevarInstruction(String varName, String fileName, boolean fNameOverride, String format) {
		return parseInstruction(getBasicCreatevarString(varName, fileName, fNameOverride, DataType.MATRIX, format));
	}

	public static Instruction prepCreatevarInstruction(String varName, String fileName, boolean fNameOverride, DataType dt, String format, DataCharacteristics mc, UpdateType update) {
		return parseInstruction(InstructionUtils.concatOperands(
			getBasicCreatevarString(varName, fileName, fNameOverride, dt, format),
			String.valueOf(mc.getRows()), String.valueOf(mc.getCols()), String.valueOf(mc.getBlocksize()),
			String.valueOf(mc.getNonZeros()), update.toString().toLowerCase()));
	}

	public static Instruction prepCreatevarInstruction(String varName, String fileName, boolean fNameOverride, DataType dt, String format, DataCharacteristics mc, UpdateType update, boolean hasHeader, String delim, boolean sparse) {
		return parseInstruction(InstructionUtils.concatOperands(
			getBasicCreatevarString(varName, fileName, fNameOverride, dt, format),
			String.valueOf(mc.getRows()), String.valueOf(mc.getCols()), String.valueOf(mc.getBlocksize()),
			String.valueOf(mc.getNonZeros()), update.toString().toLowerCase(),
			String.valueOf(hasHeader), delim, String.valueOf(sparse)));
	}

	@Override
	public void updateInstructionThreadID(String pattern, String replace) {
		if(    opcode == VariableOperationCode.CreateVariable
			|| opcode == VariableOperationCode.SetFileName )
		{
			//replace in-memory instruction
			getInput2().setName(getInput2().getName().replaceAll(pattern, replace));

			// Find a start position of file name string.
			int iPos = StringUtils.ordinalIndexOf(instString, Lop.OPERAND_DELIMITOR, CREATEVAR_FILE_NAME_VAR_POS);
			// Find an end position of file name string.
			int iPos2 = StringUtils.indexOf(instString, Lop.OPERAND_DELIMITOR, iPos+1);

			StringBuilder sb = new StringBuilder();
			sb.append(instString.substring(0,iPos+1)); // It takes first part before file name.
			// This will replace 'pattern' with 'replace' string from file name.
			sb.append(ProgramConverter.saveReplaceFilenameThreadID(instString.substring(iPos+1, iPos2+1), pattern, replace));
			sb.append(instString.substring(iPos2+1)); // It takes last part after file name.

			instString = sb.toString();
		}
	}

	@Override
	public Pair<String,LineageItem> getLineageItem(ExecutionContext ec) {
		String varname = null;
		LineageItem li = null;
		switch (getVariableOpcode()) {
			case CreateVariable:
				if (!_containsPreadPrefix)
					break; //otherwise fall through

			case Read: {
				varname = getInput1().getName();
				li = new LineageItem(toString().replace(getInput1().getName(),
					org.apache.sysds.lops.Data.PREAD_PREFIX+"xxx"), getOpcode());
				break;
			}
			case AssignVariable: {
				varname = getInput2().getName();
				li = new LineageItem(getOpcode(), new LineageItem[]{ec.getLineage().getOrCreate(getInput1())});
				break;
			}
			case CopyVariable: {
				if (!ec.getLineage().contains(getInput1()))
					throw new DMLRuntimeException("Could not find LineageItem for " + getInput1().getName());
				varname = getInput2().getName();
				li = new LineageItem(getOpcode(), new LineageItem[]{ec.getLineage().get(getInput1())});
				break;
			}
			case Write: {
				ArrayList<LineageItem> lineages = new ArrayList<>();
				for (CPOperand input : getInputs())
					if (!input.getName().isEmpty())
						lineages.add(ec.getLineage().getOrCreate(input));
				if (_formatProperties != null && _formatProperties.getDescription() != null && !_formatProperties.getDescription().isEmpty())
					lineages.add(new LineageItem(_formatProperties.getDescription()));
				varname = getInput1().getName();
				li = new LineageItem(getOpcode(), lineages.toArray(new LineageItem[0]));
				break;
			}
			case CastAsBooleanVariable:
			case CastAsDoubleVariable:
			case CastAsIntegerVariable:
			case CastAsScalarVariable:
			case CastAsMatrixVariable:
			case CastAsFrameVariable:{
				varname = getOutputVariableName();
				li = new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, getInput1()));
				break;
			}
			case CastAsListVariable:
				varname = getOutputVariableName();
				ListObject lobj = ec.getListObject(getInput1());
				if (lobj.getLength() != 1 || !(lobj.getData(0) instanceof ListObject))
					li = new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, getInput1()));
				else
					li = new LineageItem(getOpcode(), new LineageItem[] {lobj.getLineageItem(0)});
				break;
			case RemoveVariable:
			case MoveVariable:
			default:
		}

		return (li == null) ? null : Pair.of(varname, li);
	}

	public boolean isVariableCastInstruction() {
		return opcode == VariableOperationCode.CastAsScalarVariable
			|| opcode == VariableOperationCode.CastAsMatrixVariable
			|| opcode == VariableOperationCode.CastAsFrameVariable
			|| opcode == VariableOperationCode.CastAsIntegerVariable
			|| opcode == VariableOperationCode.CastAsDoubleVariable
			|| opcode == VariableOperationCode.CastAsBooleanVariable;
	}

	public static String getUniqueFileName(String fname) {
		return InstructionUtils.concatStrings(fname, "_", String.valueOf(_uniqueVarID.getNextID()));
	}

	public MetaData getMetaData() {
		return metadata;
	}
}

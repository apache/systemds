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

package org.apache.sysds.lops;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.parser.DataExpression;

import java.util.HashMap;

/**
 * Lop to represent data objects. Data objects represent matrices, vectors,
 * variables, literals. Can be for both input and output.
 */
public class DataIOGen extends Lop {
	public static final String PREAD_PREFIX = "pREAD";
	private final String formatType;
	private final OpOpData _op;
	private final boolean literal_var;
	private HashMap<String, Lop> _inputParams;

	/**
	 * Method to create literal LOPs.
	 *
	 * @param vt           value type
	 * @param literalValue literal value
	 * @return literal low-level operator
	 */
	public static DataIOGen createLiteralLop(ValueType vt, String literalValue) {
		// All literals have default format type of TEXT
		return new DataIOGen(OpOpData.PERSISTENTREAD, null, null, null, literalValue, DataType.SCALAR, vt, FileFormat.TEXT.toString());
	}

	/**
	 * Constructor to setup read or write LOP
	 * In case of write: <code>input</code> must be provided. This will always be added as the first element in <code>input</code> array.
	 * For literals: this function is invoked through the static method <code>createLiteralLop</code>.
	 *
	 * @param op                  operation type
	 * @param input               low-level operator
	 * @param inputParametersLops input lops
	 * @param name                string name
	 * @param literal             string literal
	 * @param dt                  data type
	 * @param vt                  value type
	 * @param fmt                 file format
	 */
	public DataIOGen(OpOpData op, Lop input, HashMap<String, Lop> inputParametersLops, String name, String literal, DataType dt, ValueType vt,
		String fmt) {
		super(Type.Data, dt, vt);
		_op = op;
		literal_var = (literal != null);

		// Either <code>name</code> or <code>literal</code> can be non-null.
		if(literal_var) {
			if(_op.isTransient())
				throw new LopsException("Invalid parameter values while setting up a Data LOP -- transient flag is invalid for a literal.");
			getOutputParameters().setLabel(literal);
		}
		else if(name != null) {
			if(_op.isTransient())
				getOutputParameters().setLabel(name); // tvar+name
			else {
				String code = _op == OpOpData.FUNCTIONOUTPUT ? "" : _op.isRead() ? "pREAD" : "pWRITE";
				getOutputParameters().setLabel(code + name);
			}
		}
		else {
			throw new LopsException("Invalid parameter values while setting up a Data LOP -- the lop must have either literal value or a name.");
		}

		// WRITE operation must have an input Lops, we always put this
		// input Lops as the first element of WRITE input. The parameters of
		// WRITE operation are then put as the following input elements.
		if(input != null && op.isWrite()) {
			addInput(input);
			input.addOutput(this);
		}

		_inputParams = inputParametersLops;

		if(_inputParams != null) {
			for(Lop lop : inputParametersLops.values()) {
				addInput(lop);
				lop.addOutput(this);
			}
			if(inputParametersLops.get(DataExpression.IO_FILENAME) != null) {
				OutputParameters outParams = (inputParametersLops.get(DataExpression.IO_FILENAME)).getOutputParameters();
				String fName = outParams.getLabel();
				this.getOutputParameters().setFile_name(fName);
			}
		}

		//set output format
		formatType = fmt;
		//outParams.setFormat(fmt);
		setLopProperties();
	}

	private void setLopProperties() {
		lps.setProperties(inputs, ExecType.INVALID);
	}

	/**
	 * Data-Lop-specific method to set the execution type for persistent write.
	 * TODO: split lops into MR/CP lop.
	 *
	 * @param et execution type
	 */
	public void setExecType(ExecType et) {
		lps.execType = et;
	}

	/**
	 * method to get format type for input, output files.
	 *
	 * @return file format
	 */
	public String getFileFormatType() {
		return formatType;
	}

	@Override public String toString() {
		return getID() + ":" + "File_Name: " + getOutputParameters().getFile_name() + " " + "Label: " + getOutputParameters().getLabel() + " " + "Operation: = " + _op + " " + "Format: " + outParams.getFormat() + " Datatype: " + getDataType() + " Valuetype: " + getValueType() + " num_rows = " + getOutputParameters().getNumRows() + " num_cols = " + getOutputParameters().getNumCols() + " UpdateInPlace: " + getOutputParameters().getUpdateType();
	}

	/**
	 * method to get operation type, i.e. read/write.
	 *
	 * @return operation type
	 */

	public OpOpData getOperationType() {
		return _op;
	}

	/**
	 * method to get inputParams
	 *
	 * @return input parameters
	 */
	public HashMap<String, Lop> getInputParams() {
		return _inputParams;
	}

	public Lop getNamedInputLop(String name) {
		return _inputParams.get(name);
	}

	public Lop getNamedInputLop(String name, String defaultVal) {
		if(_inputParams.containsKey(name))
			return _inputParams.get(name);
		else
			return DataIOGen.createLiteralLop(ValueType.STRING, defaultVal);
	}

	/**
	 * method to check if this data lop represents a literal.
	 *
	 * @return true if data lop is a literal
	 */
	public boolean isLiteral() {
		return literal_var;
	}

	public boolean getBooleanValue() {
		if(literal_var) {
			return Boolean.parseBoolean(getOutputParameters().getLabel());
		}
		else
			throw new LopsException("Cannot obtain the value of a non-literal variable at compile time.");
	}

	public double getDoubleValue() {
		if(literal_var) {
			return Double.parseDouble(getOutputParameters().getLabel());
		}
		else
			throw new LopsException("Cannot obtain the value of a non-literal variable at compile time.");
	}

	public long getLongValue() {
		if(literal_var) {
			ValueType vt = getValueType();
			switch(vt) {
				case INT64:
					return Long.parseLong(getOutputParameters().getLabel());
				case FP64:
					return (long) Double.parseDouble(getOutputParameters().getLabel());

				default:
					throw new LopsException("Encountered a non-numeric value " + (vt) + ", while a numeric value is expected.");
			}
		}
		else
			throw new LopsException("Can not obtain the value of a non-literal variable at compile time.");
	}

	public String getStringValue() {
		if(literal_var) {
			return getOutputParameters().getLabel();
		}
		else
			throw new LopsException("Cannot obtain the value of a non-literal variable at compile time.");
	}

	public boolean isPersistentWrite() {
		return _op == OpOpData.PERSISTENTWRITE;
	}

	public boolean isPersistentRead() {
		return _op == OpOpData.PERSISTENTREAD && !literal_var;
	}

	/**
	 * Method to get CP instructions for reading/writing scalars and matrices from/to HDFS.
	 * This method generates CP read/write instructions.
	 */
	@Override public String getInstructions(String input1, String input2) {
		if(getOutputParameters().getFile_name() == null && _op.isRead())
			throw new LopsException(
				this.printErrorLocation() + "Data.getInstructions(): Exepecting a SCALAR data type, encountered " + getDataType());

		StringBuilder sb = new StringBuilder();
		if(this.getExecType() == ExecType.SPARK)
			sb.append("SPARK");
		else
			sb.append("CP");
		sb.append(OPERAND_DELIMITOR);
		if(_op.isRead()) {
			sb.append("read");
			sb.append(OPERAND_DELIMITOR);
			sb.append(this.prepInputOperand(input1));
		}
		else if(_op.isWrite()) {
			sb.append("write");
			sb.append(OPERAND_DELIMITOR);
			sb.append(getInputs().get(0).prepInputOperand(input1));
		}
		else
			throw new LopsException(this.printErrorLocation() + "In Data Lop, Unknown operation: " + _op);

		sb.append(OPERAND_DELIMITOR);
		Lop fnameLop = _inputParams.get(DataExpression.IO_FILENAME);
		boolean literal = (fnameLop instanceof DataIOGen && ((DataIOGen) fnameLop).isLiteral());
		sb.append(prepOperand(input2, DataType.SCALAR, ValueType.STRING, literal));

		// attach outputInfo in case of matrices
		OutputParameters oparams = getOutputParameters();
		if(_op.isWrite()) {
			sb.append(OPERAND_DELIMITOR);
			String fmt = getFileFormatType();
			sb.append(prepOperand(fmt, DataType.SCALAR, ValueType.STRING, true));
		}

		if(_op.isWrite()) {
			sb.append(OPERAND_DELIMITOR);
			Lop descriptionLop = getInputParams().get(DataExpression.DESCRIPTIONPARAM);
			if(descriptionLop != null) {
				boolean descLiteral = (descriptionLop instanceof DataIOGen && ((DataIOGen) descriptionLop).isLiteral());
				sb.append(prepOperand(descriptionLop.getOutputParameters().getLabel(), DataType.SCALAR, ValueType.STRING, descLiteral));
			}
			else {
				sb.append(prepOperand("", DataType.SCALAR, ValueType.STRING, true));
			}
			sb.append(OPERAND_DELIMITOR);
			sb.append(oparams.getBlocksize());
		}

		return sb.toString();
	}

	/**
	 * Method to generate createvar instruction that updates symbol table with metadata, hdfsfile name, etc.
	 */
	@Override public String getInstructions() {
		return getCreateVarInstructions(getOutputParameters().getFile_name(), getOutputParameters().getLabel());
	}

	@Override public String getInstructions(String outputFileName) {
		return getCreateVarInstructions(outputFileName, getOutputParameters().getLabel());
	}

	public String getCreateVarInstructions(String outputFileName, String outputLabel) {
		if(getDataType() == DataType.MATRIX || getDataType() == DataType.FRAME) {

			if(_op.isTransient())
				throw new LopsException("getInstructions() should not be called for transient nodes.");

			OutputParameters oparams = getOutputParameters();

			StringBuilder sb = new StringBuilder();
			sb.append("CP");
			sb.append(OPERAND_DELIMITOR);
			sb.append("createvar");
			sb.append(OPERAND_DELIMITOR);
			sb.append(outputLabel);
			sb.append(OPERAND_DELIMITOR);
			sb.append(outputFileName);
			sb.append(OPERAND_DELIMITOR);
			sb.append(false);
			sb.append(OPERAND_DELIMITOR);
			sb.append(getDataType());
			sb.append(OPERAND_DELIMITOR); // only persistent reads come here!
			sb.append("IOGEN");
			sb.append(OPERAND_DELIMITOR);
			sb.append(getFileFormatType());
			sb.append(OPERAND_DELIMITOR);
			sb.append(oparams.getNumRows());
			sb.append(OPERAND_DELIMITOR);
			sb.append(oparams.getNumCols());
			sb.append(OPERAND_DELIMITOR);
			sb.append(oparams.getBlocksize());
			sb.append(OPERAND_DELIMITOR);
			sb.append(oparams.getNnz());
			sb.append(OPERAND_DELIMITOR);
			sb.append(oparams.getUpdateType().toString().toLowerCase());
			return sb.toString();
		}
		else {
			throw new LopsException(this.printErrorLocation() + "In Data Lop, Unexpected data type " + getDataType());
		}
	}
}

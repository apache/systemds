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

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.instructions.InstructionUtils;

/**
 * Lop to represent data objects. Data objects represent matrices, vectors, 
 * variables, literals. Can be for both input and output. 
 */
public class Data extends Lop
{
	public static final String PREAD_PREFIX = "pREAD";
	private final FileFormat formatType;
	private final OpOpData _op;
	private final boolean literal_var;
	private HashMap<String, Lop> _inputParams;

	/**
	 * Method to create literal LOPs.
	 * 
	 * @param vt value type
	 * @param literalValue literal value
	 * @return literal low-level operator
	 */
	public static Data createLiteralLop(ValueType vt, String literalValue) {
		// All literals have default format type of TEXT
		return new Data(OpOpData.PERSISTENTREAD, null, null, null, literalValue, DataType.SCALAR, vt, FileFormat.BINARY);
	}

	/**
	 * Constructor to setup read or write LOP
	 * In case of write: <code>input</code> must be provided. This will always be added as the first element in <code>input</code> array.
	 * For literals: this function is invoked through the static method <code>createLiteralLop</code>.
	 * 
	 * @param op operation type
	 * @param input low-level operator
	 * @param inputParametersLops input lops
	 * @param name string name
	 * @param literal string literal
	 * @param dt data type
	 * @param vt value type
	 * @param fmt file format
	 */
	public Data(OpOpData op, Lop input, HashMap<String, Lop>
		inputParametersLops, String name, String literal, DataType dt, ValueType vt, FileFormat fmt)
	{
		super(Lop.Type.Data, dt, vt);
		_op = op;
		literal_var = (literal != null);
		
		// Either <code>name</code> or <code>literal</code> can be non-null.
		if(literal_var){
			if ( _op.isTransient() )
				throw new LopsException("Invalid parameter values while setting up a Data LOP -- transient flag is invalid for a literal.");
			getOutputParameters().setLabel(literal);
		}
		else if(name != null) {
			if ( _op.isTransient() )
				getOutputParameters().setLabel(name); // tvar+name
			else {
				String code = _op == OpOpData.FUNCTIONOUTPUT ? "" :
					_op.isRead() ? "pREAD" : "pWRITE";
				getOutputParameters().setLabel(code+name);
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

		if ( _inputParams != null ) {
			for (Lop lop : inputParametersLops.values()) {
				addInput(lop);
				lop.addOutput(this);
			}
			if (   inputParametersLops.get(DataExpression.IO_FILENAME)!= null
				&& inputParametersLops.get(DataExpression.IO_FILENAME) instanceof Data )
			{
				OutputParameters outParams = ((Data)inputParametersLops.get(DataExpression.IO_FILENAME)).getOutputParameters();
				String fName = outParams.getLabel();
				this.getOutputParameters().setFile_name(fName);
			}
		}

		//set output format
		formatType = fmt;
		outParams.setFormat(fmt);
		setLopProperties();
	}

	private void setLopProperties() {
		lps.setProperties ( inputs, ExecType.INVALID);
	}

	/**
	 * method to get format type for input, output files.
	 * @return file format
	 */
	public FileFormat getFileFormatType() {
		return formatType ;
	}

	@Override
	public String toString() {
		return getID() + ":" + "File_Name: " + getOutputParameters().getFile_name() + " "
			+ "Label: " + getOutputParameters().getLabel() + " " + "Operation: = " + _op + " "
			+ "Format: " + outParams.getFormat() +  " Datatype: " + getDataType() + " Valuetype: " + getValueType()
			+ " num_rows = " + getOutputParameters().getNumRows() + " num_cols = " + getOutputParameters().getNumCols()
			+ " UpdateInPlace: " + getOutputParameters().getUpdateType();
	}

	/**
	 * method to get operation type, i.e. read/write.
	 * @return operation type
	 */

	public OpOpData getOperationType() {
		return _op;
	}

	/**
	 * method to get inputParams
	 * @return input parameters
	 */
	public HashMap<String, Lop> getInputParams(){
		return _inputParams;
	}

	public Lop getNamedInputLop(String name) {
		return _inputParams.get(name);
	}

	public Lop getNamedInputLop(String name, String defaultVal) {
		if( _inputParams.containsKey(name) )
			return _inputParams.get(name);
		else
			return Data.createLiteralLop(ValueType.STRING, defaultVal);
	}

	/**
	 * method to check if this data lop represents a literal.
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
		return _op == OpOpData.PERSISTENTREAD
			&& !literal_var;
	}

	public boolean isTransientWrite() {
		return _op == OpOpData.TRANSIENTWRITE;
	}

	public boolean isTransientRead() {
		return _op == OpOpData.TRANSIENTREAD;
	}

	/**
	 * Method to get CP instructions for reading/writing scalars and matrices from/to HDFS.
	 * This method generates CP read/write instructions.
	 */
	@Override
	public String getInstructions(String input1, String input2)
	{
		if ( getOutputParameters().getFile_name() == null && _op.isRead() )
			throw new LopsException(this.printErrorLocation() + "Data.getInstructions(): Exepecting a SCALAR data type, encountered " + getDataType());

		StringBuilder sb = new StringBuilder();
		if(this.getExecType() == ExecType.SPARK)
			sb.append( "SPARK" );
		else
			sb.append( "CP" );
		sb.append( OPERAND_DELIMITOR );
		if (_op.isRead()) {
			sb.append( Opcodes.READ.toString() );
			sb.append( OPERAND_DELIMITOR );
			sb.append ( this.prepInputOperand(input1) );
		}
		else if (_op.isWrite()) {
			sb.append( "write" );
			sb.append( OPERAND_DELIMITOR );
			sb.append ( getInputs().get(0).prepInputOperand(input1) );
		}
		else
			throw new LopsException(this.printErrorLocation() + "In Data Lop, Unknown operation: " + _op);

		sb.append( OPERAND_DELIMITOR );
		Lop fnameLop = _inputParams.get(DataExpression.IO_FILENAME);
		boolean literal = (fnameLop instanceof Data && ((Data)fnameLop).isLiteral());
		sb.append ( prepOperand(input2, DataType.SCALAR,  ValueType.STRING, literal) );

		// attach outputInfo in case of matrices
		OutputParameters oparams = getOutputParameters();
		if ( _op.isWrite() ) {
			sb.append( OPERAND_DELIMITOR );
			// scalars will always be written in text format
			FileFormat fmt = getDataType().isScalar() ?
				FileFormat.TEXT : oparams.getFormat();
			
			//format literal or variable
			Lop fmtLop = _inputParams.get(DataExpression.FORMAT_TYPE);
			String fmtLabel = (fmt!=FileFormat.UNKNOWN) ? fmt.toString() : fmtLop.getOutputParameters().getLabel();
			sb.append(prepOperand(fmtLabel, DataType.SCALAR, ValueType.STRING,
				(fmtLop instanceof Data && ((Data)fmtLop).isLiteral()))); //even fmtLop may be Data literal
			
			if(oparams.getFormat() == FileFormat.CSV) {
				Data headerLop = (Data) getNamedInputLop(DataExpression.DELIM_HAS_HEADER_ROW);
				Data delimLop = (Data) getNamedInputLop(DataExpression.DELIM_DELIMITER);
				Data sparseLop = (Data) getNamedInputLop(DataExpression.DELIM_SPARSE);

				if (headerLop.isVariable())
					throw new LopsException(this.printErrorLocation()
							+ "Parameter " + DataExpression.DELIM_HAS_HEADER_ROW
							+ " must be a literal for a seq operation.");
				if (delimLop.isVariable())
					throw new LopsException(this.printErrorLocation()
							+ "Parameter " + DataExpression.DELIM_DELIMITER
							+ " must be a literal for a seq operation.");
				if (sparseLop.isVariable())
					throw new LopsException(this.printErrorLocation()
							+ "Parameter " + DataExpression.DELIM_SPARSE
							+ " must be a literal for a seq operation.");

				sb.append(OPERAND_DELIMITOR);
				sb.append(headerLop.getBooleanValue());
				sb.append(OPERAND_DELIMITOR);
				sb.append(delimLop.getStringValue());
				sb.append(OPERAND_DELIMITOR);
				sb.append(sparseLop.getBooleanValue());

				if ( this.getExecType() == ExecType.SPARK )
				{
					sb.append(OPERAND_DELIMITOR);
					sb.append(true); //isInputMatrixBlock
				}
			}

			if(oparams.getFormat() == FileFormat.LIBSVM) {
				Data delimLop = (Data) getNamedInputLop(DataExpression.DELIM_DELIMITER);
				Data indexDelimLop = (Data) getNamedInputLop(DataExpression.LIBSVM_INDEX_DELIM);
				Data sparseLop = (Data) getNamedInputLop(DataExpression.DELIM_SPARSE);

				if(delimLop.isVariable())
					throw new LopsException(
						this.printErrorLocation() + "Parameter " + DataExpression.DELIM_DELIMITER + " must be a literal for a seq operation.");

				if(indexDelimLop.isVariable())
					throw new LopsException(
						this.printErrorLocation() + "Parameter " + DataExpression.LIBSVM_INDEX_DELIM + " must be a literal for a seq operation.");

				if (sparseLop.isVariable())
					throw new LopsException(this.printErrorLocation()
							+ "Parameter " + DataExpression.DELIM_SPARSE
							+ " must be a literal for a seq operation.");

				sb.append(OPERAND_DELIMITOR);
				sb.append(delimLop.getStringValue());
				sb.append(OPERAND_DELIMITOR);
				sb.append(indexDelimLop.getStringValue());
				sb.append(OPERAND_DELIMITOR);
				sb.append(sparseLop.getBooleanValue());

				if(this.getExecType() == ExecType.SPARK) {
					sb.append(OPERAND_DELIMITOR);
					sb.append(true); //isInputMatrixBlock
				}
			}

			if(oparams.getFormat() == FileFormat.HDF5) {
				Data datasetNameLop = (Data) getNamedInputLop(DataExpression.HDF5_DATASET_NAME);
				if(datasetNameLop.isVariable())
					throw new LopsException(
						this.printErrorLocation() + "Parameter " + DataExpression.HDF5_DATASET_NAME + " must be a literal for a seq operation.");

				sb.append(OPERAND_DELIMITOR);
				sb.append(datasetNameLop.getStringValue());

				if(this.getExecType() == ExecType.SPARK) {
					sb.append(OPERAND_DELIMITOR);
					sb.append(true); //isInputMatrixBlock
				}
			}

		}

		if (_op.isWrite()) {
			sb.append(OPERAND_DELIMITOR);
			Lop descriptionLop = getInputParams().get(DataExpression.DESCRIPTIONPARAM);
			if (descriptionLop != null) {
				boolean descLiteral = (descriptionLop instanceof Data && ((Data) descriptionLop).isLiteral());
				sb.append(prepOperand(descriptionLop.getOutputParameters().getLabel(), DataType.SCALAR,
						ValueType.STRING, descLiteral));
			} else {
				sb.append(prepOperand("", DataType.SCALAR, ValueType.STRING, true));
			}
			sb.append(OPERAND_DELIMITOR);
			sb.append(oparams.getBlocksize());
		}

		return sb.toString();
	}

	/**
	 * Method to generate createvar instruction that updates symbol table with metadata, hdfsfile name, etc.
	 *
	 */
	@Override
	public String getInstructions() {
		return getCreateVarInstructions(getOutputParameters().getFile_name(), getOutputParameters().getLabel());
	}

	@Override
	public String getInstructions(String outputFileName) {
		return getCreateVarInstructions(outputFileName, getOutputParameters().getLabel() );
	}

	public String getCreateVarInstructions(String outputFileName, String outputLabel) {
		if ( getDataType() == DataType.MATRIX || getDataType() == DataType.FRAME || getDataType() == DataType.LIST ) {

			if ( _op.isTransient() )
				throw new LopsException("getInstructions() should not be called for transient nodes.");

			OutputParameters oparams = getOutputParameters();

			StringBuilder sb = InstructionUtils.getStringBuilder();
			sb.append( "CP" );
			sb.append( OPERAND_DELIMITOR );
			sb.append( "createvar" );
			sb.append( OPERAND_DELIMITOR );
			sb.append( outputLabel );
			sb.append( OPERAND_DELIMITOR );
			sb.append( outputFileName );
			sb.append( OPERAND_DELIMITOR );
			sb.append( false );
			sb.append( OPERAND_DELIMITOR );
			sb.append( getDataType() );
			sb.append( OPERAND_DELIMITOR ); // only persistent reads come here!
			sb.append( oparams.getFormat().toString() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( oparams.getNumRows() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( oparams.getNumCols() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( oparams.getBlocksize() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( oparams.getNnz() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( oparams.getUpdateType().toString().toLowerCase() );

			// Format-specific properties
			if ( oparams.getFormat() == FileFormat.CSV ) {
				sb.append( OPERAND_DELIMITOR );
				sb.append( createVarCSVHelper() );
			}
			// Format-specific properties
			if ( oparams.getFormat() == FileFormat.LIBSVM ) {
				sb.append(OPERAND_DELIMITOR);
				sb.append( createVarLIBSVMHelper() );
			}

			// Format-specific properties
			if ( oparams.getFormat() == FileFormat.HDF5 ) {
				sb.append(OPERAND_DELIMITOR);
				sb.append( createVarHDF5Helper() );
			}

			// Frame-specific properties
			if( getDataType()==DataType.FRAME ) {
				Data schema = (Data) getNamedInputLop(DataExpression.SCHEMAPARAM);
				sb.append( OPERAND_DELIMITOR );
				sb.append( (schema!=null) ? schema.prepScalarLabel() : "*" );
			}
			return sb.toString();
		}
		else {
			throw new LopsException(this.printErrorLocation() + "In Data Lop, Unexpected data type " + getDataType());
		}
	}

	/**
	 * Helper function that attaches CSV format-specific properties to createvar instruction.
	 * The set of properties that are attached for a READ operation is different from that for a WRITE operation.
	 *
	 * @return instruction with csv format properties appended
	 */
	private String createVarCSVHelper() {
		StringBuilder sb = new StringBuilder();
		if ( _op.isRead() ) {
			Data headerLop = (Data) getNamedInputLop(DataExpression.DELIM_HAS_HEADER_ROW);
			Data delimLop = (Data) getNamedInputLop(DataExpression.DELIM_DELIMITER);
			Data fillLop = (Data) getNamedInputLop(DataExpression.DELIM_FILL);
			Data fillValueLop = (Data) getNamedInputLop(DataExpression.DELIM_FILL_VALUE);
			Lop naLop = getNamedInputLop(DataExpression.DELIM_NA_STRINGS);

			sb.append(headerLop.getBooleanValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(delimLop.getStringValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(fillLop.getBooleanValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(fillValueLop.getDoubleValue());
			if ( naLop != null ) {
				sb.append(OPERAND_DELIMITOR);
				if(naLop instanceof Nary){
					Nary naLops = (Nary) naLop;
					for(Lop na : naLops.getInputs()){
						sb.append(((Data)na).getStringValue());
						sb.append(DataExpression.DELIM_NA_STRING_SEP);
					}
				} else if (naLop instanceof Data){

					sb.append(((Data)naLop).getStringValue());
				}
			}
		}
		else { // (operation == OperationTypes.WRITE)
			Data headerLop = (Data) getNamedInputLop(DataExpression.DELIM_HAS_HEADER_ROW);
			Data delimLop = (Data) getNamedInputLop(DataExpression.DELIM_DELIMITER);
			Data sparseLop = (Data) getNamedInputLop(DataExpression.DELIM_SPARSE);

			if (headerLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.DELIM_HAS_HEADER_ROW
						+ " must be a literal for a seq operation.");
			if (delimLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.DELIM_DELIMITER
						+ " must be a literal for a seq operation.");
			if (sparseLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.DELIM_SPARSE
						+ " must be a literal for a seq operation.");

			sb.append(headerLop.getBooleanValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(delimLop.getStringValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(sparseLop.getBooleanValue());
		}
		return sb.toString();
	}

	private String createVarLIBSVMHelper() {
		StringBuilder sb = new StringBuilder();
		if ( _op.isRead() ) {
			Data delimLop = (Data) getNamedInputLop(DataExpression.DELIM_DELIMITER);
			Data indexDelimlLop = (Data) getNamedInputLop(DataExpression.LIBSVM_INDEX_DELIM);

			sb.append(delimLop.getStringValue());
				sb.append(OPERAND_DELIMITOR);
			sb.append(indexDelimlLop.getStringValue());
			sb.append(OPERAND_DELIMITOR);
		}
		else { // (operation == OperationTypes.WRITE)
			Data delimLop = (Data) getNamedInputLop(DataExpression.DELIM_DELIMITER);
			Data indexDelimLop = (Data) getNamedInputLop(DataExpression.LIBSVM_INDEX_DELIM);
			Data sparseLop = (Data) getNamedInputLop(DataExpression.DELIM_SPARSE);

			if(delimLop.isVariable())
				throw new LopsException(
					this.printErrorLocation() + "Parameter " + DataExpression.DELIM_DELIMITER + " must be a literal for a seq operation.");

			if(indexDelimLop.isVariable())
				throw new LopsException(
					this.printErrorLocation() + "Parameter " + DataExpression.LIBSVM_INDEX_DELIM + " must be a literal for a seq operation.");

			if (sparseLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.DELIM_SPARSE
						+ " must be a literal for a seq operation.");

			sb.append(delimLop.getStringValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(indexDelimLop.getStringValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(sparseLop.getBooleanValue());
		}
		return sb.toString();
	}

	private String createVarHDF5Helper() {
		StringBuilder sb = new StringBuilder();
		if ( _op.isRead() ) {
			Data dataset = (Data) getNamedInputLop(DataExpression.HDF5_DATASET_NAME);
			sb.append(dataset != null ? dataset.getStringValue() : "*");
			sb.append(OPERAND_DELIMITOR);
		}
		else { // (operation == OperationTypes.WRITE)
			Data datasetNameLop = (Data) getNamedInputLop(DataExpression.HDF5_DATASET_NAME);
			if(datasetNameLop.isVariable())
				throw new LopsException(
					this.printErrorLocation() + "Parameter " + DataExpression.HDF5_DATASET_NAME + " must be a literal for a seq operation.");

			sb.append(datasetNameLop.getStringValue());
			sb.append(OPERAND_DELIMITOR);
		}
		return sb.toString();
	}
}

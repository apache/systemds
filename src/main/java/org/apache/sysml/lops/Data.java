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

package org.apache.sysml.lops;

import java.util.HashMap;

import org.apache.sysml.hops.Hop.FileFormatTypes;
import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.OutputParameters.Format;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;



/**
 * Lop to represent data objects. Data objects represent matrices, vectors, 
 * variables, literals. Can be for both input and output. 
 */

public class Data extends Lop  
{
	
	public enum OperationTypes {READ,WRITE};
	
	FileFormatTypes formatType = FileFormatTypes.BINARY;
	OperationTypes operation;
	boolean literal_var = false;
	boolean transient_var = false;
	
	private HashMap<String, Lop> _inputParams;

	/**
	 * Method to create literal LOPs.
	 * 
	 * @param vt
	 * @param literalValue
	 * @return
	 * @throws LopsException
	 */
	public static Data createLiteralLop(ValueType vt, String literalValue) throws LopsException {
		// All literals have default format type of TEXT
		return new Data(OperationTypes.READ, null, null, null, literalValue, DataType.SCALAR, vt, false, FileFormatTypes.BINARY);
	}
	
	/**
	 * Constructor to setup read or write LOP
	 * In case of write: <code>input</code> must be provided. This will always be added as the first element in <code>input</code> array.
	 * For literals: this function is invoked through the static method <code>createLiteralLop</code>.
	 * 
	 * @param op
	 * @param input
	 * @param inputParametersLops
	 * @param name
	 * @param literal
	 * @param dt
	 * @param vt
	 * @param isTransient
	 * @throws LopsException
	 */
	public Data(Data.OperationTypes op, Lop input, HashMap<String, Lop> 
	inputParametersLops, String name, String literal, DataType dt, ValueType vt, boolean isTransient, FileFormatTypes fmt) throws LopsException 
	{
		super(Lop.Type.Data, dt, vt);	
		
		operation = op;	
		
		transient_var = isTransient;
		
		// Either <code>name</code> or <code>literal</code> can be non-null.
		if(literal != null){
			if (transient_var )
				throw new LopsException("Invalid parameter values while setting up a Data LOP -- transient flag is invalid for a literal.");
			literal_var = true;
			this.getOutputParameters().setLabel(literal);
		}
		else if(name != null) {
			if ( transient_var )
				this.getOutputParameters().setLabel(name); // tvar+name
			else
				this.getOutputParameters().setLabel("p"+op+name);
		}
		else {
			throw new LopsException("Invalid parameter values while setting up a Data LOP -- the lop must have either literal value or a name.");
		}
		
		// WRITE operation must have an input Lops, we always put this
		// input Lops as the first element of WRITE input. The parameters of
		// WRITE operation are then put as the following input elements.
		if(input != null && operation == OperationTypes.WRITE)
		{
			this.addInput(input);
			input.addOutput(this);
		}
		
		_inputParams = inputParametersLops;
		
		if ( _inputParams != null ) {
			for (Lop lop : inputParametersLops.values()) {
				this.addInput(lop);
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
		
		setFileFormatAndProperties(fmt);
	}

	private void setLopProperties() {
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		

		if ( getFileFormatType() == FileFormatTypes.CSV ) {
			Lop input = getInputs().get(0);
			// If the input is data transform, then csv write can be piggybacked onto TRANSFORM job.
			// Otherwise, the input must be converted to csv format via WriteCSV MR job.
			if ( input instanceof ParameterizedBuiltin 
					&& ((ParameterizedBuiltin)input).getOp() == org.apache.sysml.lops.ParameterizedBuiltin.OperationTypes.TRANSFORM ) {
				lps.addCompatibility(JobType.TRANSFORM);
				definesMRJob = false;
			}
			else {
				lps.addCompatibility(JobType.CSV_WRITE);
				definesMRJob = true;
			}
		}
		else {
			/*
			 *  This lop can be executed in all job types except for RAND, PARTITION.
			 *  RAND: because all inputs must be of type random. 
			 */
			lps.addCompatibility(JobType.ANY);
			// reads are not compatible with RAND because RAND must have all inputs that are random
			if ( operation == OperationTypes.READ )
				lps.removeCompatibility(JobType.DATAGEN);
			else if ( operation == OperationTypes.WRITE ) {
				// WRITE lops are not compatible with jobs that produce an 
				// intermediate output, which MUST be consumed by other subsequent lops 
				lps.removeCompatibility(JobType.MMCJ);
				// If at all, SORT job can only write in BinaryBlock format
				if(this.getDataType() == DataType.MATRIX && formatType != FileFormatTypes.BINARY)
					lps.removeCompatibility(JobType.SORT);
				lps.removeCompatibility(JobType.COMBINE);
			}
		}
		// ExecType is invalid for Data lop
		this.lps.setProperties ( inputs, ExecType.INVALID, ExecLocation.Data, breaksAlignment, aligner, definesMRJob );
	}
	
	/**
	 * Data-Lop-specific method to set the execution type for persistent write.
	 * TODO: split lops into MR/CP lop. 
	 * 
	 * @param et
	 */
	public void setExecType( ExecType et )
	{
		lps.execType = et;
	}
	
	/**
	 * Method to set format types for input, output files. 
	 * @param type
	 * @throws LopsException 
	 */
	public void setFileFormatAndProperties(FileFormatTypes type) throws LopsException 
	{
		this.formatType = type ;
		if(type == FileFormatTypes.BINARY)
			this.outParams.setFormat(Format.BINARY) ;
		else if(type == FileFormatTypes.TEXT)
			this.outParams.setFormat(Format.TEXT) ;
		else if (type == FileFormatTypes.MM)
			this.outParams.setFormat(Format.MM);
		else if (type == FileFormatTypes.CSV )
			this.outParams.setFormat(Format.CSV);
		else 
			throw new LopsException("Unexpected format: " + type);
		setLopProperties();
	}

	/**
	 * method to get format type for input, output files. 
	 * @return
	 */
	public FileFormatTypes getFileFormatType() 
	{
		return formatType ;
	}
 
	@Override
	public String toString() {
		
		return getID() + ":" + "File_Name: " + this.getOutputParameters().getFile_name() + " " + 
		"Label: " + this.getOutputParameters().getLabel() + " " + "Operation: = " + operation + " " + 
		"Format: " + this.outParams.getFormat() +  " Datatype: " + getDataType() + " Valuetype: " + getValueType() + " num_rows = " + this.getOutputParameters().getNumRows() + " num_cols = " + 
		this.getOutputParameters().getNumCols() + " UpdateInPlace: " + this.getOutputParameters().getUpdateType();
	}

	/**
	 * method to get operation type, i.e. read/write.
	 * @return
	 */
	 
	public OperationTypes getOperationType()
	{
		return operation;
	}
	
	/**
	 * method to get inputParams 
	 * @return
	 */
	public HashMap<String, Lop> getInputParams(){
		return _inputParams;
	}
	
	public Lop getNamedInputLop(String name) {
		return _inputParams.get(name);
	}
	
	public Lop getNamedInputLop(String name, String defaultVal) 
		throws LopsException {
		if( _inputParams.containsKey(name) )
			return _inputParams.get(name);
		else
			return Data.createLiteralLop(ValueType.STRING, defaultVal);
	}
	
	/**
	 * method to check if this data lop represents a literal.
	 * @return
	 */
	
	public boolean isLiteral()
	{
		return literal_var;

	}
	
	public boolean getBooleanValue() throws LopsException {
		if(literal_var) {
			return Boolean.parseBoolean(getOutputParameters().getLabel());
		}
		else
			throw new LopsException("Cannot obtain the value of a non-literal variable at compile time.");
	}
	
	public double getDoubleValue() throws LopsException {
		if(literal_var) {
			return Double.parseDouble(getOutputParameters().getLabel());
		}
		else
			throw new LopsException("Cannot obtain the value of a non-literal variable at compile time.");
	}
	
	public long getLongValue() throws LopsException {
		if(literal_var) {
			ValueType vt = getValueType();
			switch(vt) {
			case INT:
				return Long.parseLong(getOutputParameters().getLabel());
			case DOUBLE:
				return (long) Double.parseDouble(getOutputParameters().getLabel());
			
			default:
				throw new LopsException("Encountered a non-numeric value " + (vt) + ", while a numeric value is expected.");
			}
		}
		else
			throw new LopsException("Can not obtain the value of a non-literal variable at compile time.");
	}
	
	public String getStringValue() throws LopsException {
		if(literal_var) {
			return getOutputParameters().getLabel();
		}
		else
			throw new LopsException("Cannot obtain the value of a non-literal variable at compile time.");
	}
	
	/**
	 * Method to check if this represents a transient variable.
	 * @return
	 */
	public boolean isTransient()
	{
		return transient_var;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isPersistentWrite()
	{
		return operation == OperationTypes.WRITE && !transient_var;
	}
	
	/**
	 * Method to generate appropriate MR write instructions.
	 * Explicit write instructions are generated only in case of external file formats 
	 * (e.g., CSV) except for MatrixMarket. MM format is overridden by TextCell, instead.
	 * 
	 */
	@Override
	public String getInstructions(int input_index, int output_index) throws LopsException {
		
		OutputParameters oparams = getOutputParameters();

		if ( operation != OperationTypes.WRITE )
			throw new LopsException("This method should only be executed for generating MR Write instructions.");
		
		if ( oparams.getFormat() != Format.CSV )
			throw new LopsException("MR Write instructions can not be generated for the output format: " + oparams.getFormat() );
		
		StringBuilder sb = new StringBuilder();
		sb.append( "MR" );
		sb.append( OPERAND_DELIMITOR );
		
		// Generate opcode based on the output format
		if ( oparams.getFormat() == Format.CSV )
			sb.append( "csvwrite" );
		else 
			throw new LopsException("MR Write instructions can not be generated for the output format: " + oparams.getFormat() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append ( getInputs().get(0).prepInputOperand(input_index));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output_index));

		// Attach format-specific properties
		if(oparams.getFormat() == Format.CSV) {
			Data headerLop = (Data) getNamedInputLop(DataExpression.DELIM_HAS_HEADER_ROW);
			Data delimLop = (Data) getNamedInputLop(DataExpression.DELIM_DELIMITER);
			Data sparseLop = (Data) getNamedInputLop(DataExpression.DELIM_SPARSE);
				
			if (headerLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.DELIM_HAS_HEADER_ROW
						+ " must be a literal.");
			if (delimLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.DELIM_DELIMITER
						+ " must be a literal.");
			if (sparseLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.DELIM_SPARSE
						+ " must be a literal.");

			sb.append(OPERAND_DELIMITOR);
			sb.append(headerLop.getBooleanValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(delimLop.getStringValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(sparseLop.getBooleanValue());
		}
		else {
			throw new LopsException("MR Write instructions can not be generated for the output format: " + oparams.getFormat() );
		}
		
		return sb.toString();
	}

	/**
	 * Method to get CP instructions for reading/writing scalars and matrices from/to HDFS.
	 * This method generates CP read/write instructions.
	 */
	@Override
	public String getInstructions(String input1, String input2) 
		throws LopsException 
	{	
		if ( getOutputParameters().getFile_name() == null && operation == OperationTypes.READ ) 
			throw new LopsException(this.printErrorLocation() + "Data.getInstructions(): Exepecting a SCALAR data type, encountered " + getDataType());
			
		StringBuilder sb = new StringBuilder();
		if(this.getExecType() == ExecType.SPARK)  
			sb.append( "SPARK" );
		else
			sb.append( "CP" );
		sb.append( OPERAND_DELIMITOR );
		if ( operation == OperationTypes.READ ) 
		{
			sb.append( "read" );
			sb.append( OPERAND_DELIMITOR );
			sb.append ( this.prepInputOperand(input1) );
		}
		else if ( operation == OperationTypes.WRITE)
		{
			sb.append( "write" );
			sb.append( OPERAND_DELIMITOR );
			sb.append ( getInputs().get(0).prepInputOperand(input1) );
		}
		else
			throw new LopsException(this.printErrorLocation() + "In Data Lop, Unknown operation: " + operation);
		
		sb.append( OPERAND_DELIMITOR );			
		Lop fnameLop = _inputParams.get(DataExpression.IO_FILENAME);
		boolean literal = (fnameLop instanceof Data && ((Data)fnameLop).isLiteral());
		sb.append ( prepOperand(input2, DataType.SCALAR,  ValueType.STRING, literal) ); 
		
		// attach outputInfo in case of matrices
		OutputParameters oparams = getOutputParameters();
		if ( operation == OperationTypes.WRITE ) {
			sb.append( OPERAND_DELIMITOR );
			String fmt = "";
			if ( getDataType() == DataType.MATRIX || getDataType() == DataType.FRAME ) {
				if ( oparams.getFormat() == Format.MM )
					fmt = "matrixmarket";
				else if (oparams.getFormat() == Format.TEXT)
					fmt = "textcell";
				else if (oparams.getFormat() == Format.CSV)
					fmt = "csv";
				else if ( oparams.getFormat() == Format.BINARY ){
					if ( oparams.getRowsInBlock() > 0 || oparams.getColsInBlock() > 0 )
						fmt = "binaryblock"; 
					else
						fmt = "binarycell" ;
				}
				else {
					throw new LopsException("Unexpected format: " + oparams.getFormat());
				}
			}
			else {
				// scalars will always be written in text format
				fmt = "textcell";
			}
			
			sb.append( prepOperand(fmt, DataType.SCALAR, ValueType.STRING, true));
			
			if(oparams.getFormat() == Format.CSV) {
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
					boolean isInputMatrixBlock = true;
					Lop input = getInputs().get(0);
					if ( input instanceof ParameterizedBuiltin 
							&& ((ParameterizedBuiltin)input).getOp() == ParameterizedBuiltin.OperationTypes.TRANSFORM ) {
						// in the case of transform input, the input will be Text strings insteadof MatrixBlocks 
						// This information is used to have correct class information while accessing RDDs from the symbol table 
						isInputMatrixBlock = false;
					}
					sb.append(OPERAND_DELIMITOR);
					sb.append(isInputMatrixBlock);
				}
			}
			
		}
		
		return sb.toString();
	}
	
	/**
	 * Method to generate createvar instruction that updates symbol table with metadata, hdfsfile name, etc.
	 * 
	 * @throws LopsException 
	 */
	public String getInstructions() throws LopsException {
		return getCreateVarInstructions(getOutputParameters().getFile_name(), getOutputParameters().getLabel());
	}
	
	public String getInstructions(String outputFileName) throws LopsException {
		return getCreateVarInstructions(outputFileName, getOutputParameters().getLabel() );
	}
	
	public String getCreateVarInstructions(String outputFileName, String outputLabel) throws LopsException {
		if ( getDataType() == DataType.MATRIX || getDataType() == DataType.FRAME ) {
			
			if ( isTransient() )
				throw new LopsException("getInstructions() should not be called for transient nodes.");
			
			OutputParameters oparams = getOutputParameters();
			String fmt = "";
			if ( oparams.getFormat() == Format.TEXT )
				fmt = "textcell";
			else if ( oparams.getFormat() == Format.MM )
				fmt = "matrixmarket";
			else if ( oparams.getFormat() == Format.CSV )
				fmt = "csv";
			else { //binary
				fmt = ( getDataType() == DataType.FRAME || oparams.getRowsInBlock() > 0 
					|| oparams.getColsInBlock() > 0 ) ? "binaryblock" : "binarycell";
			}
			
			StringBuilder sb = new StringBuilder();
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
			sb.append( fmt );
			sb.append( OPERAND_DELIMITOR );
			sb.append( oparams.getNumRows() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( oparams.getNumCols() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( oparams.getRowsInBlock() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( oparams.getColsInBlock() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( oparams.getNnz() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( oparams.getUpdateType().toString().toLowerCase() );
			
			// Format-specific properties
			if ( oparams.getFormat() == Format.CSV ) {
				sb.append( OPERAND_DELIMITOR );
				sb.append( createVarCSVHelper() );
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
	 * @return
	 * @throws LopsException
	 */
	private String createVarCSVHelper() throws LopsException {
		StringBuilder sb = new StringBuilder();
		if ( operation == OperationTypes.READ ) {
			Data headerLop = (Data) getNamedInputLop(DataExpression.DELIM_HAS_HEADER_ROW);
			Data delimLop = (Data) getNamedInputLop(DataExpression.DELIM_DELIMITER);
			Data fillLop = (Data) getNamedInputLop(DataExpression.DELIM_FILL); 
			Data fillValueLop = (Data) getNamedInputLop(DataExpression.DELIM_FILL_VALUE);
			Data naLop = (Data) getNamedInputLop(DataExpression.DELIM_NA_STRINGS);
			
			sb.append(headerLop.getBooleanValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(delimLop.getStringValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(fillLop.getBooleanValue());
			sb.append(OPERAND_DELIMITOR);
			sb.append(fillValueLop.getDoubleValue());
			if ( naLop != null ) {
				sb.append(OPERAND_DELIMITOR);
				sb.append(naLop.getStringValue());
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
}
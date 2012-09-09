package com.ibm.bi.dml.lops;

import java.util.HashMap;

import com.ibm.bi.dml.hops.Hops.FileFormatTypes;
import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.OutputParameters.Format;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.parser.Expression.*;
import com.ibm.bi.dml.utils.LopsException;



/**
 * Lop to represent data objects. Data objects represent matrices, vectors, 
 * variables, literals. Can be for both input and output. 
 */

public class Data extends Lops  
{
	public enum OperationTypes {READ,WRITE};
	
	FileFormatTypes formatType;
	OperationTypes operation;
	boolean literal_var = false;
	boolean transient_var = false;
	
	private HashMap<String, Lops> _inputParams;

	/**
	 * Constructor to setup data lop.
	 * @param fName name of input file, null if not an input  file.
	 * @param op read or write lop
	 * @param name label for the lop. if provided, fName and literal should be null. should be null. 
	 * @param literal literal value for variable. if provided, fname and name should be null. 
	 * @param dt
	 * @param type
	 */
	public Data(String fName, Data.OperationTypes op, String name, String literal, DataType dt, ValueType vt, boolean isTransient) 
	{
		super(Lops.Type.Data, dt, vt);		
		operation = op;		
		
		if(literal != null)
		{
			literal_var = true;
			this.getOutputParameters().setLabel(literal);
		}

		transient_var = isTransient;
		if(name != null)
		{
			if ( transient_var )
				this.getOutputParameters().setLabel(name); // tvar+name
			else
				this.getOutputParameters().setLabel("p"+op+name);
		}

		this.getOutputParameters().setFile_name(fName);

		setLopProperties( );
	
	}
	
	// Constructor to setup data lop for read or write lops
	// If a user wants to create a write lops, input must be provided,
	// it will always be added to the first element of the Input Array
	public Data(Data.OperationTypes op, Lops input, HashMap<String, Lops> 
	inputParametersLops, String name, String literal, DataType dt, ValueType vt, boolean isTransient) throws LopsException 
	{
		super(Lops.Type.Data, dt, vt);	
		operation = op;	
		
		if(literal != null){
			literal_var = true;
			this.getOutputParameters().setLabel(literal);
		}

		transient_var = isTransient;
		if(name != null)
		{
			if ( transient_var )
				this.getOutputParameters().setLabel(name); // tvar+name
			else
				this.getOutputParameters().setLabel("p"+op+name);
		}
		
		// WRITE operation must have an input Lops, we always put this
		// input Lops as the first element of WRITE input. The parameters of
		// WRITE operation are then put as the following input elements.
		if(input != null && operation == OperationTypes.WRITE)
		{
			this.addInput(input);
			input.addOutput(this);
		}
		
		for (Lops lop : inputParametersLops.values()) {
			this.addInput(lop);
			lop.addOutput(this);
		}
		
		_inputParams = inputParametersLops;
		
		if (inputParametersLops.get(Statement.IO_FILENAME)!= null){
		
			OutputParameters outParams = ((Data)inputParametersLops.get(Statement.IO_FILENAME)).getOutputParameters();
			String fName = outParams.getLabel();
			this.getOutputParameters().setFile_name(fName);
		}
		
		setLopProperties( );
	}

	/**
	 * Same as other constructor, but with the ability to specify an input for write data lops.
	 * @param fName
	 * @param op
	 * @param input
	 * @param name
	 * @param literal
	 * @param dt
	 * @param type
	 */
	
	public Data(String fName, Data.OperationTypes op, Lops input, String name, String literal, DataType dt, ValueType vt, boolean isTransient) 
	{
		super(Lops.Type.Data, dt, vt);		
		operation = op;
		
		if(literal != null)
		{
			literal_var = true;
			this.getOutputParameters().setLabel(literal);
		}

		transient_var = isTransient;
		if(name != null)
		{
			if ( transient_var )
				this.getOutputParameters().setLabel(name);  // tvar+name
			else
				this.getOutputParameters().setLabel("p"+op+name);
		}
		
		this.getOutputParameters().setFile_name(fName);

		/** write operation must have an input **/
		if(input != null && operation == OperationTypes.WRITE)
		{
			this.addInput(input);
			input.addOutput(this);
		}
		
		setLopProperties();
	}
	
	private void setLopProperties() {
		/*
		 *  This lop can be executed in all job types except for RAND, PARTITION.
		 *  RAND: because all inputs must be of type random. 
		 */
		lps.addCompatibility(JobType.ANY);
		lps.removeCompatibility(JobType.PARTITION);
		// reads are not compatible with RAND because RAND must have all inputs that are random
		if ( operation == OperationTypes.READ )
			lps.removeCompatibility(JobType.RAND);
		else if ( operation == OperationTypes.WRITE ) {
			// WRITE lops are not compatible with jobs that produce an 
			// intermediate output, which MUST be consumed by other subsequent lops 
			lps.removeCompatibility(JobType.MMCJ);
			lps.removeCompatibility(JobType.SORT);
			lps.removeCompatibility(JobType.COMBINE);
		}
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		// ExecType is invalid for Data lop
		this.lps.setProperties ( ExecType.INVALID, ExecLocation.Data, breaksAlignment, aligner, definesMRJob );
	}
	
	/**
	 * Method to set format types for input, output files. 
	 * @param type
	 */
	public void setFileFormatType(FileFormatTypes type) 
	{
		this.formatType = type ;
		if(type == FileFormatTypes.BINARY)
			this.outParams.setFormat(Format.BINARY) ;
		else if(type == FileFormatTypes.TEXT)
			this.outParams.setFormat(Format.TEXT) ;
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
		
		return "File_Name: " + this.getOutputParameters().getFile_name() + " " + 
		"Label: " + this.getOutputParameters().getLabel() + " " + "Operation: = " + operation + " " + 
		"Format: " + this.outParams.getFormat() +  " Datatype: " + get_dataType() + " Valuetype: " + get_valueType() + " num_rows = " + this.getOutputParameters().getNum_rows() + " num_cols = " + 
		this.getOutputParameters().getNum_cols();
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
	public HashMap<String, Lops> getInputParams(){
		return _inputParams;
	}
	
	/**
	 * method to check if this data lop represents a literal.
	 * @return
	 */
	
	public boolean isLiteral()
	{
		return literal_var;

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
	 * Method to get instructions for reading/writing scalars from/to HDFS
	 * This method should be executed only when the data type = SCALAR
	 * In case of MR, Reads/Writes of matrices is done through inputs/outputs fields.
	 */
	@Override
	public String getInstructions(String input1, String input2) throws LopsException {
		
		
		if ( getOutputParameters().getFile_name() != null) {
			String str = "CP" + OPERAND_DELIMITOR;
			if ( operation == OperationTypes.READ ) 
				str += "read";
			else if ( operation == OperationTypes.WRITE)
				str += "write";
			else
				throw new LopsException(this.printErrorLocation() + "In Data Lop, Unknown operation: " + operation);
			
			str += OPERAND_DELIMITOR + 
					input1 +  
					DATATYPE_PREFIX + get_dataType() + 
					VALUETYPE_PREFIX + get_valueType() + 
					OPERAND_DELIMITOR +
					input2 + 
					DATATYPE_PREFIX + DataType.SCALAR + 
					VALUETYPE_PREFIX + ValueType.STRING;

			// attach outputInfo in case of matrices
			if ( operation == OperationTypes.WRITE ) {
				str += OPERAND_DELIMITOR;
				if ( get_dataType() == DataType.MATRIX ) {
					OutputParameters oparams = getOutputParameters();
					if ( oparams.getFormat() == Format.TEXT )
						str += "textcell";
					else {
						if ( oparams.get_rows_in_block() > 0 || oparams.get_cols_in_block() > 0 )
							str += "binaryblock";
						else
							str += "binarycell";
					}
				}
				else {
					// scalars will always be written in text format
					str += "textcell";
				}
				str += DATATYPE_PREFIX + DataType.SCALAR + 
						VALUETYPE_PREFIX + ValueType.STRING;
			}
			return str;
		}
		throw new LopsException(this.printErrorLocation() + "Data.getInstructions(): Exepecting a SCALAR data type, encountered " + get_dataType());
	}
	
	/**
	 * Method to generate an instruction that updates symbol table with metadata, hdfsfile name, etc.
	 * 
	 * @throws LopsException 
	 */
	public String getInstructions() throws LopsException {
		String inst = "CP" + OPERAND_DELIMITOR + "createvar";
		OutputParameters oparams = getOutputParameters();
		if ( get_dataType() == DataType.MATRIX ) {
			inst +=  OPERAND_DELIMITOR 
					+ oparams.getLabel() + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType() + OPERAND_DELIMITOR 
					+ oparams.getFile_name() + DATATYPE_PREFIX + DataType.SCALAR + VALUETYPE_PREFIX + ValueType.STRING + OPERAND_DELIMITOR
			        + oparams.getNum_rows() + OPERAND_DELIMITOR
			        + oparams.getNum_cols() + OPERAND_DELIMITOR
			        + oparams.get_rows_in_block() + OPERAND_DELIMITOR
			        + oparams.get_cols_in_block() + OPERAND_DELIMITOR
	        		+ oparams.getNnz() + OPERAND_DELIMITOR; 
			// TODO: following logic should change once we LOPs encode key-value-class information.
			if ( oparams.getFormat() == Format.TEXT )
				inst += "textcell";
			else {
				if ( oparams.get_rows_in_block() > 0 || oparams.get_cols_in_block() > 0 )
					inst += "binaryblock";
				else
					inst += "binarycell";
			}
			return inst;
		}
		else {
			throw new LopsException(this.printErrorLocation() + "In Data Lop, Unexpected data type " + get_dataType());
		}
	}
}
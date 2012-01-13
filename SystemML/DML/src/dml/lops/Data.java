package dml.lops;

import dml.hops.Hops.FileFormatTypes;
import dml.lops.LopProperties.ExecLocation;
import dml.lops.OutputParameters.Format;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;
import dml.utils.LopsException;

/**
 * Lop to represent data objects. Data objects represent matrices, vectors, 
 * variables, literals. Can be for both input and output. 
 * @author aghoting
 *
 */

public class Data extends Lops  
{
	public enum OperationTypes {READ,WRITE};
	
	FileFormatTypes formatType;
	OperationTypes operation;
	boolean literal_var = false;
	boolean transient_var = false;

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

		
		if(name != null)
		{
			this.getOutputParameters().setLabel(name);
		}
		transient_var = isTransient;
		
		
		
		this.getOutputParameters().setFile_name(fName);

		setLopProperties ( );
	
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

		
		if(name != null)
		{
			this.getOutputParameters().setLabel(name);
		}
		transient_var = isTransient;
		
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
		
		this.lps.setProperties ( ExecLocation.Data, breaksAlignment, aligner, definesMRJob );
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
		if ( get_dataType() == DataType.SCALAR && getOutputParameters().getFile_name() != null) {
			String str = "";
			if ( operation == OperationTypes.READ ) 
				str += "readScalar";
			else if ( operation == OperationTypes.WRITE)
				str += "writeScalar";
			else
				throw new LopsException("Unknown operation: " + operation);
			str += OPERAND_DELIMITOR + 
					input1 +  
					VALUETYPE_PREFIX + get_valueType() + 
					OPERAND_DELIMITOR +
					input2 + 
					VALUETYPE_PREFIX + ValueType.STRING;
			return str;
		}
		throw new LopsException("Data.getInstructions(): Exepecting a SCALAR data type, encountered " + get_dataType());
	}
}
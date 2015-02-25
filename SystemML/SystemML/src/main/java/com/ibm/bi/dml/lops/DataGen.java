/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import java.util.HashMap;

import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.OutputParameters.Format;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.parser.Expression.*;


/**
 * <p>Defines a LOP that generates data.</p>
 */
public class DataGen extends Lop
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String RAND_OPCODE = "rand"; //rand
	public static final String SEQ_OPCODE = "seq"; //sequence
	public static final String SINIT_OPCODE = "sinit"; //string initialize
	
	/** base dir for rand input */
	private String baseDir;
	
	private HashMap<String, Lop> _inputParams;
	DataGenMethod method;
	
	/**
	 * <p>Creates a new Rand-LOP. The target identifier has to hold the dimensions of the new random object.</p>
	 * 
	 * @param id target identifier
	 * @param inputParameterLops Lops of input parameters
	 * @param baseDir base dir for runtime
	 * @param dt Data type
	 * @param vt Value type
	 * @param ExecType Execution type
	 */	
	public DataGen(DataGenMethod mthd, DataIdentifier id, HashMap<String, Lop> 
				inputParametersLops, String baseDir, DataType dt, ValueType vt, ExecType et) throws LopsException 
	{
		super(Type.DataGen, dt, vt);
		method = mthd;
				
		for (Lop lop : inputParametersLops.values()) {
			this.addInput(lop);
			lop.addOutput(this);
		}
		
		_inputParams = inputParametersLops;
		
		init(id, baseDir, et);
	}

	public DataGenMethod getDataGenMethod() {
		return method;
	}
	
	public void init(DataIdentifier id, String baseDir, ExecType et)
	{
		this.getOutputParameters().setFormat(Format.BINARY);
		this.getOutputParameters().setBlocked(true);
		this.getOutputParameters().setNumRows(id.getDim1());
		this.getOutputParameters().setNumCols(id.getDim2());
		this.getOutputParameters().setNnz(-1);
		this.getOutputParameters().setRowsInBlock(id.getRowsInBlock());
		this.getOutputParameters().setColsInBlock(id.getColumnsInBlock());
		
		this.baseDir = baseDir;
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob;
		
		if ( et == ExecType.MR ) {
			definesMRJob = true;
			lps.addCompatibility(JobType.DATAGEN);
			this.lps.setProperties( inputs, et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob);
		}
		else {
			definesMRJob = false;
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}

	/**
	 * Function to generate CP instructions for data generation via Rand and Seq.
	 * Since DataGen Lop can have many inputs, ONLY the output variable name is 
	 * passed from piggybacking as the function argument <code>output</code>. 
	 */
	@Override
	public String getInstructions(String output) throws LopsException {
		switch(method) {
			case RAND:
				return getCPInstruction_Rand(output);
			case SINIT:
				return getCPInstruction_SInit(output);
			case SEQ:
				return getCPInstruction_Seq(output);
			
			default:
				throw new LopsException("Unknown data generation method: " + method);
		}
	}
	
	/**
	 * Private method that generates CP Instruction for Rand.
	 * @param output
	 * @return
	 * @throws LopsException
	 */
	private String getCPInstruction_Rand(String output) throws LopsException {
		
		if ( method != DataGenMethod.RAND )
			throw new LopsException("Invalid instruction generation for data generation method " + method);
		
		StringBuilder sb = new StringBuilder( );
		ExecType et = getExecType();
		sb.append( et );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		if (this.getInputs().size() == DataExpression.RAND_VALID_PARAM_NAMES.length) {

			Lop iLop = null;

			iLop = _inputParams.get(DataExpression.RAND_ROWS.toString());
			String rowsString = iLop.prepScalarLabel();
			
			iLop = _inputParams.get(DataExpression.RAND_COLS.toString());
			String colsString = iLop.prepScalarLabel();
			
			String rowsInBlockString = String.valueOf(this
					.getOutputParameters().getRowsInBlock());
			String colsInBlockString = String.valueOf(this
					.getOutputParameters().getColsInBlock());

			iLop = _inputParams.get(DataExpression.RAND_MIN.toString());
			String minString = iLop.getOutputParameters().getLabel();
			if (iLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.RAND_MIN
						+ " must be a literal for a Rand operation.");

			iLop = _inputParams.get(DataExpression.RAND_MAX.toString());
			String maxString = iLop.getOutputParameters().getLabel();
			if (iLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.RAND_MAX
						+ " must be a literal for a Rand operation.");

			iLop = _inputParams.get(DataExpression.RAND_SPARSITY.toString());
			String sparsityString = iLop.getOutputParameters().getLabel();
			if (iLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.RAND_SPARSITY
						+ " must be a literal for a Rand operation.");

			iLop = _inputParams.get(DataExpression.RAND_SEED.toString());
			String seedString = iLop.getOutputParameters().getLabel();
			if (iLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.RAND_SEED
						+ " must be a literal for a Rand operation.");

			iLop = _inputParams.get(DataExpression.RAND_PDF.toString());
			String pdfString = iLop.getOutputParameters().getLabel();
			if (iLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.RAND_PDF
						+ " must be a literal for a Rand operation.");

			sb.append(RAND_OPCODE);
			sb.append(OPERAND_DELIMITOR);
			/*sb.append(in1);
			sb.append(OPERAND_DELIMITOR);
			sb.append(in2);
			sb.append(OPERAND_DELIMITOR);*/
			sb.append(rowsString);
			sb.append(OPERAND_DELIMITOR);
			sb.append(colsString);
			sb.append(OPERAND_DELIMITOR);
			sb.append(rowsInBlockString);
			sb.append(OPERAND_DELIMITOR);
			sb.append(colsInBlockString);
			sb.append(OPERAND_DELIMITOR);
			sb.append(minString);
			sb.append(OPERAND_DELIMITOR);
			sb.append(maxString);
			sb.append(OPERAND_DELIMITOR);
			sb.append(sparsityString);
			sb.append(OPERAND_DELIMITOR);
			sb.append(seedString);
			sb.append(OPERAND_DELIMITOR);
			sb.append(pdfString);
			sb.append(OPERAND_DELIMITOR);
			if ( et == ExecType.MR ) {
				sb.append(baseDir);
				sb.append(OPERAND_DELIMITOR);
			}
			sb.append( this.prepOutputOperand(output));

			return sb.toString();

		} else {
			throw new LopsException(this.printErrorLocation()
					+ "Invalid number of operands (" + this.getInputs().size()
					+ ") for a Rand operation");
		}
	}

	/**
	 * 
	 * @param output
	 * @return
	 * @throws LopsException
	 */
	private String getCPInstruction_SInit(String output) 
		throws LopsException 
	{
		if ( method != DataGenMethod.SINIT )
			throw new LopsException("Invalid instruction generation for data generation method " + method);
		
		//prepare instruction parameters
		Lop iLop = _inputParams.get(DataExpression.RAND_ROWS.toString());
		String rowsString = iLop.prepScalarLabel();
		
		iLop = _inputParams.get(DataExpression.RAND_COLS.toString());
		String colsString = iLop.prepScalarLabel();
		
		String rowsInBlockString = String.valueOf(this
				.getOutputParameters().getRowsInBlock());
		String colsInBlockString = String.valueOf(this
				.getOutputParameters().getColsInBlock());

		iLop = _inputParams.get(DataExpression.RAND_MIN.toString());
		String minString = iLop.getOutputParameters().getLabel();
		if (iLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.RAND_MIN
					+ " must be a literal for a Rand operation.");

		//generate instruction		
		StringBuilder sb = new StringBuilder( );
		ExecType et = getExecType();
		
		sb.append( et );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append(SINIT_OPCODE);
		sb.append(OPERAND_DELIMITOR);
		sb.append(rowsString);
		sb.append(OPERAND_DELIMITOR);
		sb.append(colsString);
		sb.append(OPERAND_DELIMITOR);
		sb.append(rowsInBlockString);
		sb.append(OPERAND_DELIMITOR);
		sb.append(colsInBlockString);
		sb.append(OPERAND_DELIMITOR);
		sb.append(minString);
		sb.append(OPERAND_DELIMITOR);
		sb.append( this.prepOutputOperand(output));

		return sb.toString();
	}
	
	/**
	 * Private method that generates CP Instruction for Seq.
	 * @param output
	 * @return
	 * @throws LopsException
	 */
	private String getCPInstruction_Seq(String output) throws LopsException {
		if ( method != DataGenMethod.SEQ )
			throw new LopsException("Invalid instruction generation for data generation method " + method);
		
		StringBuilder sb = new StringBuilder( );
		ExecType et = getExecType();
		sb.append( et );
		sb.append( Lop.OPERAND_DELIMITOR );

		Lop iLop = null;

		iLop = _inputParams.get(Statement.SEQ_FROM.toString()); 
		String fromString = iLop.prepScalarLabel();
		
		iLop = _inputParams.get(Statement.SEQ_TO.toString());
		String toString = iLop.prepScalarLabel();
		
		iLop = _inputParams.get(Statement.SEQ_INCR.toString()); 
		String incrString = iLop.prepScalarLabel();
		
		String rowsString = String.valueOf(this.getOutputParameters().getNumRows());
		String colsString = String.valueOf(this.getOutputParameters().getNumCols());
		String rowsInBlockString = String.valueOf(this.getOutputParameters().getRowsInBlock());
		String colsInBlockString = String.valueOf(this.getOutputParameters().getColsInBlock());
		
		sb.append( DataGen.SEQ_OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( rowsString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( colsString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( rowsInBlockString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( colsInBlockString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( fromString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( toString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( incrString );
		sb.append( OPERAND_DELIMITOR );
		if ( et == ExecType.MR ) {
			sb.append( baseDir );
			sb.append( OPERAND_DELIMITOR );
		}
		sb.append( this.prepOutputOperand(output));

		return sb.toString();
	}
	
	@Override
	public String getInstructions(int inputIndex, int outputIndex) throws LopsException
	{
		switch(method) {
		case RAND:
			return getMRInstruction_Rand(inputIndex, outputIndex);
		case SEQ:
			return getMRInstruction_Seq(inputIndex, outputIndex);
			
		default:
			throw new LopsException("Unknown data generation method: " + method);
		}
	}
	
	/**
	 * Private method to generate MR instruction for Rand.
	 * 
	 * @param input_index
	 * @param output_index
	 * @return
	 * @throws LopsException
	 */
	private String getMRInstruction_Rand(int inputIndex, int outputIndex) throws LopsException {
		if (this.getInputs().size() == DataExpression.RAND_VALID_PARAM_NAMES.length) {

			StringBuilder sb = new StringBuilder();
			sb.append( getExecType() );
			sb.append( Lop.OPERAND_DELIMITOR );
			Lop iLop = null;

			iLop = _inputParams.get(DataExpression.RAND_ROWS.toString()); 
			String rowsString = iLop.prepScalarInputOperand(getExecType()); 
			
			iLop = _inputParams.get(DataExpression.RAND_COLS.toString()); 
			String colsString = iLop.prepScalarInputOperand(getExecType());
			
			String rowsInBlockString = String.valueOf(this.getOutputParameters().getRowsInBlock());
			String colsInBlockString = String.valueOf(this.getOutputParameters().getColsInBlock());
			
			iLop = _inputParams.get(DataExpression.RAND_MIN.toString()); 
			String minString = iLop.getOutputParameters().getLabel();
			if (iLop.isVariable())
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ DataExpression.RAND_MIN + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(DataExpression.RAND_MAX.toString()); 
			String maxString = iLop.getOutputParameters().getLabel();
			if (iLop.isVariable())
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ DataExpression.RAND_MAX + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(DataExpression.RAND_SPARSITY.toString()); 
			String sparsityString = iLop.getOutputParameters().getLabel();
			if (iLop.isVariable())
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ DataExpression.RAND_SPARSITY + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(DataExpression.RAND_SEED.toString()); 
			String seedString = iLop.getOutputParameters().getLabel();
			if (iLop.isVariable())
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ DataExpression.RAND_SEED + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(DataExpression.RAND_PDF.toString()); 
			String pdfString = iLop.getOutputParameters().getLabel();
			if (iLop.isVariable())
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ DataExpression.RAND_PDF + " must be a literal for a Rand operation.");
			
			sb.append( RAND_OPCODE );
			sb.append( OPERAND_DELIMITOR );
			sb.append( inputIndex );
			sb.append( OPERAND_DELIMITOR );
			sb.append( outputIndex );
			sb.append( OPERAND_DELIMITOR );
			sb.append( rowsString );
			sb.append( OPERAND_DELIMITOR );
			sb.append( colsString );
			sb.append( OPERAND_DELIMITOR );
			sb.append( rowsInBlockString );
			sb.append( OPERAND_DELIMITOR );
			sb.append( colsInBlockString );
			sb.append( OPERAND_DELIMITOR );
			sb.append( minString );
			sb.append( OPERAND_DELIMITOR );
			sb.append( maxString );
			sb.append( OPERAND_DELIMITOR );
			sb.append( sparsityString );
			sb.append( OPERAND_DELIMITOR );
			sb.append( seedString );
			sb.append( OPERAND_DELIMITOR );
			sb.append( pdfString );
			sb.append( OPERAND_DELIMITOR );
			sb.append( baseDir );

			return sb.toString();
			
		}
		 else {
				throw new LopsException(this.printErrorLocation() + "Invalid number of operands ("
						+ this.getInputs().size() + ") for a Rand operation");
			}
	}
	
	/**
	 * Private method to generate MR instruction for Seq.
	 * 
	 * @param input_index
	 * @param output_index
	 * @return
	 * @throws LopsException
	 */
	private String getMRInstruction_Seq(int inputIndex, int outputIndex) throws LopsException {
		StringBuilder sb = new StringBuilder();
		
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		Lop iLop = null;
		iLop = _inputParams.get(Statement.SEQ_FROM.toString()); 
		String fromString = iLop.getOutputParameters().getLabel();
		if ( (iLop.getExecLocation() == ExecLocation.Data &&
				 !((Data)iLop).isLiteral()) || !(iLop.getExecLocation() == ExecLocation.Data ))
			fromString = Lop.VARIABLE_NAME_PLACEHOLDER + fromString + Lop.VARIABLE_NAME_PLACEHOLDER;
		
		iLop = _inputParams.get(Statement.SEQ_TO.toString()); 
		String toString = iLop.getOutputParameters().getLabel();
		if ( iLop.getExecLocation() == ExecLocation.Data 
				&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
			toString = Lop.VARIABLE_NAME_PLACEHOLDER + toString + Lop.VARIABLE_NAME_PLACEHOLDER;
		
		iLop = _inputParams.get(Statement.SEQ_INCR.toString()); 
		String incrString = iLop.getOutputParameters().getLabel();
		if ( iLop.getExecLocation() == ExecLocation.Data 
				&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
			incrString = Lop.VARIABLE_NAME_PLACEHOLDER + incrString + Lop.VARIABLE_NAME_PLACEHOLDER;
		
		String rowsString = String.valueOf(this.getOutputParameters().getNumRows());
		String colsString = String.valueOf(this.getOutputParameters().getNumCols());
		String rowsInBlockString = String.valueOf(this.getOutputParameters().getRowsInBlock());
		String colsInBlockString = String.valueOf(this.getOutputParameters().getColsInBlock());
		
		sb.append( DataGen.SEQ_OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( inputIndex );
		sb.append( OPERAND_DELIMITOR );
		sb.append( outputIndex );
		sb.append( OPERAND_DELIMITOR );
		sb.append( rowsString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( colsString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( rowsInBlockString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( colsInBlockString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( fromString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( toString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( incrString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( baseDir );

		return sb.toString();
	}

	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append(method.toString());
		sb.append(" ; num_rows=" + this.getOutputParameters().getNumRows());
		sb.append(" ; num_cols=" + this.getOutputParameters().getNumCols());
		sb.append(" ; nnz=" + this.getOutputParameters().getNnz());
		sb.append(" ; num_rows_per_block=" + this.getOutputParameters().getRowsInBlock());
		sb.append(" ; num_cols_per_block=" + this.getOutputParameters().getColsInBlock());
		sb.append(" ; format=" + this.getOutputParameters().getFormat());
		sb.append(" ; blocked=" + this.getOutputParameters().isBlocked());
		sb.append(" ; dir=" + this.baseDir);
		return sb.toString();
	}
}

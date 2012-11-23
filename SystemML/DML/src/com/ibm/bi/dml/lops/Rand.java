package com.ibm.bi.dml.lops;

import java.util.HashMap;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.OutputParameters.Format;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.RandStatement;
import com.ibm.bi.dml.parser.Expression.*;
import com.ibm.bi.dml.utils.LopsException;


/**
 * <p>Defines a Rand-LOP.</p>
 */
public class Rand extends Lops
{
	/** base dir for rand input */
	private String baseDir;
	
	private HashMap<String, Lops> _inputParams;
	
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
	public Rand(DataIdentifier id, HashMap<String, Lops> 
	inputParametersLops, String baseDir, DataType dt, ValueType vt, ExecType et) throws LopsException 
	{
		super(Type.RandLop, dt, vt);	
				
		for (Lops lop : inputParametersLops.values()) {
			this.addInput(lop);
			lop.addOutput(this);
		}
		
		_inputParams = inputParametersLops;
		
		init(id, baseDir, et);
	}

	public void init(DataIdentifier id, String baseDir, ExecType et)
	{
		this.getOutputParameters().setFormat(Format.BINARY);
		this.getOutputParameters().blocked_representation = true;
		this.getOutputParameters().num_rows = id.getDim1();
		this.getOutputParameters().num_cols = id.getDim2();
		this.getOutputParameters()._nnz = -1;
		this.getOutputParameters().num_rows_in_block = id.getRowsInBlock();
		this.getOutputParameters().num_cols_in_block = id.getColumnsInBlock();
		
		this.baseDir = baseDir;
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob;
		
		if ( et == ExecType.MR ) {
			definesMRJob = true;
			lps.addCompatibility(JobType.RAND);
			this.lps.setProperties( et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob);
		}
		else {
			definesMRJob = false;
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}

	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String input6, String input7, String output) throws LopsException {
		
		StringBuilder sb = new StringBuilder( );
		sb.append( getExecType() );
		sb.append( Lops.OPERAND_DELIMITOR );
		String in1 = new String(""), in2 = new String("");
		
		if (this.getInputs().size() == RandStatement.RAND_VALID_PARAM_NAMES.length) {

			Lops iLop = null;

			iLop = _inputParams.get(RandStatement.RAND_ROWS.toString()); 
			String rowsString = iLop.getOutputParameters().getLabel();
			if ( (iLop.getExecLocation() == ExecLocation.Data &&
					 !((Data)iLop).isLiteral()) || !(iLop.getExecLocation() == ExecLocation.Data )){
				in1 = "" + rowsString + DATATYPE_PREFIX + iLop.get_dataType()+ VALUETYPE_PREFIX + iLop.get_valueType() ;
				rowsString = Lops.VARIABLE_NAME_PLACEHOLDER + rowsString + Lops.VARIABLE_NAME_PLACEHOLDER;
			}
			iLop = _inputParams.get(RandStatement.RAND_COLS.toString()); 
			String colsString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data )) {
				in2 = "" + colsString + DATATYPE_PREFIX + iLop.get_dataType()+ VALUETYPE_PREFIX + iLop.get_valueType() ;
				colsString = Lops.VARIABLE_NAME_PLACEHOLDER + colsString + Lops.VARIABLE_NAME_PLACEHOLDER;
			}
			String rowsInBlockString = String.valueOf(this.getOutputParameters().num_rows_in_block);
			String colsInBlockString = String.valueOf(this.getOutputParameters().num_cols_in_block);
			
			iLop = _inputParams.get(RandStatement.RAND_MIN.toString()); 
			String minString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ RandStatement.RAND_MIN + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(RandStatement.RAND_MAX.toString()); 
			String maxString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ))
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ RandStatement.RAND_MAX + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(RandStatement.RAND_SPARSITY.toString()); 
			String sparsityString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ))
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ RandStatement.RAND_SPARSITY + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(RandStatement.RAND_SEED.toString()); 
			String seedString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ RandStatement.RAND_SEED + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(RandStatement.RAND_PDF.toString()); 
			String pdfString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ RandStatement.RAND_PDF + " must be a literal for a Rand operation.");
			
			sb.append( "Rand" );
			sb.append( OPERAND_DELIMITOR );
			sb.append( in1 );
			sb.append( OPERAND_DELIMITOR );
			sb.append( in2 );
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
			sb.append( OPERAND_DELIMITOR );
			sb.append( output );
			sb.append( DATATYPE_PREFIX );
			sb.append( get_dataType() );
			sb.append( VALUETYPE_PREFIX );
			sb.append( get_valueType() );

			return sb.toString();
			
		}
		 else {
				throw new LopsException(this.printErrorLocation() + "Invalid number of operands ("
						+ this.getInputs().size() + ") for a Rand operation");
			}
	}
	
	@Override
	public String getInstructions(int inputIndex, int outputIndex) throws LopsException
	{
		
		if (this.getInputs().size() == RandStatement.RAND_VALID_PARAM_NAMES.length) {

			Lops iLop = null;

			iLop = _inputParams.get(RandStatement.RAND_ROWS.toString()); 
			String rowsString = iLop.getOutputParameters().getLabel();
			if ( (iLop.getExecLocation() == ExecLocation.Data &&
					 !((Data)iLop).isLiteral()) || !(iLop.getExecLocation() == ExecLocation.Data ))
				rowsString = Lops.VARIABLE_NAME_PLACEHOLDER + rowsString + Lops.VARIABLE_NAME_PLACEHOLDER;
			
			iLop = _inputParams.get(RandStatement.RAND_COLS.toString()); 
			String colsString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				colsString = Lops.VARIABLE_NAME_PLACEHOLDER + colsString + Lops.VARIABLE_NAME_PLACEHOLDER;
			
			String rowsInBlockString = String.valueOf(this.getOutputParameters().num_rows_in_block);
			String colsInBlockString = String.valueOf(this.getOutputParameters().num_cols_in_block);
			
			iLop = _inputParams.get(RandStatement.RAND_MIN.toString()); 
			String minString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ RandStatement.RAND_MIN + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(RandStatement.RAND_MAX.toString()); 
			String maxString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ RandStatement.RAND_MAX + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(RandStatement.RAND_SPARSITY.toString()); 
			String sparsityString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ RandStatement.RAND_SPARSITY + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(RandStatement.RAND_SEED.toString()); 
			String seedString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ RandStatement.RAND_SEED + " must be a literal for a Rand operation.");
			
			iLop = _inputParams.get(RandStatement.RAND_PDF.toString()); 
			String pdfString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				throw new LopsException(this.printErrorLocation() + "Parameter " 
						+ RandStatement.RAND_PDF + " must be a literal for a Rand operation.");
			
			StringBuilder sb = new StringBuilder();
			sb.append( getExecType() );
			sb.append( Lops.OPERAND_DELIMITOR );
			sb.append( "Rand" );
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

	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("Rand");
		sb.append(" ; num_rows=" + this.getOutputParameters().getNum_rows());
		sb.append(" ; num_cols=" + this.getOutputParameters().getNum_cols());
		sb.append(" ; nnz=" + this.getOutputParameters().getNnz());
		sb.append(" ; num_rows_per_block=" + this.getOutputParameters().get_rows_in_block());
		sb.append(" ; num_cols_per_block=" + this.getOutputParameters().get_cols_in_block());
		sb.append(" ; format=" + this.getOutputParameters().getFormat());
		sb.append(" ; blocked=" + this.getOutputParameters().isBlocked_representation());
		sb.append(" ; dir=" + this.baseDir);
		return sb.toString();
	}
}

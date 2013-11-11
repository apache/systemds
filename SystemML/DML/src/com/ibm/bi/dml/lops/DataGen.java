/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
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
import com.ibm.bi.dml.parser.RandStatement;
import com.ibm.bi.dml.parser.Expression.*;


/**
 * <p>Defines a LOP that generates data.</p>
 */
public class DataGen extends Lop
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		super(Type.RandLop, dt, vt);
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
			this.lps.setProperties( inputs, et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob);
		}
		else {
			definesMRJob = false;
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}

	@Override
	public String getInstructions(String output) throws LopsException {
		if ( method != DataGenMethod.SEQ )
			throw new LopsException("Invalid instruction generation for data generation method " + method);
		
		StringBuilder sb = new StringBuilder( );
		ExecType et = getExecType();
		sb.append( et );
		sb.append( Lop.OPERAND_DELIMITOR );

		Lop iLop = null;

		iLop = _inputParams.get(RandStatement.SEQ_FROM.toString()); 
		String fromString = iLop.getOutputParameters().getLabel();
		if ( (iLop.getExecLocation() == ExecLocation.Data &&
				 !((Data)iLop).isLiteral()) || !(iLop.getExecLocation() == ExecLocation.Data ))
			fromString = Lop.VARIABLE_NAME_PLACEHOLDER + fromString + Lop.VARIABLE_NAME_PLACEHOLDER;
		
		iLop = _inputParams.get(RandStatement.SEQ_TO.toString()); 
		String toString = iLop.getOutputParameters().getLabel();
		if ( iLop.getExecLocation() == ExecLocation.Data 
				&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
			toString = Lop.VARIABLE_NAME_PLACEHOLDER + toString + Lop.VARIABLE_NAME_PLACEHOLDER;
		
		iLop = _inputParams.get(RandStatement.SEQ_INCR.toString()); 
		String incrString = iLop.getOutputParameters().getLabel();
		if ( iLop.getExecLocation() == ExecLocation.Data 
				&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
			incrString = Lop.VARIABLE_NAME_PLACEHOLDER + incrString + Lop.VARIABLE_NAME_PLACEHOLDER;
		
		String rowsString = String.valueOf(this.getOutputParameters().getNum_rows());
		String colsString = String.valueOf(this.getOutputParameters().getNum_cols());
		String rowsInBlockString = String.valueOf(this.getOutputParameters().num_rows_in_block);
		String colsInBlockString = String.valueOf(this.getOutputParameters().num_cols_in_block);
		
		sb.append( "seq" );
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
		sb.append( output );
		sb.append(DATATYPE_PREFIX);
		sb.append(get_dataType());
		sb.append(VALUETYPE_PREFIX);
		sb.append(get_valueType());

		return sb.toString();
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String input6, String input7, String output) throws LopsException {
		
		if ( method != DataGenMethod.RAND )
			throw new LopsException("Invalid instruction generation for data generation method " + method);
		
		StringBuilder sb = new StringBuilder( );
		ExecType et = getExecType();
		sb.append( et );
		sb.append( Lop.OPERAND_DELIMITOR );
		String in1 = new String(""), in2 = new String("");
		
		if (this.getInputs().size() == RandStatement.RAND_VALID_PARAM_NAMES.length) {

			Lop iLop = null;

			iLop = _inputParams.get(RandStatement.RAND_ROWS.toString());
			String rowsString = iLop.getOutputParameters().getLabel();
			if ((iLop.getExecLocation() == ExecLocation.Data && !((Data) iLop)
					.isLiteral())
					|| !(iLop.getExecLocation() == ExecLocation.Data)) {
				in1 = "" + rowsString + DATATYPE_PREFIX + iLop.get_dataType()
						+ VALUETYPE_PREFIX + iLop.get_valueType();
				rowsString = Lop.VARIABLE_NAME_PLACEHOLDER + rowsString
						+ Lop.VARIABLE_NAME_PLACEHOLDER;
			}
			iLop = _inputParams.get(RandStatement.RAND_COLS.toString());
			String colsString = iLop.getOutputParameters().getLabel();
			if (iLop.getExecLocation() == ExecLocation.Data
					&& !((Data) iLop).isLiteral()
					|| !(iLop.getExecLocation() == ExecLocation.Data)) {
				in2 = "" + colsString + DATATYPE_PREFIX + iLop.get_dataType()
						+ VALUETYPE_PREFIX + iLop.get_valueType();
				colsString = Lop.VARIABLE_NAME_PLACEHOLDER + colsString
						+ Lop.VARIABLE_NAME_PLACEHOLDER;
			}
			String rowsInBlockString = String.valueOf(this
					.getOutputParameters().num_rows_in_block);
			String colsInBlockString = String.valueOf(this
					.getOutputParameters().num_cols_in_block);

			iLop = _inputParams.get(RandStatement.RAND_MIN.toString());
			String minString = iLop.getOutputParameters().getLabel();
			if (iLop.getExecLocation() == ExecLocation.Data
					&& !((Data) iLop).isLiteral()
					|| !(iLop.getExecLocation() == ExecLocation.Data))
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + RandStatement.RAND_MIN
						+ " must be a literal for a Rand operation.");

			iLop = _inputParams.get(RandStatement.RAND_MAX.toString());
			String maxString = iLop.getOutputParameters().getLabel();
			if (iLop.getExecLocation() == ExecLocation.Data
					&& !((Data) iLop).isLiteral()
					|| !(iLop.getExecLocation() == ExecLocation.Data))
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + RandStatement.RAND_MAX
						+ " must be a literal for a Rand operation.");

			iLop = _inputParams.get(RandStatement.RAND_SPARSITY.toString());
			String sparsityString = iLop.getOutputParameters().getLabel();
			if (iLop.getExecLocation() == ExecLocation.Data
					&& !((Data) iLop).isLiteral()
					|| !(iLop.getExecLocation() == ExecLocation.Data))
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + RandStatement.RAND_SPARSITY
						+ " must be a literal for a Rand operation.");

			iLop = _inputParams.get(RandStatement.RAND_SEED.toString());
			String seedString = iLop.getOutputParameters().getLabel();
			if (iLop.getExecLocation() == ExecLocation.Data
					&& !((Data) iLop).isLiteral()
					|| !(iLop.getExecLocation() == ExecLocation.Data))
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + RandStatement.RAND_SEED
						+ " must be a literal for a Rand operation.");

			iLop = _inputParams.get(RandStatement.RAND_PDF.toString());
			String pdfString = iLop.getOutputParameters().getLabel();
			if (iLop.getExecLocation() == ExecLocation.Data
					&& !((Data) iLop).isLiteral()
					|| !(iLop.getExecLocation() == ExecLocation.Data))
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + RandStatement.RAND_PDF
						+ " must be a literal for a Rand operation.");

			sb.append("Rand");
			sb.append(OPERAND_DELIMITOR);
			sb.append(in1);
			sb.append(OPERAND_DELIMITOR);
			sb.append(in2);
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
			sb.append(output);
			sb.append(DATATYPE_PREFIX);
			sb.append(get_dataType());
			sb.append(VALUETYPE_PREFIX);
			sb.append(get_valueType());

			return sb.toString();

		} else {
			throw new LopsException(this.printErrorLocation()
					+ "Invalid number of operands (" + this.getInputs().size()
					+ ") for a Rand operation");
		}
	}
	
	@Override
	public String getInstructions(int inputIndex, int outputIndex) throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		if ( method == DataGenMethod.SEQ ) {
			Lop iLop = null;

			iLop = _inputParams.get(RandStatement.SEQ_FROM.toString()); 
			String fromString = iLop.getOutputParameters().getLabel();
			if ( (iLop.getExecLocation() == ExecLocation.Data &&
					 !((Data)iLop).isLiteral()) || !(iLop.getExecLocation() == ExecLocation.Data ))
				fromString = Lop.VARIABLE_NAME_PLACEHOLDER + fromString + Lop.VARIABLE_NAME_PLACEHOLDER;
			
			iLop = _inputParams.get(RandStatement.SEQ_TO.toString()); 
			String toString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				toString = Lop.VARIABLE_NAME_PLACEHOLDER + toString + Lop.VARIABLE_NAME_PLACEHOLDER;
			
			iLop = _inputParams.get(RandStatement.SEQ_INCR.toString()); 
			String incrString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				incrString = Lop.VARIABLE_NAME_PLACEHOLDER + incrString + Lop.VARIABLE_NAME_PLACEHOLDER;
			
			String rowsString = String.valueOf(this.getOutputParameters().getNum_rows());
			String colsString = String.valueOf(this.getOutputParameters().getNum_cols());
			String rowsInBlockString = String.valueOf(this.getOutputParameters().num_rows_in_block);
			String colsInBlockString = String.valueOf(this.getOutputParameters().num_cols_in_block);
			
			sb.append( "seq" );
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
		if (this.getInputs().size() == RandStatement.RAND_VALID_PARAM_NAMES.length) {

			Lop iLop = null;

			iLop = _inputParams.get(RandStatement.RAND_ROWS.toString()); 
			String rowsString = iLop.getOutputParameters().getLabel();
			if ( (iLop.getExecLocation() == ExecLocation.Data &&
					 !((Data)iLop).isLiteral()) || !(iLop.getExecLocation() == ExecLocation.Data ))
				rowsString = Lop.VARIABLE_NAME_PLACEHOLDER + rowsString + Lop.VARIABLE_NAME_PLACEHOLDER;
			
			iLop = _inputParams.get(RandStatement.RAND_COLS.toString()); 
			String colsString = iLop.getOutputParameters().getLabel();
			if ( iLop.getExecLocation() == ExecLocation.Data 
					&& !((Data)iLop).isLiteral() || !(iLop.getExecLocation() == ExecLocation.Data ) )
				colsString = Lop.VARIABLE_NAME_PLACEHOLDER + colsString + Lop.VARIABLE_NAME_PLACEHOLDER;
			
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

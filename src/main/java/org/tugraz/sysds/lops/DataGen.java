/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.lops;

import java.util.HashMap;

import org.tugraz.sysds.hops.Hop.DataGenMethod;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.lops.OutputParameters.Format;
import org.tugraz.sysds.parser.DataExpression;
import org.tugraz.sysds.parser.DataIdentifier;
import org.tugraz.sysds.parser.Statement;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;


/**
 * <p>Defines a LOP that generates data.</p>
 */
public class DataGen extends Lop
{
	public static final String RAND_OPCODE   = "rand"; //rand
	public static final String SEQ_OPCODE    = "seq"; //sequence
	public static final String SINIT_OPCODE  = "sinit"; //string initialize
	public static final String SAMPLE_OPCODE = "sample"; //sample.int
	public static final String TIME_OPCODE = "time"; //time
	
	private int _numThreads = 1;
	
	/** base dir for rand input */
	private String baseDir;
	
	private HashMap<String, Lop> _inputParams;
	DataGenMethod method;
	
	/**
	 * <p>Creates a new Rand-LOP. The target identifier has to hold the dimensions of the new random object.</p>
	 * 
	 * @param mthd data gen method
	 * @param id target identifier
	 * @param inputParametersLops Lops of input parameters
	 * @param baseDir base dir for runtime
	 * @param dt Data type
	 * @param vt Value type
	 * @param et Execution type
	 */
	public DataGen(DataGenMethod mthd, DataIdentifier id, HashMap<String, Lop> 
		inputParametersLops, String baseDir, DataType dt, ValueType vt, ExecType et)
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
	
	public void init(DataIdentifier id, String baseDir, ExecType et) {
		getOutputParameters().setFormat(Format.BINARY);
		getOutputParameters().setBlocked(true);
		// TODO size for tensor
		getOutputParameters().setNumRows(id.getDim1());
		getOutputParameters().setNumCols(id.getDim2());
		getOutputParameters().setNnz(-1);
		getOutputParameters().setBlocksize(id.getBlocksize());
		lps.setProperties( inputs, et);
	}

	/**
	 * Function to generate CP/SP instructions for data generation via Rand and Seq.
	 * Since DataGen Lop can have many inputs, ONLY the output variable name is 
	 * passed from piggybacking as the function argument <code>output</code>. 
	 */
	@Override
	public String getInstructions(String output)
	{
		switch( method ) 
		{
			case RAND:
				return getRandInstructionCPSpark(output);
			case SINIT:
				return getSInitInstructionCPSpark(output);
			case SEQ:
				return getSeqInstructionCPSpark(output);
			case SAMPLE:
				return getSampleInstructionCPSpark(output);
			case TIME:
				return getTimeInstructionCP(output);
				
			default:
				throw new LopsException("Unknown data generation method: " + method);
		}
	}
	
	/**
	 * Private method that generates CP Instruction for Rand.
	 * 
	 * @param output output operand
	 * @return cp instruction for rand
	 */
	private String getRandInstructionCPSpark(String output) 
	{
		//sanity checks
		if ( method != DataGenMethod.RAND )
			throw new LopsException("Invalid instruction generation for data generation method " + method);
		if( getInputs().size() != DataExpression.RAND_VALID_PARAM_NAMES.length ) {
			throw new LopsException(printErrorLocation() + "Invalid number of operands (" 
				+ getInputs().size() + ") for a Rand operation");
		}
		
		StringBuilder sb = new StringBuilder();
		
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );

		sb.append(RAND_OPCODE);
		sb.append(OPERAND_DELIMITOR);
		
		Lop iLop = _inputParams.get(DataExpression.RAND_ROWS);
		sb.append(iLop.prepScalarInputOperand(getExecType()));
		sb.append(OPERAND_DELIMITOR);

		iLop = _inputParams.get(DataExpression.RAND_COLS);
		sb.append(iLop.prepScalarInputOperand(getExecType()));
		sb.append(OPERAND_DELIMITOR);

		iLop = _inputParams.get(DataExpression.RAND_DIMS);
		sb.append(iLop.prepScalarInputOperand(getExecType()));
		sb.append(OPERAND_DELIMITOR);

		sb.append(getOutputParameters().getBlocksize());
		sb.append(OPERAND_DELIMITOR);

		iLop = _inputParams.get(DataExpression.RAND_MIN);
		sb.append(iLop.prepScalarLabel());
		sb.append(OPERAND_DELIMITOR);
		
		iLop = _inputParams.get(DataExpression.RAND_MAX);
		sb.append(iLop.prepScalarLabel());
		sb.append(OPERAND_DELIMITOR);
		
		iLop = _inputParams.get(DataExpression.RAND_SPARSITY);
		if (iLop.isVariable())
			sb.append(iLop.prepScalarLabel());
		else
			sb.append(iLop.getOutputParameters().getLabel()); 
		sb.append(OPERAND_DELIMITOR);
		
		iLop = _inputParams.get(DataExpression.RAND_SEED);
		sb.append(iLop.prepScalarLabel());
		sb.append(OPERAND_DELIMITOR);
		
		if( getExecType() == ExecType.SPARK ) {
			sb.append(baseDir);
			sb.append(OPERAND_DELIMITOR);
		}
		
		iLop = _inputParams.get(DataExpression.RAND_PDF); //no variable support
		if (iLop.isVariable())
			throw new LopsException(printErrorLocation()
					+ "Parameter " + DataExpression.RAND_PDF
					+ " must be a literal for a Rand operation.");
		sb.append(iLop.getOutputParameters().getLabel()); 
		sb.append(OPERAND_DELIMITOR);
		
		iLop = _inputParams.get(DataExpression.RAND_LAMBDA); //no variable support
		sb.append(iLop == null ? "" : iLop.prepScalarLabel()); 
		sb.append(OPERAND_DELIMITOR);
		
		if( getExecType() == ExecType.CP ) {
			//append degree of parallelism
			sb.append( _numThreads );
			sb.append( OPERAND_DELIMITOR );
		}
		
		sb.append( prepOutputOperand(output));

		return sb.toString(); 
	}

	private String getSInitInstructionCPSpark(String output) 
	{
		if ( method != DataGenMethod.SINIT )
			throw new LopsException("Invalid instruction generation for data generation method " + method);
		
		//prepare instruction parameters
		Lop iLop = _inputParams.get(DataExpression.RAND_ROWS);
		String rowsString = iLop.prepScalarLabel();
		
		iLop = _inputParams.get(DataExpression.RAND_COLS);
		String colsString = iLop.prepScalarLabel();

		iLop = _inputParams.get(DataExpression.RAND_DIMS);
		String dimsString = iLop.prepScalarLabel();

		String blen = String.valueOf(getOutputParameters().getBlocksize());

		iLop = _inputParams.get(DataExpression.RAND_MIN);
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
		sb.append(dimsString);
		sb.append(OPERAND_DELIMITOR);
		sb.append(blen);
		sb.append(OPERAND_DELIMITOR);
		sb.append(minString);
		sb.append(OPERAND_DELIMITOR);
		sb.append(prepOutputOperand(output));

		return sb.toString();
	}
	
	private String getSampleInstructionCPSpark(String output) {
		if ( method != DataGenMethod.SAMPLE )
			throw new LopsException("Invalid instruction generation for data generation method " + method);
		
		//prepare instruction parameters
		Lop lsize = _inputParams.get(DataExpression.RAND_ROWS.toString());
		Lop lrange = _inputParams.get(DataExpression.RAND_MAX.toString());
		Lop lreplace = _inputParams.get(DataExpression.RAND_PDF.toString());
		Lop lseed = _inputParams.get(DataExpression.RAND_SEED.toString());
		
		return InstructionUtils.concatOperands(
			getExecType().name(),
			"sample",
			lrange.prepScalarLabel(),
			lsize.prepScalarInputOperand(getExecType()),
			lreplace.prepScalarLabel(),
			lseed.prepScalarLabel(),
			String.valueOf(getOutputParameters().getBlocksize()),
			prepOutputOperand(output));
	}
	
	private String getTimeInstructionCP(String output)
	{
		if (method != DataGenMethod.TIME )
			throw new LopsException("Invalid instruction generation for data generation method " + method);
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "time" );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output) );
		
		return sb.toString();
	}
	
	/**
	 * Private method that generates CP Instruction for Seq.
	 * 
	 * @param output output operand
	 * @return cp instruction for seq
	 */
	private String getSeqInstructionCPSpark(String output) {
		if ( method != DataGenMethod.SEQ )
			throw new LopsException("Invalid instruction generation for data generation method " + method);
		
		StringBuilder sb = new StringBuilder( );
		ExecType et = getExecType();
		sb.append( et );
		sb.append( Lop.OPERAND_DELIMITOR );

		Lop iLop = null;

		iLop = _inputParams.get(Statement.SEQ_FROM.toString());
		String fromString = iLop.prepScalarInputOperand(et);
		
		iLop = _inputParams.get(Statement.SEQ_TO.toString());
		String toString = iLop.prepScalarInputOperand(et);
		
		iLop = _inputParams.get(Statement.SEQ_INCR.toString());
		String incrString = iLop.prepScalarInputOperand(et);
		
		String rowsString = String.valueOf(this.getOutputParameters().getNumRows());
		String colsString = String.valueOf(this.getOutputParameters().getNumCols());
		String blen = String.valueOf(this.getOutputParameters().getBlocksize());
		
		sb.append( DataGen.SEQ_OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( rowsString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( colsString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( blen );
		sb.append( OPERAND_DELIMITOR );
		sb.append( fromString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( toString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( incrString );
		sb.append( OPERAND_DELIMITOR );
		sb.append(prepOutputOperand(output));

		return sb.toString();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(method.toString());
		sb.append(" ; num_rows=" + getOutputParameters().getNumRows());
		sb.append(" ; num_cols=" + getOutputParameters().getNumCols());
		sb.append(" ; nnz=" + getOutputParameters().getNnz());
		sb.append(" ; blocksize=" + getOutputParameters().getBlocksize());
		sb.append(" ; format=" + getOutputParameters().getFormat());
		sb.append(" ; blocked=" + getOutputParameters().isBlocked());
		sb.append(" ; dir=" + baseDir);
		return sb.toString();
	}
	
	public void setNumThreads(int k) {
		_numThreads = k;
	}
}

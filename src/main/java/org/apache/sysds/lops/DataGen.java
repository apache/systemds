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
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.ValueType;


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
	public static final String FRAME_OPCODE = "frame"; //time

	private int _numThreads = 1;
	
	/** base dir for rand input */
	private String baseDir;
	
	private HashMap<String, Lop> _inputParams;
	private final OpOpDG _op;
	
	/**
	 * <p>Creates a new Rand-LOP. The target identifier has to hold the dimensions of the new random object.</p>
	 * 
	 * @param op data gen method
	 * @param id target identifier
	 * @param inputParametersLops Lops of input parameters
	 * @param baseDir base dir for runtime
	 * @param dt Data type
	 * @param vt Value type
	 * @param et Execution type
	 */
	public DataGen(OpOpDG op, DataIdentifier id, HashMap<String, Lop> 
		inputParametersLops, String baseDir, DataType dt, ValueType vt, ExecType et)
	{
		super(Type.DataGen, dt, vt);
		_op = op;
		
		for (Lop lop : inputParametersLops.values()) {
			addInput(lop);
			lop.addOutput(this);
		}
		
		_inputParams = inputParametersLops;
		
		init(id, baseDir, et);
	}

	public OpOpDG getDataGenMethod() {
		return _op;
	}
	
	public void init(DataIdentifier id, String baseDir, ExecType et) {
		getOutputParameters().setFormat(FileFormat.BINARY);
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
	public String getInstructions(String output) {
		switch( _op ) {
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
			case FRAMEINIT:
				return getFrameInstructionCPSpark(output);
			default:
				throw new LopsException("Unknown data generation method: " + _op);
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
		if ( _op != OpOpDG.RAND )
			throw new LopsException("Invalid instruction generation for data generation method " + _op);
		if( getInputs().size() != DataExpression.RAND_VALID_PARAM_NAMES.size() - 2 && // tensor
				getInputs().size() != DataExpression.RAND_VALID_PARAM_NAMES.size() - 1 ) { // matrix
			throw new LopsException(printErrorLocation() + "Invalid number of operands (" 
				+ getInputs().size() + ") for a Rand operation");
		}
		
		StringBuilder sb = new StringBuilder();
		
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );

		sb.append(RAND_OPCODE);
		sb.append(OPERAND_DELIMITOR);
		
		Lop iLop = _inputParams.get(DataExpression.RAND_DIMS);
		if (iLop != null) {
			sb.append(iLop.prepScalarInputOperand(getExecType()));
			sb.append(OPERAND_DELIMITOR);
		}
		else {
			iLop = _inputParams.get(DataExpression.RAND_ROWS);
			sb.append(iLop.prepScalarInputOperand(getExecType()));
			sb.append(OPERAND_DELIMITOR);
			
			iLop = _inputParams.get(DataExpression.RAND_COLS);
			sb.append(iLop.prepScalarInputOperand(getExecType()));
			sb.append(OPERAND_DELIMITOR);
		}

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
		
		if( getExecType() == ExecType.CP || getExecType() == ExecType.OOC ) {
			//append degree of parallelism
			sb.append( _numThreads );
			sb.append( OPERAND_DELIMITOR );
		}
		
		sb.append( prepOutputOperand(output));

		return sb.toString(); 
	}

	private String getFrameInstructionCPSpark(String output)
	{
		//sanity checks
		if ( _op != OpOpDG.FRAMEINIT )
			throw new LopsException("Invalid instruction generation for data generation method " + _op);
		if( getInputs().size() != DataExpression.RAND_VALID_PARAM_NAMES.size() - 5  ) { // frame
			throw new LopsException(printErrorLocation() + "Invalid number of operands ("
				+ getInputs().size() + ") for a frame operation");
		}

		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );

		sb.append(FRAME_OPCODE);
		sb.append(OPERAND_DELIMITOR);

		Lop iLop = _inputParams.get(DataExpression.RAND_DATA);
		if ( iLop != null ) {
			if(iLop instanceof Nary) {
				for(Lop lop : iLop.getInputs()) {
					sb.append(((Data)lop).getStringValue());
					sb.append(DataExpression.DELIM_NA_STRING_SEP);
				}
			}
			else if(iLop instanceof Data) {
				sb.append(((Data)iLop).getStringValue());
			}
		}

		sb.append(OPERAND_DELIMITOR);

		iLop = _inputParams.get(DataExpression.RAND_DIMS);
		if (iLop != null) {
			sb.append(iLop.prepScalarInputOperand(getExecType()));
			sb.append(OPERAND_DELIMITOR);
		}
		else {
			iLop = _inputParams.get(DataExpression.RAND_ROWS);
			sb.append(iLop.prepScalarInputOperand(getExecType()));
			sb.append(OPERAND_DELIMITOR);

			iLop = _inputParams.get(DataExpression.RAND_COLS);
			sb.append(iLop.prepScalarInputOperand(getExecType()));
			sb.append(OPERAND_DELIMITOR);
		}
		iLop = _inputParams.get(DataExpression.SCHEMAPARAM);
		if ( iLop != null ) {
			if(iLop instanceof Nary) {
				for(Lop lop : iLop.getInputs()) {
					sb.append(((Data)lop).getStringValue());
					sb.append(DataExpression.DELIM_NA_STRING_SEP);
				}
			}
			else if(iLop instanceof Data) {
				sb.append(((Data)iLop).getStringValue());
			}
		}

		sb.append(OPERAND_DELIMITOR);

		if( getExecType() == ExecType.SPARK ) {
			sb.append(baseDir);
			sb.append(OPERAND_DELIMITOR);
		}

		sb.append( prepOutputOperand(output));
		return sb.toString();
	}
	
	private String getSInitInstructionCPSpark(String output) 
	{
		if ( _op != OpOpDG.SINIT )
			throw new LopsException("Invalid instruction generation for data generation method " + _op);
		
		//prepare instruction parameters
		Lop iLop = _inputParams.get(DataExpression.RAND_ROWS);
		String rowsString = iLop.prepScalarLabel();
		
		iLop = _inputParams.get(DataExpression.RAND_COLS);
		String colsString = iLop.prepScalarLabel();

		String blen = String.valueOf(getOutputParameters().getBlocksize());

		iLop = _inputParams.get(DataExpression.RAND_MIN);
		String minString = iLop.getOutputParameters().getLabel();
		if (iLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.RAND_MIN
					+ " must be a literal for a Rand operation.");

		return InstructionUtils.concatOperands(
			getExecType().toString(), SINIT_OPCODE,
			rowsString, colsString, blen, minString,
			prepOutputOperand(output));
	}
	
	private String getSampleInstructionCPSpark(String output) {
		if ( _op != OpOpDG.SAMPLE )
			throw new LopsException("Invalid instruction generation for data generation method " + _op);
		
		ExecType et = getExecType();
		return InstructionUtils.concatOperands(
			getExecType().name(), "sample",
			_inputParams.get(DataExpression.RAND_MAX.toString()).prepScalarLabel(),
			_inputParams.get(DataExpression.RAND_ROWS.toString()).prepScalarInputOperand(et),
			_inputParams.get(DataExpression.RAND_PDF.toString()).prepScalarLabel(),
			_inputParams.get(DataExpression.RAND_SEED.toString()).prepScalarLabel(),
			String.valueOf(getOutputParameters().getBlocksize()),
			prepOutputOperand(output));
	}
	
	private String getTimeInstructionCP(String output) {
		if (_op != OpOpDG.TIME )
			throw new LopsException("Invalid instruction generation for data generation method " + _op);
		
		return InstructionUtils.concatOperands(
			getExecType().toString(), "time",
			prepOutputOperand(output));
	}
	
	/**
	 * Private method that generates CP Instruction for Seq.
	 * 
	 * @param output output operand
	 * @return cp instruction for seq
	 */
	private String getSeqInstructionCPSpark(String output) {
		if ( _op != OpOpDG.SEQ )
			throw new LopsException("Invalid instruction generation for data generation method " + _op);
		
		ExecType et = getExecType();
		return InstructionUtils.concatOperands(
			et.toString(), Opcodes.SEQUENCE.toString(),
			String.valueOf(getOutputParameters().getNumRows()),
			String.valueOf(getOutputParameters().getNumCols()),
			String.valueOf(getOutputParameters().getBlocksize()),
			_inputParams.get(Statement.SEQ_FROM.toString()).prepScalarInputOperand(et),
			_inputParams.get(Statement.SEQ_TO.toString()).prepScalarInputOperand(et),
			_inputParams.get(Statement.SEQ_INCR.toString()).prepScalarInputOperand(et),
			prepOutputOperand(output));
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(_op.toString());
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

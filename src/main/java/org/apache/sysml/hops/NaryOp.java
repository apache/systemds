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

package org.apache.sysml.hops;

import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.Nary;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;

/**
 * The NaryOp Hop allows for a variable number of operands. Functionality
 * such as 'printf' (overloaded into the existing print function) is an example
 * of an operation that potentially takes a variable number of operands.
 *
 */
public class NaryOp extends Hop {
	protected OpOpN _op = null;

	protected NaryOp() {
	}

	/**
	 * NaryOp constructor.
	 * 
	 * @param name
	 *            the target name, typically set by the DMLTranslator when
	 *            constructing Hops. (For example, 'parsertemp1'.)
	 * @param dataType
	 *            the target data type (SCALAR for printf)
	 * @param valueType
	 *            the target value type (STRING for printf)
	 * @param multipleOperandOperation
	 *            the operation type (such as PRINTF)
	 * @param inputs
	 *            a variable number of input Hops
	 * @throws HopsException
	 *             thrown if a HopsException occurs
	 */
	public NaryOp(String name, DataType dataType, ValueType valueType,
			OpOpN op, Hop... inputs) throws HopsException {
		super(name, dataType, valueType);
		_op = op;

		for (int i = 0; i < inputs.length; i++) {
			getInput().add(i, inputs[i]);
			inputs[i].getParent().add(this);
		}
	}

	/** MultipleOp may have any number of inputs. */
	@Override
	public void checkArity() throws HopsException {}

	public OpOpN getOp() {
		return _op;
	}

	@Override
	public String getOpString() {
		return "m(" + _op.name().toLowerCase() + ")";
	}
	
	@Override
	public boolean isGPUEnabled() {
		return false;
	}

	/**
	 * Construct the corresponding Lops for this Hop
	 */
	@Override
	public Lop constructLops() throws HopsException, LopsException {
		// reuse existing lop
		if (getLops() != null)
			return getLops();

		try {
			Lop[] inLops = new Lop[getInput().size()];
			for (int i = 0; i < getInput().size(); i++)
				inLops[i] = getInput().get(i).constructLops();
			
			Nary.OperationType opType = HopsOpOpNLops.get(_op);
			if (opType == null)
				throw new HopsException("Unknown Nary Lop type for '"+_op+"'");
			
			ExecType et = optFindExecType();
			Nary multipleCPLop = new Nary(opType, getDataType(), getValueType(), inLops, et);
			setOutputDimensions(multipleCPLop);
			setLineNumbers(multipleCPLop);
			setLops(multipleCPLop);
		} 
		catch (Exception e) {
			throw new HopsException(this.printErrorLocation() + "error constructing Lops for NaryOp -- \n ", e);
		}

		// add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();

		return getLops();
	}
	
	@Override
	public boolean allowsAllExecTypes() {
		return false;
	}

	@Override
	protected double computeOutputMemEstimate(long dim1, long dim2, long nnz) {
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
	}

	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
		
		ExecType REMOTE = OptimizerUtils.isSparkExecutionMode() ? ExecType.SPARK : ExecType.MR;
		
		//forced / memory-based / threshold-based decision
		if( _etypeForced != null ) {
			_etype = _etypeForced;
		}
		else
		{
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) 
				_etype = findExecTypeByMemEstimate();
			// Choose CP, if the input dimensions are below threshold or if the input is a vector
			else if ( areDimsBelowThreshold() )
				_etype = ExecType.CP;
			else 
				_etype = REMOTE;
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
		
		//mark for recompile (forever)
		setRequiresRecompileIfNecessary();
		
		//ensure cp exec type for single-node operations
		if ( _op == OpOpN.PRINTF  || _op == OpOpN.EVAL)
			_etype = ExecType.CP;
		
		return _etype;
	}

	@Override
	protected double computeIntermediateMemEstimate(long dim1, long dim2, long nnz) {
		return 0;
	}

	@Override
	protected long[] inferOutputCharacteristics(MemoTable memo) {
		return null; //do nothing
	}
	
	@Override
	public void refreshSizeInformation() {
		switch( _op ) {
			case CBIND:
				setDim1(HopRewriteUtils.getMaxInputDim(this, true));
				setDim2(HopRewriteUtils.getSumValidInputDims(this, false));
				break;
			case RBIND:
				setDim1(HopRewriteUtils.getSumValidInputDims(this, true));
				setDim2(HopRewriteUtils.getMaxInputDim(this, false));
				break;
			case PRINTF:
			case EVAL:
				//do nothing:
		}
	}

	@Override
	public Object clone() throws CloneNotSupportedException {
		NaryOp multipleOp = new NaryOp();

		// copy generic attributes
		multipleOp.clone(this, false);

		// copy specific attributes
		multipleOp._op = _op;

		return multipleOp;
	}

	@Override
	public boolean compare(Hop that) {
		if (!(that instanceof NaryOp) || _op == OpOpN.PRINTF)
			return false;
		
		NaryOp that2 = (NaryOp) that;
		boolean ret = (_op == that2._op
			&& getInput().size() == that2.getInput().size());
		for( int i=0; i<getInput().size() && ret; i++ )
			ret &= (getInput().get(i) == that2.getInput().get(i));
		return ret;
	}
}

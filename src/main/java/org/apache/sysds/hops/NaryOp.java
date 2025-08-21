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

package org.apache.sysds.hops;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.Nary;
import org.apache.sysds.runtime.einsum.EinsumEquationValidator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

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
	 * @param op
	 *            the operation type (such as PRINTF)
	 * @param inputs
	 *            a variable number of input Hops
	 */
	public NaryOp(String name, DataType dataType, ValueType valueType,
			OpOpN op, Hop... inputs) {
		super(name, dataType, valueType);
		_op = op;
		for (int i = 0; i < inputs.length; i++) {
			getInput().add(i, inputs[i]);
			inputs[i].getParent().add(this);
		}
		refreshSizeInformation();
	}

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
	public Lop constructLops() {
		// reuse existing lop
		if (getLops() != null)
			return getLops();

		try {
			Lop[] inLops = new Lop[getInput().size()];
			for (int i = 0; i < getInput().size(); i++)
				inLops[i] = getInput().get(i).constructLops();
			
			ExecType et = optFindExecType();
			Nary multipleCPLop = new Nary(_op, getDataType(), getValueType(), inLops, et);
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
	public void computeMemEstimate(MemoTable memo) {
		//overwrites default hops behavior
		super.computeMemEstimate(memo);

		//specific case for function call
		if( _op == OpOpN.EVAL || _op == OpOpN.LIST ) {
			_memEstimate = OptimizerUtils.INT_SIZE;
			_outputMemEstimate = OptimizerUtils.INT_SIZE;
			_processingMemEstimate = 0;
		}
	}
	
	@Override
	protected double computeOutputMemEstimate(long dim1, long dim2, long nnz) {
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
	}

	@Override
	protected ExecType optFindExecType(boolean transitive) {
		
		checkAndSetForcedPlatform();
		
		ExecType REMOTE = ExecType.SPARK;
		
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
		if ( _op == OpOpN.PRINTF  || _op == OpOpN.EVAL || _op == OpOpN.LIST
			//TODO: cbind/rbind of lists only support in CP right now
			|| (_op == OpOpN.CBIND && getInput().get(0).getDataType().isList())
			|| (_op == OpOpN.RBIND && getInput().get(0).getDataType().isList())
			|| _op.isCellOp() && getInput().stream().allMatch(h -> h.getDataType().isScalar()))
			_etype = ExecType.CP;

		return _etype;
	}

	@Override
	protected double computeIntermediateMemEstimate(long dim1, long dim2, long nnz) {
		return 0;
	}

	@Override
	@SuppressWarnings("incomplete-switch")
	protected DataCharacteristics inferOutputCharacteristics(MemoTable memo) {
		if( !getDataType().isScalar() && !getDataType().isFrame()) {
			DataCharacteristics[] dc = memo.getAllInputStats(getInput());
			
			switch( _op ) {
				case CBIND: return new MatrixCharacteristics(
					HopRewriteUtils.getMaxInputDim(dc, true),
					HopRewriteUtils.getSumValidInputDims(dc, false), -1,
					HopRewriteUtils.getSumValidInputNnz(dc, true));
				case RBIND: return new MatrixCharacteristics(
					HopRewriteUtils.getSumValidInputDims(dc, true),
					HopRewriteUtils.getMaxInputDim(dc, false), -1,
					HopRewriteUtils.getSumValidInputNnz(dc, true));
				case MIN:
				case MAX:
				case PLUS:
				case MULT: return new MatrixCharacteristics(
						HopRewriteUtils.getMaxInputDim(this, true),
						HopRewriteUtils.getMaxInputDim(this, false), -1, -1);
				case LIST:
					return new MatrixCharacteristics(getInput().size(), 1, -1, -1);
			}
		}
		return null; //do nothing
	}

	@Override
	public void refreshSizeInformation() {
		switch( _op ) {
			case CBIND:
				if( !getInput().get(0).getDataType().isList() ) {
					setDim1(HopRewriteUtils.getMaxInputDim(this, true));
					setDim2(HopRewriteUtils.getSumValidInputDims(this, false));
					setNnz(HopRewriteUtils.getSumValidInputNnz(this));
				}
				break;
			case RBIND:
				if( !getInput().get(0).getDataType().isList() ) {
					setDim1(HopRewriteUtils.getSumValidInputDims(this, true));
					setDim2(HopRewriteUtils.getMaxInputDim(this, false));
					setNnz(HopRewriteUtils.getSumValidInputNnz(this));
				}
				break;
			case MIN:
			case MAX:
			case PLUS:
			case MULT:
				setDim1(getDataType().isScalar() ? 0 : HopRewriteUtils.getMaxInputDim(this, true));
				setDim2(getDataType().isScalar() ? 0 : HopRewriteUtils.getMaxInputDim(this, false));
				break;
			case LIST:
				setDim1(getInput().size());
				setDim2(1);
				break;
			case EINSUM:
				String equationString = ((LiteralOp) _input.get(0)).getStringValue();
				var dims = EinsumEquationValidator.validateEinsumEquationAndReturnDimensions(equationString, this.getInput().subList(1, this.getInput().size()));

				setDim1(dims.getLeft());
				setDim2(dims.getMiddle());
				setDataType(dims.getRight());
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

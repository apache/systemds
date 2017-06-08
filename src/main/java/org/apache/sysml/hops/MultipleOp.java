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

import java.util.ArrayList;

import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.MultipleCP;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;

/**
 * The MultipleOp Hop allows for a variable number of operands. Functionality
 * such as 'printf' (overloaded into the existing print function) is an example
 * of an operation that potentially takes a variable number of operands.
 *
 */
public class MultipleOp extends Hop {
	protected MultiInputOp multipleOperandOperation = null;

	protected MultipleOp() {
	}

	/**
	 * MultipleOp constructor.
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
	public MultipleOp(String name, DataType dataType, ValueType valueType,
			MultiInputOp multipleOperandOperation, Hop... inputs) throws HopsException {
		super(name, dataType, valueType);
		this.multipleOperandOperation = multipleOperandOperation;

		for (int i = 0; i < inputs.length; i++) {
			getInput().add(i, inputs[i]);
			inputs[i].getParent().add(this);
		}

		// compute unknown dims and nnz
		refreshSizeInformation();
	}

	/** MultipleOp may have any number of inputs. */
	@Override
	public void checkArity() throws HopsException {}

	public MultiInputOp getOp() {
		return multipleOperandOperation;
	}

	@Override
	public String getOpString() {
		return "m(" + multipleOperandOperation.toString().toLowerCase() + ")";
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
			ArrayList<Hop> inHops = getInput();
			Lop[] inLops = new Lop[inHops.size()];
			for (int i = 0; i < inHops.size(); i++) {
				Hop inHop = inHops.get(i);
				Lop inLop = inHop.constructLops();
				inLops[i] = inLop;
			}

			MultipleCP.OperationType opType = MultipleOperandOperationHopTypeToLopType.get(multipleOperandOperation);
			if (opType == null) {
				throw new HopsException("Unknown MultipleCP Lop operation type for MultipleOperandOperation Hop type '"
						+ multipleOperandOperation + "'");
			}

			MultipleCP multipleCPLop = new MultipleCP(opType, getDataType(), getValueType(), inLops);
			setOutputDimensions(multipleCPLop);
			setLineNumbers(multipleCPLop);
			setLops(multipleCPLop);
		} catch (Exception e) {
			throw new HopsException(this.printErrorLocation() + "error constructing Lops for MultipleOp Hop -- \n ", e);
		}

		// add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();

		return getLops();
	}

	@Override
	protected double computeOutputMemEstimate(long dim1, long dim2, long nnz) {
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
	}

	@Override
	public boolean allowsAllExecTypes() {
		return false; // true?
	}

	@Override
	protected ExecType optFindExecType() throws HopsException {
		checkAndSetForcedPlatform(); // ?
		return ExecType.CP;
	}

	@Override
	public void refreshSizeInformation() {
		// do nothing
	}

	@Override
	public Object clone() throws CloneNotSupportedException {
		MultipleOp multipleOp = new MultipleOp();

		// copy generic attributes
		multipleOp.clone(this, false);

		// copy specific attributes
		multipleOp.multipleOperandOperation = multipleOperandOperation;

		return multipleOp;
	}

	@Override
	public boolean compare(Hop that) {
		if (!(that instanceof MultipleOp))
			return false;

		if (multipleOperandOperation == MultiInputOp.PRINTF) {
			return false;
		}

		// if add new multiple operand types in addition to PRINTF,
		// probably need to modify this.
		MultipleOp mo = (MultipleOp) that;
		return (multipleOperandOperation == mo.multipleOperandOperation);
	}

	@Override
	protected double computeIntermediateMemEstimate(long dim1, long dim2, long nnz) {
		return 0;
	}

	@Override
	protected long[] inferOutputCharacteristics(MemoTable memo) {
		return null;
	}
}

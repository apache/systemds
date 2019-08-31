/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.runtime.data;

import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheBlock;
import org.tugraz.sysds.runtime.matrix.operators.AggregateOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.BinaryOperator;
import org.tugraz.sysds.runtime.util.UtilFunctions;

public abstract class TensorBlock implements CacheBlock
{
	public static final int[] DEFAULT_DIMS = new int[]{0, 0};

	//min 2 dimensions to preserve proper matrix semantics
	protected int[] _dims; //[2,inf)

	public abstract void reset();

	public abstract void reset(int[] dims);

	public abstract boolean isAllocated();

	public abstract TensorBlock allocateBlock();

	public int getNumDims() {
		return _dims.length;
	}

	public int getNumRows() {
		return getDim(0);
	}

	public int getNumColumns() {
		return getDim(1);
	}

	public int getDim(int i) {
		return _dims[i];
	}

	/**
	 * Calculates the next index array. Note that if the given index array was the last element, the next index will
	 * be the first one.
	 *
	 * @param ix the index array which will be incremented to the next index array
	 */
	public void getNextIndexes(int[] ix) {
		int i = ix.length - 1;
		ix[i]++;
		//calculating next index
		if (ix[i] == getDim(i)) {
			while (ix[i] == getDim(i)) {
				ix[i] = 0;
				i--;
				if (i < 0) {
					//we are finished
					break;
				}
				ix[i]++;
			}
		}
	}

	public boolean isVector() {
		return getNumDims() <= 2
				&& (getDim(0) == 1 || getDim(1) == 1);
	}

	public boolean isMatrix() {
		return getNumDims() == 2
				&& (getDim(0) > 1 && getDim(1) > 1);
	}

	public long getLength() {
		return UtilFunctions.prod(_dims);
	}

	public boolean isEmpty() {
		return isEmpty(false);
	}

	public abstract boolean isEmpty(boolean safe);

	public abstract long getNonZeros();

	public abstract Object get(int[] ix);

	public abstract double get(int r, int c);

	/**
	 * Set a cell to the value given as an `Object`. The type is inferred by either the `schema` or `valueType`, depending
	 * if the `TensorBlock` is a `BasicTensor` or `DataTensor`.
	 * @param ix indexes in each dimension, starting with 0
	 * @param v value to set
	 */
	public abstract void set(int[] ix, Object v);

	/**
	 * Set a cell in a 2-dimensional tensor.
	 * @param r row of the cell
	 * @param c column of the cell
	 * @param v value to set
	 */
	public abstract void set(int r, int c, double v);

	/**
	 * Aggregate a unary operation on this tensor.
	 * @param op the operation to apply
	 * @param result the result tensor
	 * @return the result tensor
	 */
	public abstract TensorBlock aggregateUnaryOperations(AggregateUnaryOperator op, TensorBlock result);

	public abstract void incrementalAggregate(AggregateOperator aggOp, TensorBlock partialResult);

	public abstract TensorBlock binaryOperations(BinaryOperator op, TensorBlock thatValue, TensorBlock result);

	protected abstract TensorBlock checkType(TensorBlock that);

	public static ValueType resultValueType(ValueType in1, ValueType in2) {
		// TODO reconsider with operation types
		if (in1 == ValueType.UNKNOWN || in2 == ValueType.UNKNOWN)
			throw new DMLRuntimeException("Operations on unknown value types not possible");
		else if (in1 == ValueType.STRING || in2 == ValueType.STRING)
			return ValueType.STRING;
		else if (in1 == ValueType.FP64 || in2 == ValueType.FP64)
			return ValueType.FP64;
		else if (in1 == ValueType.FP32 || in2 == ValueType.FP32)
			return ValueType.FP32;
		else if (in1 == ValueType.INT64 || in2 == ValueType.INT64)
			return ValueType.INT64;
		else if (in1 == ValueType.INT32 || in2 == ValueType.INT32)
			return ValueType.INT32;
		else // Boolean - Boolean
			return ValueType.INT64;
	}
}

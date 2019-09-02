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

package org.tugraz.sysds.runtime.meta;

import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.ReorgOperator;

import java.io.Serializable;

public abstract class DataCharacteristics implements Serializable {
	private static final long serialVersionUID = 3411056029517599342L;

	protected int _blocksize;
	
	public DataCharacteristics set(long nr, long nc, int len) {
		throw new DMLRuntimeException("DataCharacteristics.set(long, long, int): should never get called in the base class");
	}

	public DataCharacteristics set(long nr, long nc, int blen, long nnz) {
		throw new DMLRuntimeException("DataCharacteristics.set(long, long, int, long): should never get called in the base class");
	}

	public DataCharacteristics set(long[] dims, int blocksize) {
		throw new DMLRuntimeException("DataCharacteristics.set(long[], int): should never get called in the base class");
	}

	public DataCharacteristics set(long[] dims, int blocksize, long nnz) {
		throw new DMLRuntimeException("DataCharacteristics.set(long[], int, long): should never get called in the base class");
	}

	public DataCharacteristics set(DataCharacteristics that) {
		throw new DMLRuntimeException("DataCharacteristics.set(DataCharacteristics): should never get called in the base class");
	}

	public long getRows() {
		throw new DMLRuntimeException("DataCharacteristics.getRows(): should never get called in the base class");
	}

	public void setRows(long rlen) {
		throw new DMLRuntimeException("DataCharacteristics.setRows(long): should never get called in the base class");
	}

	public long getCols() {
		throw new DMLRuntimeException("DataCharacteristics.getCols(): should never get called in the base class");
	}

	public void setCols(long clen) {
		throw new DMLRuntimeException("DataCharacteristics.setCols(long): should never get called in the base class");
	}

	public long getLength() {
		throw new DMLRuntimeException("DataCharacteristics.getLength(): should never get called in the base class");
	}

	public int getBlocksize() {
		return _blocksize;
	}

	public DataCharacteristics setBlocksize(int blen){
		_blocksize = blen;
		return this;
	}

	public long getNumBlocks() {
		throw new DMLRuntimeException("DataCharacteristics.getNumBlocks(int): should never get called in the base class");
	}

	public long getNumRowBlocks() {
		throw new DMLRuntimeException("DataCharacteristics.getNumRowBlocks(): should never get called in the base class");
	}

	public long getNumColBlocks() {
		throw new DMLRuntimeException("DataCharacteristics.getNumColBlocks(): should never get called in the base class");
	}

	public void setDimension(long nr, long nc) {
		throw new DMLRuntimeException("DataCharacteristics.setDimension(long, long): should never get called in the base class");
	}

	public int getNumDims() {
		throw new DMLRuntimeException("DataCharacteristics.getNumDims(): should never get called in the base class");
	}

	public long getDim(int i) {
		throw new DMLRuntimeException("DataCharacteristics.getDim(int): should never get called in the base class");
	}

	public long[] getDims() {
		throw new DMLRuntimeException("DataCharacteristics.getDims(): should never get called in the base class");
	}

	public TensorCharacteristics setDim(int i, long dim) {
		throw new DMLRuntimeException("DataCharacteristics.setDim(int, long): should never get called in the base class");
	}

	public TensorCharacteristics setDims(long[] dims) {
		throw new DMLRuntimeException("DataCharacteristics.setDims(long[]): should never get called in the base class");
	}

	public long getNumBlocks(int i) {
		throw new DMLRuntimeException("DataCharacteristics.getNumBlocks(i): should never get called in the base class");
	}

	public void setNonZeros(long nnz) {
		throw new DMLRuntimeException("DataCharacteristics.setNonZeros(long): should never get called in the base class");
	}

	public long getNonZeros() {
		throw new DMLRuntimeException("DataCharacteristics.getNonZeros(): should never get called in the base class");
	}

	public void setNonZerosBound(long nnz) {
		throw new DMLRuntimeException("DataCharacteristics.setNonZerosBound(long): should never get called in the base class");
	}

	public long getNonZerosBound() {
		throw new DMLRuntimeException("DataCharacteristics.getNonZerosBound(): should never get called in the base class");
	}

	public double getSparsity() {
		throw new DMLRuntimeException("DataCharacteristics.getSparsity(): should never get called in the base class");
	}

	public boolean dimsKnown() {
		throw new DMLRuntimeException("DataCharacteristics.dimsKnown(): should never get called in the base class");
	}

	public boolean dimsKnown(boolean includeNnz) {
		throw new DMLRuntimeException("DataCharacteristics.dimsKnown(boolean): should never get called in the base class");
	}

	public boolean rowsKnown() {
		throw new DMLRuntimeException("DataCharacteristics.rowsKnown(): should never get called in the base class");
	}

	public boolean colsKnown() {
		throw new DMLRuntimeException("DataCharacteristics.colsKnown(): should never get called in the base class");
	}

	public boolean nnzKnown() {
		throw new DMLRuntimeException("DataCharacteristics.nnzKnown(): should never get called in the base class");
	}

	public boolean isUltraSparse() {
		throw new DMLRuntimeException("DataCharacteristics.isUltraSparse(): should never get called in the base class");
	}

	public boolean mightHaveEmptyBlocks() {
		throw new DMLRuntimeException("DataCharacteristics.mightHaveEmptyBlocks(): should never get called in the base class");
	}

	public static void reorg(DataCharacteristics dim, ReorgOperator op, DataCharacteristics dimOut) {
		op.fn.computeDimension(dim, dimOut);
	}

	public static void aggregateUnary(DataCharacteristics dim, AggregateUnaryOperator op, DataCharacteristics dimOut) {
		op.indexFn.computeDimension(dim, dimOut);
	}

	public static void aggregateBinary(DataCharacteristics dim1, DataCharacteristics dim2, AggregateBinaryOperator op, DataCharacteristics dimOut) {
		dimOut.set(dim1.getRows(), dim2.getCols(), dim1.getBlocksize());
	}

	@Override
	public abstract boolean equals(Object anObject);

	@Override
	public abstract int hashCode();
}

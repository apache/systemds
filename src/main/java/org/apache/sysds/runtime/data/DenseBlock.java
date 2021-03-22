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

package org.apache.sysds.runtime.data;

import java.io.Serializable;
import java.util.Arrays;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * This DenseBlock is an abstraction for different dense, row-major 
 * matrix formats. For efficient dense operations, this API does not
 * expose a row but a row-block abstraction, where a block can contain
 * one or many contiguous rows.
 * 
 */
public abstract class DenseBlock implements Serializable
{
	private static final long serialVersionUID = 7517220490270237832L;

	public enum Type {
		DRB, //dense row block
		LDRB, //large dense row block
	}
	
	//NOTE: for a MxNxPxQ tensor the dimensions are given as
	//rlen=M, odims=[NxPxQ, PxQ, Q]
	protected int _rlen;  //number of rows
	protected int[] _odims; //cumprod other dims
	private double[] _reuse;
	
	protected DenseBlock(int[] dims) {
		setDims(dims);
	}

	/**
	 * Create a block in the internal blocks array. `allocateBlocks` has to be called
	 * before it for Large-Dense-Blocks.
	 *
	 * @param bix       block index
	 * @param length    space to allocate
	 */
	protected abstract void allocateBlock(int bix, int length);

	/**
	 * Get the ith dimensions size of the dense block.
	 *
	 * @param i the number of dimension to get
	 * @return the size of the dimension
	 */
	public final int getDim(int i) {
		return (i == 0) ?_rlen : 
			(i == _odims.length) ? _odims[i - 1] :
			_odims[i - 1] / _odims[i];
	}

	/**
	 * Get the ith cumulative dimensions size of the dense block, without row.
	 *
	 * @param i the number of the cumulative dimension to get (0 equals the second dimension!)
	 * @return the size of the dimension cumulative with all following dimensions
	 */
	public final int getCumODims(int i) {
		return _odims[i];
	}

	/**
	 * Resets the dense block by deleting non-zero values. After this
	 * call all countNonZeros() calls are guaranteed to return 0.
	 */
	public final void reset() {
		reset(_rlen, _odims, 0);
	}
	
	/**
	 * Resets the dense block by deleting non-zero values. After this
	 * call all countNonZeros() calls are guaranteed to return 0. If
	 * the new dimensions exceed the current capacity, the underlying
	 * storage is extended accordingly.
	 * 
	 * @param dims length and size of dimensions.
	 */
	public final void reset(int[] dims) {
		reset(dims[0], createDimOffsets(dims), 0);
	}
	
	/**
	 * Resets the dense block by deleting non-zeros.
	 * 
	 * @param dims lenth and size of dimensions
	 * @param v value
	 */
	public final void reset(int[] dims, double v) {
		reset(dims[0], createDimOffsets(dims), v);
	}
	
	/**
	 * Resets the dense block by deleting non-zeros.
	 * 
	 * @param rlen number of rows
	 * @param clen number of columns
	 */
	public final void reset(int rlen, int clen) {
		reset(rlen, new int[]{clen}, 0);
	}
	
	/**
	 * Resets the dense block by deleting non-zeros.
	 * 
	 * @param rlen number of rows
	 * @param odims offsets of other dimensions
	 */
	public final void reset(int rlen, int[] odims) {
		reset(rlen, odims, 0);
	}
	
	/**
	 * Resets the dense block by setting the given value.
	 * 
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @param v value
	 */
	public final void reset(int rlen, int clen, double v) {
		reset(rlen, new int[]{clen}, v);
	}
	
	
	/**
	 * Resets the dense block by setting the given value.
	 * 
	 * @param rlen number of rows
	 * @param odims other dimensions
	 * @param v value
	 */
	public abstract void reset(int rlen, int[] odims, double v);
	
	
	public static double estimateMemory(long nrows, long ncols){
		long size = 16; // object
		size += 4; // int
		size += 4; // padding
		size += MemoryEstimates.intArrayCost(1); // odims typically 1
		size += 8; // pointer to reuse that is typically null;
		return size;
	}

	/**
	 * Set the dimensions of the dense MatrixBlock.
	 * @param dims The dimensions to set, first dimension is rows, second cols.
	 */
	public void setDims(int[] dims){
		long odims = UtilFunctions.prod(dims, 1);
		if( odims > Integer.MAX_VALUE )
			throw new DMLRuntimeException("Invalid dims: "+Arrays.toString(dims));
		_rlen = dims[0];
		//materialize dim offsets (reverse cumprod)
		_odims = createDimOffsets(dims);
	}

	/**
	 * Get the number of rows.
	 * 
	 * @return number of rows
	 */
	public final int numRows() {
		return _rlen;
	}
	
	/**
	 * Get the number of dimensions.
	 * 
	 * @return number of dimensions, min 2
	 */
	public final int numDims() {
		return 1 + _odims.length;
	}
	
	/**
	 * Get the number of allocated blocks.
	 * 
	 * @return number of blocks
	 */
	public abstract int numBlocks();
	
	/**
	 * Get the number of rows per block, except last one.
	 * 
	 * @return number of rows in block
	 */
	public abstract int blockSize();
	
	/**
	 * Get the number of rows of the given block.
	 * 
	 * @param bix block index
	 * @return number of rows in block
	 */
	public abstract int blockSize(int bix);
	
	/**
	 * Indicates if the dense block is numeric.
	 * @return true if numeric (FP, INT, BOOLEAN)
	 */
	public abstract boolean isNumeric();
	
	/**
	 * Indicates if the dense block has a single
	 * underlying block, i.e., if numBlocks==1.
	 * 
	 * @return true if single block
	 */
	public abstract boolean isContiguous();
	
	/**
	 * Indicates if the dense block has a single
	 * underlying block for the given row range.
	 * 
	 * @param rl row lower index
	 * @param ru row upper index (inclusive)
	 * @return true if single block in row range
	 */
	public abstract boolean isContiguous(int rl, int ru);
	
	/**
	 * Get the length of the dense block as the product
	 * of all dimensions.
	 * 
	 * @return length
	 */
	public final long size() {
		return (long)_rlen * _odims[0];
	}
	
	/**
	 * Get the length of the given block.
	 * 
	 * @param bix block index
	 * @return length
	 */
	public abstract int size(int bix);
	
	
	/**
	 * Get the total length of allocated blocks.
	 * 
	 * @return capacity
	 */
	public abstract long capacity();


	/**
	 * Computes the number of non zero elements of a certain range of elements in a block.
	 *
	 * @param bix       index of block
	 * @param start     start index in block
	 * @param length    number of elements to check
	 * @return          number of elements that are not zero
	 */
	protected abstract long computeNnz(int bix, int start, int length);

	/**
	 * Compute the number of non-zero values, which potentially 
	 * makes a full pass over the underlying blocks.
	 * 
	 * @return number of non-zeros
	 */
	public abstract long countNonZeros();
	
	/**
	 * Compute the number of non-zero values for the given row,
	 * which potentially makes a full pass over the underlying row.
	 * 
	 * @param r row index
	 * @return number of non-zeros
	 */
	public abstract int countNonZeros(int r);
	
	/**
	 * Compute the number of non-zero values, which potentially 
	 * makes a full pass over the underlying blocks in the row range.
	 * 
	 * @param rl row lower index
	 * @param ru row upper index (exclusive)
	 * @param cl column lower index 
	 * @param cu column upper index (exclusive)
	 * @return number of non-zeros
	 */
	public abstract long countNonZeros(int rl, int ru, int cl, int cu);

	/**
	 * Get the allocated block for the given row. This call
	 * is equivalent to valuesAt(indexes(r)).
	 * 
	 * @param r row index
	 * @return block
	 */
	public abstract double[] values(int r);
	
	
	/**
	 * Get an allocated block.
	 * 
	 * @param bix block index
	 * @return block
	 */
	public abstract double[] valuesAt(int bix);
	
	/**
	 * Get the block index for a given row.
	 * 
	 * @param r row index
	 * @return block index
	 */
	public abstract int index(int r);
	
	/**
	 * Get the position for a given row within
	 * its associated block.
	 * 
	 * @param r row index
	 * @return block position
	 */
	public abstract int pos(int r);
	
	/**
	 * Get the position for a given row and column
	 * within the associated block.
	 * 
	 * @param r row index
	 * @param c column index
	 * @return block position
	 */
	public abstract int pos(int r, int c);
	
	/**
	 * Get the position for a given cell
	 * within the associated block.
	 * 
	 * @param ix cell indexes
	 * @return block position
	 */
	public abstract int pos(int[] ix);
	
	
	/**
	 * Increments the given value for a given row and column.
	 * 
	 * @param r row index
	 * @param c column index
	 */
	public abstract void incr(int r, int c);
	
	/**
	 * Increments the given value for a given row and column
	 * by delta.
	 * 
	 * @param r row index
	 * @param c column index
	 * @param delta increment value
	 */
	public abstract void incr(int r, int c, double delta);

	/**
	 * Fill a certain range of elements of a block.
	 *
	 * @param bix       index of block
	 * @param fromIndex starting index in block
	 * @param toIndex   ending index in block (exclusive)
	 * @param v         value
	 */
	protected abstract void fillBlock(int bix, int fromIndex, int toIndex, double v);

	/**
	 * Set a value at a position given by block index and index in that block.
	 * @param bix   block index
	 * @param ix    block-array index
	 * @param v     value
	 */
	protected abstract void setInternal(int bix, int ix, double v);

	/**
	 * Set the given value for the entire dense block (fill).
	 * 
	 * @param v value
	 * @return self
	 */
	public abstract DenseBlock set(double v);

	/**
	 * Set the given string for the entire dense block (fill). Generally the string will be parsed, except for string
	 * DenseBlock.
	 *
	 * @param s string
	 * @return self
	 */
	public DenseBlock set(String s) {
		set(Double.parseDouble(s));
		return this;
	}

	/**
	 * Set the given value for an entire index range of the 
	 * dense block (fill).
	 * 
	 * @param rl row lower index
	 * @param ru row upper index (exclusive)
	 * @param cl column lower index 
	 * @param cu column upper index (exclusive)
	 * @param v value
	 * @return self
	 */
	public abstract DenseBlock set(int rl, int ru, int cl, int cu, double v);
	
	
	/**
	 * Set the given value for a given row and column.
	 * 
	 * @param r row index
	 * @param c column index
	 * @param v value
	 * @return self
	 */
	public abstract DenseBlock set(int r, int c, double v);

	/**
	 * Copy the given vector into the given row.
	 * 
	 * @param r row index
	 * @param v value vector
	 * @return self
	 */
	public abstract DenseBlock set(int r, double[] v);
	
	/**
	 * Copy the given dense block.
	 * 
	 * @param db dense block
	 * @return self
	 */
	public abstract DenseBlock set(DenseBlock db);
	
	/**
	 * Copy the given dense block into the specified
	 * index range.
	 * 
	 * @param rl row lower index
	 * @param ru row upper index (exclusive)
	 * @param cl column lower index 
	 * @param cu column upper index (exclusive)
	 * @param db dense block
	 * @return self
	 */
	public DenseBlock set(int rl, int ru, int cl, int cu, DenseBlock db) {
		// TODO: Optimize if dense block types match
		// TODO: Performance
		boolean allColumns = cl == 0 && cu == _odims[0];
		if (db.isNumeric()) {
			int rowOther = 0;
			int colOther = 0;
			for (int bi = index(rl); bi <= index(ru-1); bi++) {
				int rpos = bi*blockSize();
				int rposl = Math.max(rl-rpos, 0);
				if (allColumns) {
					int offset = rposl * _odims[0] + cl;
					int rlen = Math.min(ru-Math.max(rpos,rl), blockSize(bi)-rposl);
					for (int i = 0; i < rlen * _odims[0]; i++) {
						setInternal(bi, offset + i, db.get(rowOther, colOther));
						colOther++;
						if (colOther == db.getCumODims(0)) {
							rowOther++;
							colOther = 0;
						}
					}
				}
				else {
					int len = cu - cl;
					for (int i = rl, ix1 = rl * _odims[0] + cl; i < ru; i++, ix1 += _odims[0]) {
						for (int ix = 0; ix < len; ix++) {
							setInternal(bi, ix1 + ix, db.get(rowOther, colOther));
							colOther++;
							if (colOther == db.getCumODims(0)) {
								rowOther++;
								colOther = 0;
							}
						}
					}
				}
				rl = 0;
			}
		}
		else {
			int[] otherIx = new int[db.numDims()];
			for (int bi = index(rl); bi <= index(ru - 1); bi++) {
				String[] data;
				if (this instanceof DenseBlockString) {
					data = ((DenseBlockString) this)._data;
				} else {
					data = ((DenseBlockLString) this)._blocks[bi];
				}
				if (allColumns) {
					int offset = rl * _odims[0] + cl;
					for (int i = 0; i < (ru - rl) * _odims[0]; i++) {
						data[offset + i] = db.getString(otherIx);
						getNextIndexes(otherIx);
					}
				}
				else {
					int len = cu - cl;
					for (int i = rl, ix1 = rl * _odims[0] + cl; i < ru; i++, ix1 += _odims[0]) {
						for (int ix = 0; ix < len; ix++) {
							data[ix1 + ix] = db.getString(otherIx);
							getNextIndexes(otherIx);
						}
						otherIx[0] = i - rl + 1;
						otherIx[1] = 0;
						Arrays.fill(otherIx, 2, otherIx.length, 0);
					}
				}
				rl = 0;
			}
		}
		return this;
	}

	/**
	 * Calculates the next index array. Note that if the given index array was the last element, the next index will
	 * be outside of range.
	 *
	 * @param ix the index array which will be incremented to the next index array
	 */
	public void getNextIndexes(int[] ix) {
		int i = ix.length - 1;
		ix[i]++;
		//calculating next index
		if (ix[i] == getDim(i)) {
			while (ix[i] == getDim(i)) {
				if (i - 1 < 0) {
					//we are finished
					break;
				}
				ix[i] = 0;
				i--;
				ix[i]++;
			}
		}
	}

	/**
	 * Copy the given kahan object sum and correction.
	 * 
	 * @param kbuff kahan object
	 * @return self
	 */
	public DenseBlock set(KahanObject kbuff) {
		set(0, 0, kbuff._sum);
		set(0, 1, kbuff._correction);
		return this;
	}
	
	/**
	 * Set the specified cell to the given value.
	 * 
	 * @param ix cell indexes
	 * @param v value
	 * @return self
	 */
	public abstract DenseBlock set(int[] ix, double v);

	/**
	 * Set the specified cell to the given value.
	 *
	 * @param ix cell indexes
	 * @param v value
	 * @return self
	 */
	public abstract DenseBlock set(int[] ix, long v);

	/**
	 * Set the specified cell to the given value.
	 *
	 * @param ix cell indexes
	 * @param v value as String
	 * @return self
	 */
	public abstract DenseBlock set(int[] ix, String v);

	/**
	 * Copy the given kahan object sum and correction
	 * into the given row.
	 * 
	 * @param r row index
	 * @param kbuff kahan object
	 * @return self
	 */
	public DenseBlock set(int r, KahanObject kbuff) {
		set(r, 0, kbuff._sum);
		set(r, 1, kbuff._correction);
		return this;
	}
	
	/**
	 * Get the value for a given row and column.
	 * 
	 * @param r row index
	 * @param c column index
	 * @return value
	 */
	public abstract double get(int r, int c);
	
	/**
	 * Get the value of a given cell
	 * 
	 * @param ix cell indexes
	 * @return value
	 */
	public abstract double get(int[] ix);

	/**
	 * Get the value of a given cell as a String
	 *
	 * @param ix cell indexes
	 * @return value as String
	 */
	public abstract String getString(int[] ix);

	/**
	 * Get the value of a given cell as long
	 *
	 * @param ix cell indexes
	 * @return value as long
	 */
	public abstract long getLong(int[] ix);

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(int i=0; i<_rlen; i++) {
			double[] data = values(i);
			int ix = pos(i);
			for(int j=0; j<_odims[0]; j++) {
				sb.append(data[ix+j]);
				sb.append("\t");
			}
			sb.append("\n");
		}
		return sb.toString();
	}
	
	protected double[] getReuseRow(boolean reset) {
		if( _reuse != null && reset )
			Arrays.fill(_reuse, 0);
		if( _reuse == null )
			_reuse = new double[_odims[0]];
		return _reuse;
	}
	
	private static int[] createDimOffsets(int[] dims) {
		int[] ret = new int[dims.length-1];
		int prod = 1;
		for(int i=dims.length-1; i>=1; i--) {
			prod *= dims[i];
			ret[i-1] = prod;
		}
		return ret;
	}
}

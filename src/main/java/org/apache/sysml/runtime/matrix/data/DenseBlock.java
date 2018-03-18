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


package org.apache.sysml.runtime.matrix.data;

import java.io.Serializable;

import org.apache.sysml.runtime.instructions.cp.KahanObject;

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
	
	/**
	 * Resets the dense block by deleting non-zero values. After this
	 * call all countNonZeros() calls are guaranteed to return 0.
	 */
	public abstract void reset();
	
	/**
	 * Resets the dense block by deleting non-zero values. After this
	 * call all countNonZeros() calls are guaranteed to return 0. If
	 * the new dimensions exceed the current capacity, the underlying
	 * storage is extended accordingly.
	 * 
	 * @param rlen number of rows
	 * @param clen number of columns
	 */
	public abstract void reset(int rlen, int clen);
	
	/**
	 * Resets the dense block by setting the given value.
	 * 
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @param v value
	 */
	public abstract void reset(int rlen, int clen, double v);
	
	
	/**
	 * Get the number of rows.
	 * 
	 * @return number of rows
	 */
	public abstract int numRows();
	
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
	 * of row and column dimensions.
	 * 
	 * @return length
	 */
	public abstract long size();
	
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
	 * Get the allocated blocks.
	 * 
	 * @return blocks
	 */
	public abstract double[][] values();
	
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
	 * Set the given value for the entire dense block (fill).
	 * 
	 * @param v value
	 */
	public abstract void set(double v);
	
	/**
	 * Set the given value for an entire index range of the 
	 * dense block (fill).
	 * 
	 * @param rl row lower index
	 * @param ru row upper index (exclusive)
	 * @param cl column lower index 
	 * @param cu column upper index (exclusive)
	 * @param v value
	 */
	public abstract void set(int rl, int ru, int cl, int cu, double v);
	
	
	/**
	 * Set the given value for a given row and column.
	 * 
	 * @param r row index
	 * @param c column index
	 * @param v value
	 */
	public abstract void set(int r, int c, double v);
	
	/**
	 * Copy the given vector into the given row.
	 * 
	 * @param r row index
	 * @param v value vector
	 */
	public abstract void set(int r, double[] v);
	
	/**
	 * Copy the given dense block.
	 * 
	 * @param db dense block
	 */
	public abstract void set(DenseBlock db);
	
	/**
	 * Copy the given dense block into the specified
	 * index range.
	 * 
	 * @param rl row lower index
	 * @param ru row upper index (exclusive)
	 * @param cl column lower index 
	 * @param cu column upper index (exclusive)
	 * @param db dense block
	 */
	public abstract void set(int rl, int ru, int cl, int cu, DenseBlock db);
	
	
	/**
	 * Copy the given kahan object sum and correction.
	 * 
	 * @param kbuff kahan object
	 */
	public void set(KahanObject kbuff) {
		set(0, 0, kbuff._sum);
		set(0, 1, kbuff._correction);
	}
	
	/**
	 * Copy the given kahan object sum and correction
	 * into the given row.
	 * 
	 * @param r row index
	 * @param kbuff kahan object
	 */
	public void set(int r, KahanObject kbuff) {
		set(r, 0, kbuff._sum);
		set(r, 1, kbuff._correction);
	}
	
	/**
	 * Get the value for a given row and column.
	 * 
	 * @param r row index
	 * @param c column index
	 * @return value
	 */
	public abstract double get(int r, int c);
	
	@Override 
	public abstract String toString();
}

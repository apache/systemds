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
	 * Get the length of the dense block as the product
	 * of row and column dimensions.
	 * 
	 * @return length
	 */
	public abstract long size();
	
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
	 * Get the allocated blocks.
	 * 
	 * @return blocks
	 */
	public abstract double[][] values();
	
	
	/**
	 * Get an allocated block.
	 * 
	 * @param bix block index
	 * @return block
	 */
	public abstract double[] values(int bix);
	
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
	 * Set the given value for a given row and column.
	 * 
	 * @param r row index
	 * @param c column index
	 * @param v value
	 */
	public abstract void set(int r, int c, double v);
	
	/**
	 * Get the value for a given row and column.
	 * 
	 * @param r row index
	 * @param c column index
	 * @return value
	 */
	public abstract double get(int r, int c);
}

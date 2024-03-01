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

package org.apache.sysds.runtime.compress.cost;

import java.io.Serializable;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * A cost estimator interface. It is used to estimate the cost at different stages of compression, and used to compare
 * to uncompressed.
 */
public abstract class ACostEstimate implements Serializable {
	private static final long serialVersionUID = -3241425555L;
	protected static final Log LOG = LogFactory.getLog(ACostEstimate.class.getName());

	protected ACostEstimate() {
		// constructor for sub classes
	}

	/**
	 * Get the cost of a collection of column groups.
	 * 
	 * @param cgs   A collection of column groups
	 * @param nRows The number of rows in the column group
	 * @return The summed cost of the groups
	 */
	public double getCost(Collection<AColGroup> cgs, int nRows) {
		double c = 0;
		for(AColGroup g : cgs)
			c += getCost(g, nRows);
		return c;
	}

	/**
	 * Get cost of a single column group estimate.
	 * 
	 * If the input is null the cost is positive infinity indicating that it is impossible to compress the null case.
	 * 
	 * If the instruction does not care about the inter column group cost, such as in memory cost or in computation cost
	 * of right or left matrix multiplication we simply estimate the cost of individual column groups.
	 * 
	 * @param g Column group to estimate the cost of
	 * @return The Cost of this column group
	 */
	public final double getCost(CompressedSizeInfoColGroup g) {
		return g == null ? Double.POSITIVE_INFINITY : getCostSafe(g);
	}

	/**
	 * Get cost of an entire compression plan
	 * 
	 * @param i The compression plan to get the cost of
	 * @return The cost
	 */
	public double getCost(CompressedSizeInfo i) {
		double c = 0;
		for(CompressedSizeInfoColGroup g : i.getInfo())
			c += getCost(g);
		
		return c;
	}

	/**
	 * Get cost of a compressed matrix block
	 * 
	 * @param cmb The compressed matrix block
	 * @return The cost
	 */
	public double getCost(CompressedMatrixBlock cmb) {
		int nRows = cmb.getNumRows();
		double c = 0;
		for(AColGroup g : cmb.getColGroups())
			c += getCost(g, nRows);
		return c;
	}

	/**
	 * Get the cost of a matrix block.
	 * 
	 * @param mb A MatrixBlock
	 * @return The cost subject to the internal cost functions
	 */
	public abstract double getCost(MatrixBlock mb);

	/**
	 * Get the cost of a compressed columnGroup.
	 * 
	 * @param cg    A ColumnGroup
	 * @param nRows The number of rows in the column group
	 * @return The cost subject to the internal cost functions
	 */
	public abstract double getCost(AColGroup cg, int nRows);

	/**
	 * Get cost of a single column group estimate.
	 * 
	 * @param g The estimated information about a specific column group.
	 * @return The estimated cost
	 */
	protected abstract double getCostSafe(CompressedSizeInfoColGroup g);

	/**
	 * Ask the cost estimator if it is a good idea to try to sparsify a column group. It is the same as asking if it is a
	 * good idea to make FOR on top of the column group.
	 * 
	 * @return true if yes
	 */
	public abstract boolean shouldSparsify();

	@Override
	public String toString() {
		return this.getClass().getSimpleName();
	}

}

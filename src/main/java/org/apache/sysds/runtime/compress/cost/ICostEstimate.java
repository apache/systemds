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
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

/**
 * A cost estimator interface.
 * 
 * Cost here is either memory based or computation based.
 * 
 * It is used to estimate the cost of a specific compression layout.
 */
public interface ICostEstimate extends Serializable {

	public static final long serialVersionUID = -8885748390L;

	public static final Log LOG = LogFactory.getLog(ICostEstimate.class.getName());

	/**
	 * Calculate the cost of execution if uncompressed.
	 * 
	 * This is used as an optimization goal to at least achieve, since if the uncompressed cost of calculation is lower
	 * than in compressed space we don't want to compress.
	 * 
	 * @param nRows    Number of rows in compression
	 * @param nCols    Number of columns in compression
	 * @param sparsity Sparsity of the input
	 * @return The uncompressed cost.
	 */
	public double getUncompressedCost(int nRows, int nCols, int sparsity);

	/**
	 * If the instruction does not care about the inter column group cost, such as in memory cost or in computation cost
	 * of right or left matrix multiplication we simply estimate the cost of individual column groups.
	 * 
	 * @param g Column group to estimate the cost of
	 * @return The Cost of this column group
	 */
	public double getCostOfColumnGroup(CompressedSizeInfoColGroup g);

	/**
	 * Get the computation cost of a collection of column groups information.
	 * 
	 * Since a some operations execution time depends on all compare all calculating the cost needs all the current
	 * groups to consider. This method therefore provide a method of comparing cost of all compare all.
	 * 
	 * @param gs All the groups considered for this cost calculation
	 * @return The cost of the calculation
	 */
	public double getCostOfCollectionOfGroups(Collection<CompressedSizeInfoColGroup> gs);

	public double getCostOfCollectionOfGroups(Collection<CompressedSizeInfoColGroup> gs, CompressedSizeInfoColGroup g);

	public double getCostOfTwoGroups(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2);

	/**
	 * Function that returns true if the cost is based an an all compare all column groups false otherwise.
	 * 
	 * @return boolean
	 */
	public boolean isCompareAll();

	/**
	 * Decide if the column groups should try to join, this is to filter obviously bad joins out.
	 * 
	 * @param g1 Group 1
	 * @param g2 Group 2
	 * @return Boolean specifying if they should try to join.
	 */
	public boolean shouldTryJoin(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2);

	/**
	 * Decide if the column groups should be analysed, or the worst case join should be expected. This is use full if
	 * the column groups are very small and the in practice difference between joining or not is small.
	 * 
	 * @param g1 Group 1
	 * @param g2 Group 2
	 * @return If the joining should be analyzed.
	 */
	public boolean shouldAnalyze(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2);


	/**
	 * Heuristic function that define if compression should be tried
	 * @return boolean specifying if we should try to compress
	 */
	public boolean shouldTryToCompress();
}

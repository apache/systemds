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

package org.apache.sysml.hops.estim;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public abstract class SparsityEstimator 
{
	protected static final Log LOG = LogFactory.getLog(SparsityEstimator.class.getName());
	
	/**
	 * Estimates the output sparsity of a DAG of matrix multiplications
	 * for the given operator graph of a single root node.
	 * 
	 * @param root
	 * @return
	 */
	public abstract double estim(MMNode root);
	
	/**
	 * Estimates the output sparsity of a single matrix multiplication
	 * for the two given matrices.
	 * 
	 * @param m1 left-hand-side operand
	 * @param m2 right-hand-side operand
	 * @return sparsity
	 */
	public abstract double estim(MatrixBlock m1, MatrixBlock m2);
	
	/**
	 * Estimates the output sparsity of a single matrix multiplication
	 * for the two given matrices represented by meta data only.
	 * 
	 * @param mc1 left-hand-side operand
	 * @param mc2 right-hand-side operand
	 * @return sparsity
	 */
	public abstract double estim(MatrixCharacteristics mc1, MatrixCharacteristics mc2);

}

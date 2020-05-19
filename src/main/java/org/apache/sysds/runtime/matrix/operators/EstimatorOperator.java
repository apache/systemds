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

package org.apache.sysds.runtime.matrix.operators;

import org.apache.sysds.runtime.matrix.data.LibMatrixEstimator.EstimatorType;
import org.apache.sysds.utils.Hash.HashType;

public class EstimatorOperator extends Operator {
	private static final long serialVersionUID = 7615123453265129670L;

	public final EstimatorType operatorType;
	public final HashType hashType;

	public EstimatorOperator(EstimatorType operatorType) {
		super(true);
		this.operatorType = operatorType;
		this.hashType = HashType.StandardJava;
	}

	public EstimatorOperator(EstimatorType operatorType, HashType hashType) {
		super(true);
		this.operatorType = operatorType;
		this.hashType = hashType;
	}
}
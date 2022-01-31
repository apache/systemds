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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction.AUType;
import org.apache.sysds.utils.Hash.HashType;

public class CountDistinctOperator extends Operator {
	private static final long serialVersionUID = 7615123453265129670L;

	private final CountDistinctOperatorTypes operatorType;
	private final HashType hashType;
	private Types.Direction direction;
	private IndexFunction indexFunction;

	public CountDistinctOperator(AUType opType) {
		super(true);
		switch (opType) {
			case COUNT_DISTINCT:
				this.operatorType = CountDistinctOperatorTypes.COUNT;
				break;
			case COUNT_DISTINCT_APPROX:
				this.operatorType = CountDistinctOperatorTypes.KMV;
				break;
			default:
				throw new DMLRuntimeException(opType + " not supported for CountDistinct Operator");
		}
		this.hashType = HashType.LinearHash;
	}

	public CountDistinctOperator(CountDistinctOperatorTypes operatorType) {
		super(true);
		this.operatorType = operatorType;
		this.hashType = HashType.StandardJava;
	}

	public CountDistinctOperator(CountDistinctOperatorTypes operatorType, HashType hashType) {
		super(true);
		this.operatorType = operatorType;
		this.hashType = hashType;
	}

	public CountDistinctOperator(CountDistinctOperatorTypes operatorType, IndexFunction indexFunction, HashType hashType) {
		super(true);
		this.operatorType = operatorType;
		this.indexFunction = indexFunction;
		this.hashType = hashType;
	}

	public CountDistinctOperatorTypes getOperatorType() {
		return operatorType;
	}

	public HashType getHashType() {
		return hashType;
	}

	public IndexFunction getIndexFunction() {
		return indexFunction;
	}

	public CountDistinctOperator setIndexFunction(IndexFunction indexFunction) {
		this.indexFunction = indexFunction;
		return this;
	}

	public Types.Direction getDirection() {
		return direction;
	}

	public CountDistinctOperator setDirection(Types.Direction direction) {
		this.direction = direction;
		return this;
	}
}
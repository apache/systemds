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
import org.apache.sysds.runtime.functionobjects.IndexFunction;

public class UnarySketchOperator extends AggregateUnaryOperator {
	private static final long serialVersionUID = 7615123453265129671L;

	private final Types.Direction direction;

	public UnarySketchOperator(AggregateOperator aop, IndexFunction indexFunction, Types.Direction direction) {
		super(aop, indexFunction);
		this.direction = direction;
	}

	public UnarySketchOperator(AggregateOperator aop, IndexFunction indexFunction,
							   Types.Direction direction, int numThreads) {
		super(aop, indexFunction, numThreads);
		this.direction = direction;
	}

	public Types.Direction getDirection() {
		return direction;
	}
}

/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
package org.tugraz.sysds.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.tugraz.sysds.runtime.data.BasicTensorBlock;

/**
 * General purpose copy function for binary block rdds. This function can be used in
 * mapValues (copy tensor blocks). It supports both deep and shallow copies of values.
 */
public class CopyTensorBlockFunction implements Function<BasicTensorBlock, BasicTensorBlock> {
	private static final long serialVersionUID = 707987326466592670L;
	private boolean _deepCopy;

	public CopyTensorBlockFunction() {
		this(true);
	}

	public CopyTensorBlockFunction(boolean deepCopy) {
		_deepCopy = deepCopy;
	}

	@Override
	public BasicTensorBlock call(BasicTensorBlock arg0)
			throws Exception {
		if (_deepCopy)
			return new BasicTensorBlock(arg0);
		else
			return arg0;
	}
}
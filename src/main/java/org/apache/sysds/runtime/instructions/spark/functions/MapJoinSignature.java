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

package org.apache.sysds.runtime.instructions.spark.functions;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.api.java.function.Function;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import scala.Tuple2;

public class MapJoinSignature implements Function<Tuple2<MatrixBlock[],MatrixBlock>, MatrixBlock[]> {
	private static final long serialVersionUID = -704403012606821854L;

	@Override
	public MatrixBlock[] call(Tuple2<MatrixBlock[], MatrixBlock> v1) throws Exception {
		return ArrayUtils.add(v1._1(), v1._2());
	}
}

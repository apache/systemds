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

package org.apache.sysml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.UtilFunctions;

public class IsFrameBlockInRange implements Function<Tuple2<Long,FrameBlock>, Boolean> 
{
	private static final long serialVersionUID = 4433918122474769296L;

	private long _rl; long _ru; long _cl; long _cu;
	private int _brlen; int _bclen;
	
	public IsFrameBlockInRange(long rl, long ru, long cl, long cu, MatrixCharacteristics mcOut) {
		_rl = rl;
		_ru = ru;
		_cl = cl;
		_cu = cu;
		_brlen = OptimizerUtils.getDefaultFrameSize();
		_bclen = (int) mcOut.getCols();

	}

	@Override
	public Boolean call(Tuple2<Long, FrameBlock> kv) 
		throws Exception 
	{
		return UtilFunctions.isInFrameBlockRange(kv._1(), _brlen, _bclen, _rl, _ru);
	}
}

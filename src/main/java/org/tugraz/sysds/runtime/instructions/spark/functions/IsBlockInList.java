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
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

public class IsBlockInList implements Function<Tuple2<MatrixIndexes,MatrixBlock>, Boolean> 
{
	private static final long serialVersionUID = -1956151588590369875L;
	
	private final long[] _cols;
	private final int _blen;
	
	public IsBlockInList(long[] cols, DataCharacteristics mc) {
		_cols = cols;
		_blen = mc.getBlocksize();
	}

	@Override
	public Boolean call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
		throws Exception 
	{
		for( int i=0; i<_cols.length; i++ )
			if( UtilFunctions.isInBlockRange(kv._1(), _blen, 1, Long.MAX_VALUE, _cols[i], _cols[i]) )
				return true;
		return false;
	}
}

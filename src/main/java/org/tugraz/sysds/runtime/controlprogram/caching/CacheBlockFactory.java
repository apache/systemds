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

package org.tugraz.sysds.runtime.controlprogram.caching;

import java.util.ArrayList;

import org.tugraz.sysds.runtime.data.BasicTensor;
import org.tugraz.sysds.runtime.data.DataTensor;
import org.tugraz.sysds.runtime.data.TensorIndexes;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.Pair;

/**
 * Factory to create instances of matrix/frame blocks given
 * internal codes.
 * 
 */
public class CacheBlockFactory 
{
	public static CacheBlock newInstance(int code) {
		switch( code ) {
			case 0: return new MatrixBlock();
			case 1: return new FrameBlock();
			case 2: return new BasicTensor();
			case 3: return new DataTensor();
		}
		throw new RuntimeException("Unsupported cache block type: "+code);
	}

	public static int getCode(CacheBlock block) {
		if (block instanceof MatrixBlock)
			return 0;
		else if (block instanceof FrameBlock)
			return 1;
		else if (block instanceof BasicTensor)
			return 2;
		else if (block instanceof DataTensor)
			return 3;
		throw new RuntimeException("Unsupported cache block type: " + block.getClass().getName());
	}

	public static ArrayList<?> getPairList(CacheBlock block) {
		int code = getCode(block);
		switch (code) {
			case 0: return new ArrayList<Pair<MatrixIndexes, MatrixBlock>>();
			case 1: return new ArrayList<Pair<Long, FrameBlock>>();
			case 2: return new ArrayList<Pair<TensorIndexes, BasicTensor>>();
			case 3: return new ArrayList<Pair<TensorIndexes, DataTensor>>();
		}
		throw new RuntimeException("Unsupported cache block type: "+code);
	}
}

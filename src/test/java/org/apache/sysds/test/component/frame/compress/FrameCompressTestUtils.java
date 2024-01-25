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

package org.apache.sysds.test.component.frame.compress;

import java.util.Random;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.test.component.frame.array.FrameArrayTests;

public class FrameCompressTestUtils {
	protected static final Log LOG = LogFactory.getLog(FrameCompressTest.class.getName());

	public static FrameBlock generateCompressableBlock(int rows, int cols, int seed, ValueType vt) {
		Array<?>[] data = new Array<?>[cols];
		for(int i = 0; i < cols; i++)
			data[i] = generateArray(rows, seed + i, i + 1, vt);

		return new FrameBlock(data);
	}

	public static FrameBlock generateCompressableBlockRandomTypes(int rows, int cols, int seed) {
		Array<?>[] data = new Array<?>[cols];
		Random r = new Random(seed + 13);
		for(int i = 0; i < cols; i++) {
			ValueType vt = ValueType.values()[r.nextInt(ValueType.values().length)];
			data[i] = generateArray(rows, seed + i, i + 1, vt);
		}

		return new FrameBlock(data);
	}

	public static Array<?> generateArray(int size, int seed, int nUnique, ValueType vt) {
		switch(vt) {
			case BOOLEAN:
				return ArrayFactory.create(FrameArrayTests.generateRandomBooleanOpt(size, seed));
			case UINT8:
			case UINT4:
			case INT32:
				return ArrayFactory.create(FrameArrayTests.generateRandomIntegerNUniqueLengthOpt(size, seed, nUnique));
			case INT64:
				return ArrayFactory.create(FrameArrayTests.generateRandomLongNUniqueLengthOpt(size, seed, nUnique));
			case FP32:
				return ArrayFactory.create(FrameArrayTests.generateRandomFloatNUniqueLengthOpt(size, seed, nUnique));
			case FP64:
				return ArrayFactory.create(FrameArrayTests.generateRandomDoubleNUniqueLengthOpt(size, seed, nUnique));
			case CHARACTER:
				return ArrayFactory.create(FrameArrayTests.generateRandomCharacterNUniqueLengthOpt(size, seed, nUnique));
			case HASH64:
				return ArrayFactory.create(FrameArrayTests.generateRandomHash64OptNUnique(size, seed, nUnique));
			case HASH32:
				throw new NotImplementedException();
			case STRING:
				return ArrayFactory.create(FrameArrayTests.generateRandomStringNUniqueLengthOpt(size, seed, nUnique, 132));
			default:
				throw new NotImplementedException(vt + "");
		}
	}
}

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

package org.apache.sysds.runtime.compress.colgroup.offset;

import java.io.DataInput;
import java.io.IOException;

import org.apache.sysds.runtime.DMLCompressionException;

public class OffsetFactory {

	protected enum Types {
		BYTE, CHAR
	}

	public static AOffset create(int[] indexes, int nRows) {
		if((float) nRows / (float) indexes.length < 256)
			return new OffsetByte(indexes);
		else
			return new OffsetChar(indexes);
	}

	public static AOffset readIn(DataInput in) throws IOException {
		Types t = Types.values()[in.readByte()];
		switch(t) {
			case BYTE:
				return OffsetByte.readFields(in);
			case CHAR:
				return OffsetChar.readFields(in);
			default:
				throw new DMLCompressionException("Unknown input");
		}

	}

	public static long estimateInMemorySize(int size, int nRows) {
		if((float) nRows / (float) size < 256)
			return OffsetByte.getInMemorySize(size);
		else
			return OffsetChar.getInMemorySize(size);

	}
}

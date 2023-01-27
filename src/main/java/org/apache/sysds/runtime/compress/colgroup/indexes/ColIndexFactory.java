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

package org.apache.sysds.runtime.compress.colgroup.indexes;

import java.io.DataInput;
import java.io.IOException;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex.ColIndexType;

public interface ColIndexFactory {

	public static IColIndex read(DataInput in) throws IOException {
		final ColIndexType t = ColIndexType.values()[in.readByte()];
		switch(t) {
			case SINGLE:
				return new SingleIndex(in.readInt());
			case TWO:
				return new TwoIndex(in.readInt(), in.readInt());
			default:
				throw new DMLCompressionException("Failed reading column index of type: " + t);
		}
	}

	public static IColIndex create(int[] indexes) {
		if(indexes.length == 1)
			return new SingleIndex(indexes[0]);
		else if(indexes.length == 2)
			return new TwoIndex(indexes[0], indexes[1]);
		throw new NotImplementedException();
	}

	public static IColIndex create(int l, int u) {
		if(u - 1 == l)
			return new SingleIndex(l);
		throw new NotImplementedException();
	}

	public static IColIndex create(int nCol) {
		if(nCol == 1)
			return new SingleIndex(0);
		throw new NotImplementedException();
	}
}

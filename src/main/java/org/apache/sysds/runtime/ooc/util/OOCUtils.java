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

package org.apache.sysds.runtime.ooc.util;

import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class OOCUtils {
	public static IndexRange getRangeOfTile(MatrixIndexes tileIdx, long blen) {
		long rs = 1 + tileIdx.getRowIndex() * blen;
		long re = (tileIdx.getRowIndex() + 1) * blen;
		long cs = 1 + tileIdx.getColumnIndex() * blen;
		long ce = (tileIdx.getColumnIndex() + 1) * blen;
		return new IndexRange(rs, re, cs, ce);
	}

	public static Collection<MatrixIndexes> getTilesOfRange(IndexRange range, long blen) {
		long rs = (range.rowStart - 1) / blen + 1;
		long re = (range.rowEnd - 1) / blen + 1;
		long cs = (range.colStart - 1) / blen + 1;
		long ce = (range.colEnd - 1) / blen + 1;

		if(rs == re) {
			if(cs == ce) {
				return Collections.singleton(new MatrixIndexes(rs, cs));
			}
			else {
				List<MatrixIndexes> list = new ArrayList<>((int)(ce-cs+1));
				for(long i = cs; i <= ce; i++)
					list.add(new MatrixIndexes(rs, i));
				return list;
			}
		}

		List<MatrixIndexes> list = new ArrayList<>((int)((re-rs+1)*(ce-cs+1)));
		for(long r = rs; r <= re; r++)
			for (long c = cs; c <= ce; c++)
				list.add(new MatrixIndexes(r, c));
		return list;
	}
}

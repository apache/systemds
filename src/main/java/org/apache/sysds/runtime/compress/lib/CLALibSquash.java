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

package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;

/**
 * Squash or recompress is processing each column group and trying to find a better compressed representation for each.
 */
public final class CLALibSquash {

	private CLALibSquash() {
		// private constructor
	}

	/**
	 * Squash or recompress is process each column group in the given Compressed Matrix Block and tries to recompress
	 * each column.
	 * 
	 * @param m The input compressed matrix
	 * @param k The parallelization degree allowed in this process
	 * @return A replaced Compressed Matrix Block, note the old block is also modified
	 */
	public static CompressedMatrixBlock squash(CompressedMatrixBlock m, int k) {
		List<AColGroup> before = m.getColGroups();
		List<AColGroup> groups = new ArrayList<>(before.size());

		for(AColGroup g : before)
			groups.add(g.recompress());

		m.allocateColGroupList(groups);
		return m;
	}
}

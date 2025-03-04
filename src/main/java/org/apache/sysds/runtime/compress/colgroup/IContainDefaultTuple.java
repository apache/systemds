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

package org.apache.sysds.runtime.compress.colgroup;

import java.util.List;

public interface IContainDefaultTuple {
	public double[] getDefaultTuple();

	public static double[] combineDefaultTuples(double[] defTup,  List<AColGroup> right) {
		final int l = defTup.length;
		final double[] out = new double[l * right.size() + l];
		System.arraycopy(defTup, 0, out, 0, l);
		for(int i = l, j = 0; j < right.size(); i += l, j++) {
			final IContainDefaultTuple dtg = (IContainDefaultTuple) right.get(j);
			System.arraycopy(dtg.getDefaultTuple(), 0, out, i, l);
		}
		return out;
	}
}

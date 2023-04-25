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

package org.apache.sysds.runtime.compress.plan;

import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.SingleIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.TwoIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.SchemeFactory;

public class CompressionPlanFactory {

	public static IPlanEncode singleCols(int nCol, CompressionType type, int k) {
		ICLAScheme[] schemes = new ICLAScheme[nCol];
		for(int i = 0; i < nCol; i++)
			schemes[i] = SchemeFactory.create(new SingleIndex(i), type);
		return new NaivePlanEncode(schemes, k, false);
	}

	public static IPlanEncode twoCols(int nCol, CompressionType type, int k) {
		ICLAScheme[] schemes = new ICLAScheme[nCol / 2 + nCol % 2];
		for(int i = 0; i < nCol; i += 2) {
			schemes[i/2] = i + 1 >= nCol ? //
				SchemeFactory.create(new SingleIndex(i), type) : //
				SchemeFactory.create(new TwoIndex(i, i + 1), type);
		}
		return new NaivePlanEncode(schemes, k, false);
	}

	public static IPlanEncode nCols(int nCol, int n, CompressionType type, int k){
		ICLAScheme[] schemes = new ICLAScheme[nCol / n + (nCol % n  != 0 ? 1 : 0)];
		for(int i = 0; i < nCol; i += n) {
			schemes[i/n] = i + n < nCol ? //
				SchemeFactory.create( ColIndexFactory.create(i, i+n), type) : //
				SchemeFactory.create(ColIndexFactory.create(i, nCol), type);
		}
		return new NaivePlanEncode(schemes, k, false);
	}

	public static IPlanEncode create(IColIndex[] columnGroups, CompressionType[] type, int k) {
		ICLAScheme[] schemes = new ICLAScheme[columnGroups.length];
		for(int i = 0; i < columnGroups.length; i++)
			schemes[i] = SchemeFactory.create(columnGroups[i], type[i]);

		// TODO check for overlapping
		return new NaivePlanEncode(schemes, k, false);
	}
}

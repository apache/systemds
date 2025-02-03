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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibReplace {
	private static final Log LOG = LogFactory.getLog(CLALibReplace.class.getName());

	public static MatrixBlock replace(CompressedMatrixBlock in, MatrixBlock out, double pattern, double replacement,
		int k) {
		if(Double.isInfinite(pattern)) {
			LOG.info("Ignoring replace infinite in compression since it does not contain this value");
			return in;
		}
		else if(in.isOverlapping()) {
			final String message = "replaceOperations " + pattern + " -> " + replacement;
			return in.getUncompressed(message).replaceOperations(out, pattern, replacement);
		}
		else
			return replaceNormal(in, out, pattern, replacement, k);
	}

	private static MatrixBlock replaceNormal(CompressedMatrixBlock in, MatrixBlock out, double pattern,
		double replacement, int k) {
		CompressedMatrixBlock ret = new CompressedMatrixBlock(in.getNumRows(), in.getNumColumns());
		final List<AColGroup> prev = in.getColGroups();
		final int colGroupsLength = prev.size();
		final List<AColGroup> retList = new ArrayList<>(colGroupsLength);

		if(k <= 0) {
			for(int i = 0; i < colGroupsLength; i++)
				retList.add(prev.get(i).replace(pattern, replacement));
		}
		else {
			ExecutorService pool = CommonThreadPool.get(k);

			try {
				List<Future<AColGroup>> tasks = new ArrayList<>(colGroupsLength);
				for(int i = 0; i < colGroupsLength; i++) {
					final int j = i;
					tasks.add(pool.submit(() -> prev.get(j).replace(pattern, replacement)));
				}
				for(int i = 0; i < colGroupsLength; i++) {
					retList.add(tasks.get(i).get());
				}
			}
			catch(Exception e) {
				throw new RuntimeException("Failed parallel replace", e);
			}
			finally {
				pool.shutdown();
			}
		}

		ret.allocateColGroupList(retList);
		if(replacement == 0)
			ret.recomputeNonZeros();
		else if( pattern == 0)
			ret.setNonZeros(((long)in.getNumRows()) * in.getNumColumns());
		else
			ret.setNonZeros(in.getNonZeros());
		return ret;
	}
}

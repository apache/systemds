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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Naive implementation of encoding based on a plan. This does not reuse plans across groups, and does not smartly
 * extract encodings.
 */
public class NaivePlanEncode implements IPlanEncode {

	/** The schemes to apply to the input. */
	private final ICLAScheme[] schemes;
	/** The parallelization degree to use in this encoder. */
	private final int k;
	/** If the schemes are overlapping */
	private final boolean overlapping;

	public NaivePlanEncode(ICLAScheme[] schemes, int k, boolean overlapping) {
		this.schemes = schemes;
		this.k = k;
		this.overlapping = overlapping;
	}

	@Override
	public CompressedMatrixBlock encode(MatrixBlock in) {
		try {
			final List<AColGroup> groups = k <= 1 ? encodeSingleThread(in) : encodeMultiThread(in);
			return new CompressedMatrixBlock(in.getNumRows(), in.getNumColumns(), in.getNonZeros(), overlapping, groups);
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed encoding matrix", e);
		}
	}

	private List<AColGroup> encodeSingleThread(MatrixBlock in) {
		List<AColGroup> groups = new ArrayList<>(schemes.length);
		for(int i = 0; i < schemes.length; i++)
			groups.add(schemes[i].encode(in));
		return groups;
	}

	private List<AColGroup> encodeMultiThread(MatrixBlock in) throws Exception {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {

			List<EncodeTask> t = new ArrayList<>(schemes.length);
			for(int i = 0; i < schemes.length; i++)
				t.add(new EncodeTask(in, schemes[i]));

			List<AColGroup> groups = new ArrayList<>(schemes.length);
			for(Future<AColGroup> f : pool.invokeAll(t))
				groups.add(f.get());
			return groups;
		}
		finally {
			pool.shutdown();
		}
	}

	@Override
	public void expandPlan(MatrixBlock in) {
		try {
			if(k <= 1)
				expandPlanSingleThread(in);
			else
				expandPlanMultiThread(in);
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed expanding plan", e);
		}
	}

	public void expandPlanSingleThread(MatrixBlock in) {

		for(int i = 0; i < schemes.length; i++)
			schemes[i] = schemes[i].update(in);
	}

	public void expandPlanMultiThread(MatrixBlock in) throws Exception {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			List<ExpandTask> t = new ArrayList<>(schemes.length);
			for(int i = 0; i < schemes.length; i++)
				t.add(new ExpandTask(in, schemes[i]));
			int i = 0;
			for(Future<ICLAScheme> f : pool.invokeAll(t))
				schemes[i++] = f.get();

		}
		finally {
			pool.shutdown();
		}

	}

	@Override
	public final String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" Parallelization: " + k);
		sb.append(" Overlapping: " + overlapping);
		sb.append("\n");
		for(int i = 0; i < schemes.length; i++) {
			sb.append(schemes[i]);
			sb.append("\n");
		}
		return sb.toString();
	}

	private static class EncodeTask implements Callable<AColGroup> {
		private final MatrixBlock in;
		private final ICLAScheme sc;

		protected EncodeTask(MatrixBlock in, ICLAScheme sc) {
			this.in = in;
			this.sc = sc;
		}

		@Override
		public AColGroup call() throws Exception {
			try {
				return sc.encode(in);
			}
			catch(Exception e) {
				throw new DMLCompressionException("Failed encoding schema");
			}
		}
	}

	private static class ExpandTask implements Callable<ICLAScheme> {
		private final MatrixBlock in;
		private final ICLAScheme sc;

		protected ExpandTask(MatrixBlock in, ICLAScheme sc) {
			this.in = in;
			this.sc = sc;
		}

		@Override
		public ICLAScheme call() throws Exception {
			try {

				return sc.update(in);
			}
			catch(Exception e) {
				throw new DMLCompressionException("Failed Expanding schema");
			}
		}
	}
}

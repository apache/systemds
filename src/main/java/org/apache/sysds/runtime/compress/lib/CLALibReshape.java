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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class CLALibReshape {

	protected static final Log LOG = LogFactory.getLog(CLALibReshape.class.getName());

	/** The minimum number of rows threshold for returning a compressed output */
	public static int COMPRESSED_RESHAPE_THRESHOLD = 1000;

	final CompressedMatrixBlock in;

	final int clen;
	final int rlen;
	final int rows;
	final int cols;

	final boolean rowwise;

	final ExecutorService pool;

	private CLALibReshape(CompressedMatrixBlock in, int rows, int cols, boolean rowwise, int k) {
		this.in = in;
		this.rlen = in.getNumRows();
		this.clen = in.getNumColumns();
		this.rows = rows;
		this.cols = cols;
		this.rowwise = rowwise;
		this.pool = k > 1 ? CommonThreadPool.get(k) : null;
	}

	public static MatrixBlock reshape(CompressedMatrixBlock in, int rows, int cols, boolean rowwise) {
		return new CLALibReshape(in, rows, cols, rowwise, InfrastructureAnalyzer.getLocalParallelism()).apply();
	}

	public static MatrixBlock reshape(CompressedMatrixBlock in, int rows, int cols, boolean rowwise, int k) {
		return new CLALibReshape(in, rows, cols, rowwise, k).apply();
	}

	private MatrixBlock apply() {
		try {
			checkValidity();
			if(shouldItBeCompressedOutputs())
				return applyCompressed();
			else
				return in.decompress().reshape(rows, cols, rowwise);
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed reshaping of compressed matrix", e);
		}
		finally {
			if(pool != null)
				pool.shutdown();
		}
	}

	private MatrixBlock applyCompressed() throws Exception {
		final int multiplier = rlen / rows;
		final List<AColGroup> retGroups;
		if(pool == null)
			retGroups = applySingleThread(multiplier);
		else if (in.getColGroups().size() == 1)
			retGroups = applyParallelPushDown(multiplier);
		else
			retGroups = applyParallel(multiplier);

		CompressedMatrixBlock ret = new CompressedMatrixBlock(rows, cols);
		ret.allocateColGroupList(retGroups);
		ret.setNonZeros(in.getNonZeros());
		return ret;
	}

	private List<AColGroup> applySingleThread(int multiplier) {
		List<AColGroup> groups = in.getColGroups();
		List<AColGroup> retGroups = new ArrayList<>(groups.size() * multiplier);

		for(AColGroup g : groups) {
			final AColGroup[] tg = g.splitReshape(multiplier, rlen, clen);
			for(int i = 0; i < tg.length; i++)
				retGroups.add(tg[i]);
		}

		return retGroups;

	}


	private List<AColGroup> applyParallelPushDown(int multiplier) throws Exception {
		List<AColGroup> groups = in.getColGroups();

		List<AColGroup> retGroups = new ArrayList<>(groups.size() * multiplier);
		for(AColGroup g : groups){
			final AColGroup[] tg =  g.splitReshapePushDown(multiplier, rlen, clen, pool);

			for(int i = 0; i < tg.length; i++)
				retGroups.add(tg[i]);
		}

		return retGroups;
	}

	private List<AColGroup> applyParallel(int multiplier) throws Exception {
		List<AColGroup> groups = in.getColGroups();
		List<Future<AColGroup[]>> tasks = new ArrayList<>(groups.size());

		for(AColGroup g : groups)
			tasks.add(pool.submit(() -> g.splitReshape(multiplier, rlen, clen)));

		List<AColGroup> retGroups = new ArrayList<>(groups.size() * multiplier);

		for(Future<AColGroup[]> f : tasks) {
			final AColGroup[] tg = f.get();
			for(int i = 0; i < tg.length; i++)
				retGroups.add(tg[i]);
		}

		return retGroups;
	}

	private void checkValidity() {

		// check validity
		if(((long) rlen) * clen != ((long) rows) * cols)
			throw new DMLRuntimeException("Reshape matrix requires consistent numbers of input/output cells (" + rlen + ":"
				+ clen + ", " + rows + ":" + cols + ").");

	}

	private boolean shouldItBeCompressedOutputs() {
		// The number of rows in the reshaped allocations is fairly large.
		return rlen > COMPRESSED_RESHAPE_THRESHOLD && rowwise &&
			// the reshape is a clean multiplier of number of rows, meaning each column group cleanly reshape into x others
			(double) rlen / rows % 1.0 == 0.0;
	}

}

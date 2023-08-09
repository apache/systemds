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

package org.apache.sysds.runtime.compress.colgroup.scheme;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * A Class that contains a full compression scheme that can be applied to MatrixBlocks.
 */
public class CompressionScheme {

	protected static final Log LOG = LogFactory.getLog(CompressionScheme.class.getName());

	private final ICLAScheme[] encodings;

	public CompressionScheme(ICLAScheme[] encodings) {
		this.encodings = encodings;
	}

	/**
	 * Get the encoding in a specific index.
	 * 
	 * @param i the index
	 * @return The encoding in that index
	 */
	public ICLAScheme get(int i) {
		return encodings[i];
	}

	/**
	 * Encode the given matrix block, it is assumed that the given MatrixBlock already fit the current scheme.
	 * 
	 * @param mb A MatrixBlock given that should fit the scheme
	 * @return A Compressed instance of the given matrixBlock;
	 */
	public CompressedMatrixBlock encode(MatrixBlock mb) {
		if(mb instanceof CompressedMatrixBlock)
			throw new NotImplementedException(
				"Not implemented schema encode/apply on an already compressed MatrixBlock");

		List<AColGroup> ret = new ArrayList<>(encodings.length);

		for(int i = 0; i < encodings.length; i++)
			ret.add(encodings[i].encode(mb));

		return new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns(), mb.getNonZeros(), false, ret);
	}

	/**
	 * Encode the given matrix block, it is assumed that the given MatrixBlock already fit the current scheme.
	 * 
	 * @param mb A MatrixBlock given that should fit the scheme
	 * @param k  The parallelization degree
	 * @return A Compressed instance of the given matrixBlock;
	 */
	public CompressedMatrixBlock encode(MatrixBlock mb, int k) {
		if(k == 1)
			return encode(mb);
		final ExecutorService pool = CommonThreadPool.get(k);
		try {

			List<EncodeTask> tasks = new ArrayList<>();
			for(int i = 0; i < encodings.length; i++)
				tasks.add(new EncodeTask(encodings[i], mb));

			List<AColGroup> ret = new ArrayList<>(encodings.length);
			for(Future<AColGroup> t : pool.invokeAll(tasks))
				ret.add(t.get());

			return new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns(), mb.getNonZeros(), false, ret);

		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed encoding", e);
		}
		finally {
			pool.shutdown();
		}
	}

	/**
	 * Update the encodings contained to also enable compression of the given mb.
	 * 
	 * @param mb The matrixBlock to enable compression on.
	 * @return The updated scheme. (It is updated in place)
	 */
	public CompressionScheme update(MatrixBlock mb) {
		if(mb instanceof CompressedMatrixBlock)
			throw new NotImplementedException(
				"Not implemented schema encode/apply on an already compressed MatrixBlock");

		for(int i = 0; i < encodings.length; i++)
			encodings[i] = encodings[i].update(mb);

		return this;

	}

	/**
	 * Update the encodings contained to also enable compression of the given mb.
	 * 
	 * @param mb The matrixBlock to enable compression on.
	 * @param k  The parallelization degree
	 * @return The updated scheme. (It is updated in place)
	 */
	public CompressionScheme update(MatrixBlock mb, int k) {
		if(k == 1)
			return update(mb);
		final ExecutorService pool = CommonThreadPool.get(k);
		try {

			List<UpdateTask> tasks = new ArrayList<>();
			for(int i = 0; i < encodings.length; i++)
				tasks.add(new UpdateTask(encodings[i], mb));

			List<Future<ICLAScheme>> ret = pool.invokeAll(tasks);

			for(int i = 0; i < encodings.length; i++)
				encodings[i] = ret.get(i).get();

			return this;
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed encoding", e);
		}
		finally {
			pool.shutdown();
		}
	}

	/** Extract a compression scheme for the given matrix block */

	/**
	 * Extract a compression scheme for the given matrix block
	 * 
	 * @param cmb The given compressed matrix block
	 * @return A Compression scheme that can be applied to new encodings.
	 */
	public static CompressionScheme getScheme(CompressedMatrixBlock cmb) {
		if(cmb.isOverlapping())
			throw new DMLCompressionException("Invalid to extract CompressionScheme from an overlapping compression");

		List<AColGroup> gs = cmb.getColGroups();

		ICLAScheme[] ret = new ICLAScheme[gs.size()];

		for(int i = 0; i < gs.size(); i++)
			ret[i] = gs.get(i).getCompressionScheme();

		return new CompressionScheme(ret);
	}

	public CompressedMatrixBlock updateAndEncode(MatrixBlock mb, int k) {
		if(k == 1)
			return updateAndEncode(mb);

		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final int nCol = mb.getNumColumns();
			AColGroup[] ret = new AColGroup[encodings.length];
			List<UpdateAndEncodeTask> tasks = new ArrayList<>();
			int taskSize = Math.max(1, encodings.length / (4 * k));
			for(int i = 0; i < encodings.length; i += taskSize)
				tasks.add(new UpdateAndEncodeTask(i, Math.min(encodings.length, i + taskSize), ret, mb));

			for(Future<Object> t : pool.invokeAll(tasks))
				t.get();

			List<AColGroup> retA = new ArrayList<>(Arrays.asList(ret));
			return new CompressedMatrixBlock(mb.getNumRows(), nCol, mb.getNonZeros(), false, retA);

		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed encoding", e);
		}
		finally {
			pool.shutdown();
		}
	}

	public CompressedMatrixBlock updateAndEncode(MatrixBlock mb) {
		if(mb instanceof CompressedMatrixBlock)
			throw new NotImplementedException(
				"Not implemented schema encode/apply on an already compressed MatrixBlock");

		List<AColGroup> ret = new ArrayList<>(encodings.length);

		for(int i = 0; i < encodings.length; i++) {
			Pair<ICLAScheme, AColGroup> p = encodings[i].updateAndEncode(mb);
			encodings[i] = p.getKey();
			ret.add(p.getValue());
		}

		return new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns(), mb.getNonZeros(), false, ret);

	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("\n");
		sb.append(Arrays.toString(encodings));
		return sb.toString();
	}

	protected class EncodeTask implements Callable<AColGroup> {
		final ICLAScheme enc;
		final MatrixBlock mb;

		protected EncodeTask(ICLAScheme enc, MatrixBlock mb) {
			this.enc = enc;
			this.mb = mb;
		}

		@Override
		public AColGroup call() throws Exception {
			return enc.encode(mb);
		}
	}

	protected class UpdateTask implements Callable<ICLAScheme> {
		final ICLAScheme enc;
		final MatrixBlock mb;

		protected UpdateTask(ICLAScheme enc, MatrixBlock mb) {
			this.enc = enc;
			this.mb = mb;
		}

		@Override
		public ICLAScheme call() throws Exception {
			return enc.update(mb);
		}
	}

	protected class UpdateAndEncodeTask implements Callable<Object> {
		final int i;
		final int e;
		final MatrixBlock mb;
		final AColGroup[] ret;

		protected UpdateAndEncodeTask(int i, int e, AColGroup[] ret, MatrixBlock mb) {
			this.i = i;
			this.e = e;
			this.mb = mb;
			this.ret = ret;
		}

		@Override
		public Object call() throws Exception {

			for(int j = i; j < e; j++) {
				Pair<ICLAScheme, AColGroup> p = encodings[j].updateAndEncode(mb);
				encodings[j] = p.getKey();
				ret[j] = p.getValue();
			}
			return null;
		}
	}
}

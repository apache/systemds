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

package org.apache.sysds.runtime.frame.data.lib;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

public class FrameFromMatrixBlock {
	protected static final Log LOG = LogFactory.getLog(FrameFromMatrixBlock.class.getName());

	private final MatrixBlock mb;
	private final ValueType[] schema;
	private final FrameBlock frame;

	private final int blocksizeIJ = 32; // blocks of a/c+overhead in L1 cache
	private final int blocksizeParallel = 64; // block size for each task

	private final int m;
	private final int n;

	/** Parallelization degree */
	private final int k;
	private final ExecutorService pool;

	private FrameFromMatrixBlock(MatrixBlock mb, ValueType[] schema, int k) {
		this.mb = mb;
		m = mb.getNumRows();
		n = mb.getNumColumns();

		this.schema = schema == null ? getSchema(mb) : schema;
		this.frame = new FrameBlock(this.schema);
		this.k = k;
		if(k > 1)
			pool = CommonThreadPool.get(k);
		else
			pool = null;
	}

	public static FrameBlock convertToFrameBlock(MatrixBlock mb, int k) {
		return new FrameFromMatrixBlock(mb, null, k).apply();
	}

	public static FrameBlock convertToFrameBlock(MatrixBlock mb, ValueType vt, int k) {
		return new FrameFromMatrixBlock(mb, UtilFunctions.nCopies(mb.getNumColumns(), vt), k).apply();
	}

	public static FrameBlock convertToFrameBlock(MatrixBlock mb, ValueType[] schema, int k) {
		return new FrameFromMatrixBlock(mb, schema, k).apply();
	}

	private ValueType[] getSchema(MatrixBlock mb) {
		final int nCol = mb.getNumColumns();
		final int nRow = mb.getNumRows();
		ValueType[] schema = UtilFunctions.nCopies(nCol, ValueType.BITSET);
		for(int r = 0; r < nRow; r++)
			for(int c = 0; c < nCol; c++)
				schema[c] = FrameUtil.isType(mb.quickGetValue(r, c), schema[c]);

		return schema;
	}

	private FrameBlock apply() {
		try {

			if(mb.isEmpty())
				convertToEmptyFrameBlock();
			else if(mb.isInSparseFormat())
				convertToFrameBlockSparse();
			else
				convertToFrameBlockDense();
			if(pool != null)
				pool.shutdown();
			if(frame.getNumRows() != mb.getNumRows())
				throw new DMLRuntimeException("Invalid result");

			return frame;
		}
		catch(InterruptedException | ExecutionException e) {
			pool.shutdown();
			throw new DMLRuntimeException("failed to convert to matrix block");
		}
	}

	private void convertToEmptyFrameBlock() {
		frame.ensureAllocatedColumns(mb.getNumRows());
	}

	private void convertToFrameBlockSparse() {
		SparseBlock sblock = mb.getSparseBlock();
		Array<?>[] columns = new Array<?>[mb.getNumColumns()];
		for(int i = 0; i < columns.length; i++)
			columns[i] = ArrayFactory.allocate(schema[i], mb.getNumRows());

		for(int i = 0; i < mb.getNumRows(); i++) {
			if(sblock.isEmpty(i))
				continue;

			int apos = sblock.pos(i);
			int alen = sblock.size(i);
			int[] aix = sblock.indexes(i);
			double[] aval = sblock.values(i);
			for(int j = apos; j < apos + alen; j++)
				columns[aix[j]].set(i, aval[j]);

		}
		frame.reset();
		for(int i = 0; i < columns.length; i++)
			frame.appendColumn(columns[i]);
	}

	private void convertToFrameBlockDense() throws InterruptedException, ExecutionException {
		// the frequency of double type
		final int dFreq = UtilFunctions.frequency(schema, ValueType.FP64);

		if(schema.length == 1) {
			if(dFreq == 1)
				convertToFrameDenseSingleColDouble();
			else
				convertToFrameDenseSingleColOther(schema[0]);
		}
		else if(dFreq == schema.length)
			convertToFrameDenseMultiColDouble();
		else
			convertToFrameDenseMultiColGeneric();

	}

	private void convertToFrameDenseSingleColDouble() {
		frame.reset();
		frame.appendColumn(mb.getDenseBlockValues());
	}

	private void convertToFrameDenseSingleColOther(ValueType vt) {
		Array<?> d = ArrayFactory.create(mb.getDenseBlockValues());
		frame.reset();
		frame.appendColumn(d.changeType(vt));
	}

	private void convertToFrameDenseMultiColDouble() throws InterruptedException, ExecutionException {
		double[][] c = (mb.getDenseBlock().isContiguous()) //
			? convertToFrameDenseMultiColContiguous() //
			: convertToFrameDenseMultiColMultiBlock();

		frame.reset();
		frame.appendColumns(c);
	}

	private double[][] convertToFrameDenseMultiColContiguous() throws InterruptedException, ExecutionException {
		return k == 1 //
			? convertToFrameDenseMultiColContiguousSingleThread() //
			: convertToFrameDenseMultiColContiguousMultiThread();
	}

	private double[][] convertToFrameDenseMultiColContiguousSingleThread() {

		final double[][] c = new double[n][m];
		final double[] a = mb.getDenseBlockValues();
		for(int bi = 0; bi < m; bi += blocksizeIJ) {
			for(int bj = 0; bj < n; bj += blocksizeIJ) {
				int bimin = Math.min(bi + blocksizeIJ, m);
				int bjmin = Math.min(bj + blocksizeIJ, n);
				for(int i = bi, aix = bi * n; i < bimin; i++, aix += n)
					for(int j = bj; j < bjmin; j++)
						c[j][i] = a[aix + j];
			}
		}
		return c;
	}

	private double[][] convertToFrameDenseMultiColContiguousMultiThread()
		throws InterruptedException, ExecutionException {

		final double[][] c = new double[n][m];

		final List<CVB> tasks = new ArrayList<>();
		for(int bi = 0; bi < m; bi += blocksizeParallel)
			for(int bj = 0; bj < n; bj += blocksizeParallel)
				tasks.add(new CVB(bi, bj, c));

		for(Future<Object> rt : pool.invokeAll(tasks))
			rt.get();
		return c;

	}

	protected class CVB implements Callable<Object> {

		private final int bi;
		private final int bj;
		private final double[][] c;

		protected CVB(int bi, int bj, double[][] c) {
			this.bi = bi;
			this.bj = bj;
			this.c = c;
		}

		@Override
		public Object call() {
			final double[] a = mb.getDenseBlockValues();
			int bimin = Math.min(bi + blocksizeParallel, m);
			int bjmin = Math.min(bj + blocksizeParallel, n);
			for(int i = bi, aix = bi * n; i < bimin; i++, aix += n)
				for(int j = bj; j < bjmin; j++)
					c[j][i] = a[aix + j];
			return null;
		}
	}

	private double[][] convertToFrameDenseMultiColMultiBlock() throws InterruptedException, ExecutionException {
		return k == 1 //
			? convertToFrameDenseMultiColMultiBlockSingleThread() //
			: convertToFrameDenseMultiColMultiBlockMultiThread();//
	}

	private double[][] convertToFrameDenseMultiColMultiBlockSingleThread() {

		final double[][] c = new double[n][m];
		final DenseBlock a = mb.getDenseBlock();
		for(int bi = 0; bi < m; bi += blocksizeIJ) {
			for(int bj = 0; bj < n; bj += blocksizeIJ) {
				int bimin = Math.min(bi + blocksizeIJ, m);
				int bjmin = Math.min(bj + blocksizeIJ, n);
				for(int i = bi; i < bimin; i++) {
					double[] avals = a.values(i);
					int apos = a.pos(i);
					for(int j = bj; j < bjmin; j++)
						c[j][i] = avals[apos + j];
				}
			}
		}
		return c;
	}

	private double[][] convertToFrameDenseMultiColMultiBlockMultiThread()
		throws InterruptedException, ExecutionException {

		final double[][] c = new double[n][m];

		final List<CVMB> tasks = new ArrayList<>();
		for(int bi = 0; bi < m; bi += blocksizeParallel)
			for(int bj = 0; bj < n; bj += blocksizeParallel)
				tasks.add(new CVMB(bi, bj, c));

		for(Future<Object> rt : pool.invokeAll(tasks))
			rt.get();
		return c;
	}

	protected class CVMB implements Callable<Object> {

		private final int bi;
		private final int bj;
		private final double[][] c;

		protected CVMB(int bi, int bj, double[][] c) {
			this.bi = bi;
			this.bj = bj;
			this.c = c;
		}

		@Override
		public Object call() {
			final DenseBlock a = mb.getDenseBlock();
			int bimin = Math.min(bi + blocksizeParallel, m);
			int bjmin = Math.min(bj + blocksizeParallel, n);
			for(int i = bi; i < bimin; i++) {
				double[] avals = a.values(i);
				int apos = a.pos(i);
				for(int j = bj; j < bjmin; j++)
					c[j][i] = avals[apos + j];
			}
			return null;
		}
	}

	private void convertToFrameDenseMultiColGeneric() throws InterruptedException, ExecutionException {
		Array<?>[] c = (mb.getDenseBlock().isContiguous()) //
			? convertToFrameDenseMultiColGenericContiguous() //
			: convertToFrameDenseMultiColGenericMultiBlock();

		frame.reset();
		for(Array<?> col : c)
			frame.appendColumn(col);
	}

	private Array<?>[] convertToFrameDenseMultiColGenericContiguous() throws InterruptedException, ExecutionException {
		return k == 1 //
			? convertToFrameDenseMultiColGenericContiguousSingleThread() //
			: convertToFrameDenseMultiColGenericContiguousMultiThread();//
	}

	private Array<?>[] convertToFrameDenseMultiColGenericContiguousSingleThread() {

		final Array<?>[] c = new Array<?>[n];

		for(int i = 0; i < n; i++)
			c[i] = ArrayFactory.allocate(schema[i], m);

		final double[] a = mb.getDenseBlockValues();
		for(int bi = 0; bi < m; bi += blocksizeIJ) {
			for(int bj = 0; bj < n; bj += blocksizeIJ) {
				int bimin = Math.min(bi + blocksizeIJ, m);
				int bjmin = Math.min(bj + blocksizeIJ, n);
				for(int i = bi, aix = bi * n; i < bimin; i++, aix += n)
					for(int j = bj; j < bjmin; j++)
						c[j].set(i, a[aix + j]);
			}
		}
		return c;

	}

	private Array<?>[] convertToFrameDenseMultiColGenericContiguousMultiThread()
		throws InterruptedException, ExecutionException {

		final Array<?>[] c = new Array<?>[n];

		for(int i = 0; i < n; i++)
			c[i] = ArrayFactory.allocate(schema[i], m);

		final List<CVTAB> tasks = new ArrayList<>();
		for(int bi = 0; bi < m; bi += blocksizeParallel)
			for(int bj = 0; bj < n; bj += blocksizeParallel)
				tasks.add(new CVTAB(bi, bj, c));

		for(Future<Object> rt : pool.invokeAll(tasks))
			rt.get();
		return c;
	}

	protected class CVTAB implements Callable<Object> {

		private final int bi;
		private final int bj;
		private final Array<?>[] c;

		protected CVTAB(int bi, int bj, Array<?>[] c) {
			this.bi = bi;
			this.bj = bj;
			this.c = c;
		}

		@Override
		public Object call() {
			final double[] a = mb.getDenseBlockValues();
			int bimin = Math.min(bi + blocksizeParallel, m);
			int bjmin = Math.min(bj + blocksizeParallel, n);
			for(int i = bi, aix = bi * n; i < bimin; i++, aix += n)
				for(int j = bj; j < bjmin; j++)
					c[j].set(i, a[aix + j]);
			return null;
		}
	}

	private Array<?>[] convertToFrameDenseMultiColGenericMultiBlock() {
		final Array<?>[] c = new Array<?>[n];
		for(int i = 0; i < n; i++)
			c[i] = ArrayFactory.allocate(schema[i], m);

		final DenseBlock a = mb.getDenseBlock();
		for(int bi = 0; bi < m; bi += blocksizeIJ) {
			for(int bj = 0; bj < n; bj += blocksizeIJ) {
				final int bimin = Math.min(bi + blocksizeIJ, m);
				final int bjmin = Math.min(bj + blocksizeIJ, n);
				for(int i = bi; i < bimin; i++) {
					final double[] avals = a.values(i);
					int apos = a.pos(i);
					for(int j = bj; j < bjmin; j++)
						c[j].set(i, avals[apos + j]);
				}
			}
		}
		return c;
	}

}

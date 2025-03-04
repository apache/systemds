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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroupCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.IContainDefaultTuple;
import org.apache.sysds.runtime.compress.colgroup.IFrameOfReferenceGroup;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IIterate;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.encoding.ConstEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.DenseEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.EmptyEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.estim.encoding.SparseEncoding;
import org.apache.sysds.runtime.compress.utils.HashMapLongInt;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Library functions to combine column groups inside a compressed matrix.
 */
public final class CLALibCombineGroups {
	protected static final Log LOG = LogFactory.getLog(CLALibCombineGroups.class.getName());

	private CLALibCombineGroups() {
		// private constructor
	}

	public static List<AColGroup> combine(CompressedMatrixBlock cmb, CompressedSizeInfo csi, ExecutorService pool, int k)
		throws InterruptedException, ExecutionException {
		if(pool == null || csi.getInfo().size() == 1)
			return combineSingle(cmb, csi, pool, k);
		else
			return combineParallel(cmb, csi, pool);
	}

	private static List<AColGroup> combineSingle(CompressedMatrixBlock cmb, CompressedSizeInfo csi, ExecutorService pool,
		int k) throws InterruptedException, ExecutionException {
		List<AColGroup> input = cmb.getColGroups();
		final int nRow = cmb.getNumRows();
		final boolean filterFor = CLALibUtils.shouldFilterFOR(input);
		double[] c = filterFor ? new double[cmb.getNumColumns()] : null;
		if(filterFor)
			input = CLALibUtils.filterFOR(input, c);

		final List<CompressedSizeInfoColGroup> csiI = csi.getInfo();
		csiI.sort((a, b) -> a.getNumVals() < b.getNumVals() ? -1 : a.getNumVals() == b.getNumVals() ? 0 : 1);
		final List<AColGroup> ret = new ArrayList<>(csiI.size());
		for(CompressedSizeInfoColGroup gi : csiI) {
			List<AColGroup> groupsToCombine = findGroupsInIndex(gi.getColumns(), input);
			AColGroup combined = combineN(groupsToCombine, nRow, pool, k - csiI.size());
			combined = combined.morph(gi.getBestCompressionType(), nRow);
			combined = filterFor ? combined.addVector(c) : combined;
			ret.add(combined);
		}

		return ret;
	}

	private static List<AColGroup> combineParallel(CompressedMatrixBlock cmb, CompressedSizeInfo csi,
		ExecutorService pool) throws InterruptedException, ExecutionException {
		List<AColGroup> input = cmb.getColGroups();
		final int nRow = cmb.getNumRows();
		final boolean filterFor = CLALibUtils.shouldFilterFOR(input);
		double[] c = filterFor ? new double[cmb.getNumColumns()] : null;
		if(filterFor)
			input = CLALibUtils.filterFOR(input, c);

		final List<AColGroup> filteredGroups = input;
		final List<CompressedSizeInfoColGroup> csiI = csi.getInfo();
		final List<Future<AColGroup>> tasks = new ArrayList<>();
		for(CompressedSizeInfoColGroup gi : csiI) {
			Future<AColGroup> fcg = pool.submit(() -> {
				List<AColGroup> groupsToCombine = findGroupsInIndex(gi.getColumns(), filteredGroups);
				AColGroup combined = combineN(groupsToCombine, nRow, pool, nRow);
				combined = combined.morph(gi.getBestCompressionType(), nRow);
				combined = filterFor ? combined.addVector(c) : combined;
				return combined;
			});

			tasks.add(fcg);

		}
		final List<AColGroup> ret = new ArrayList<>(csiI.size());

		for(Future<AColGroup> fcg : tasks) {
			ret.add(fcg.get());
		}

		return ret;
	}

	public static List<AColGroup> findGroupsInIndex(IColIndex idx, List<AColGroup> groups) {
		List<AColGroup> ret = new ArrayList<>();
		for(AColGroup g : groups)
			if(g.getColIndices().containsAny(idx))
				ret.add(g);

		return ret;
	}

	public static AColGroup combineN(List<AColGroup> groups, int nRows, ExecutorService pool, int k)
		throws InterruptedException, ExecutionException {
		if(k > 0 && groups.size() > 3) {
			// parallelize over a tree.
			return combineNMergeTree(groups, nRows, pool, k);
		}
		else {
			return combineNSingleAtATime(groups, nRows);
		}
	}

	@SuppressWarnings("unchecked")
	public static AColGroup combineNMergeTree(List<AColGroup> groups, int nRows, ExecutorService pool, int k)
		throws InterruptedException, ExecutionException {

		Future<AColGroup>[] tree = (Future<AColGroup>[]) new Future[groups.size() / 2 + groups.size() % 2];

		// AColGroup base = groups.get(0);
		// Inefficient combine N but base line
		for(int i = 0; i < groups.size(); i += 2) {
			final int c1 = i;
			final int c2 = i + 1;
			tree[i / 2] = pool.submit(() -> combine(groups.get(c1), groups.get(c2), nRows));
		}
		if(groups.size() % 2 != 0) {
			tree[tree.length - 1] = pool.submit(() -> groups.get(groups.size() - 1));
		}

		while(tree.length > 1) {
			final Future<AColGroup>[] treeTmp = (Future<AColGroup>[]) new Future[tree.length / 2 + tree.length % 2];
			final Future<AColGroup>[] curTree = tree;
			for(int i = 0; i < curTree.length; i += 2) {
				final int c1 = i;
				final int c2 = i + 1;
				treeTmp[i / 2] = pool.submit(() -> combine(curTree[c1].get(), curTree[c2].get(), nRows));
			}
			if(curTree.length % 2 != 0) {
				treeTmp[treeTmp.length - 1] = curTree[curTree.length - 1];
			}
			tree = treeTmp;
		}

		return tree[0].get();
	}

	public static AColGroup combineNSingleAtATime(List<AColGroup> groups, int nRows) {
		AColGroup base = groups.get(0);
		// Inefficient combine N but base line
		for(int i = 1; i < groups.size(); i++)
			base = combine(base, groups.get(i), nRows);
		return base;
	}

	/**
	 * Combine the column groups A and B together.
	 * 
	 * The number of rows should be equal, and it is not verified so there will be unexpected behavior in such cases.
	 * 
	 * It is assumed that this method is not called with FOR groups
	 * 
	 * @param a    The first group to combine.
	 * @param b    The second group to combine.
	 * @param nRow The number of rows in the two groups.
	 * @return A new column group containing the two.
	 */
	public static AColGroup combine(AColGroup a, AColGroup b, int nRow) {

		if(a instanceof IFrameOfReferenceGroup || b instanceof IFrameOfReferenceGroup)
			throw new DMLCompressionException("Invalid call with frame of reference group to combine");

		IColIndex combinedColumns = ColIndexFactory.combine(a, b);

		// try to recompress a and b if uncompressed
		if( (a instanceof ColGroupUncompressed) && (b instanceof ColGroupUncompressed)){
			// do not try to compress if both are uncompressed
		}
		else if(a instanceof ColGroupUncompressed)
			a = a.recompress();
		else if(b instanceof ColGroupUncompressed)
			b = b.recompress();

		long maxEst = (long) a.getNumValues() * b.getNumValues();
		final AColGroup ret;
		if(a instanceof AColGroupCompressed && b instanceof AColGroupCompressed //
			&& (long) Integer.MAX_VALUE > maxEst)
			ret = combineCompressed(combinedColumns, (AColGroupCompressed) a, (AColGroupCompressed) b);
		else
			ret = combineUC(combinedColumns, a, b);

		try {
			if(!CompressedMatrixBlock.debug)
				return ret;
			double sumCombined = ret.getSum(nRow);
			double sumIndividualA = a.getSum(nRow);
			double sumIndividualB = b.getSum(nRow);
			if(Math.abs(sumCombined - sumIndividualA - sumIndividualB) > Math
				.abs(((sumIndividualA + sumIndividualB) / 1000000))) {
				throw new DMLCompressionException(
					"Invalid combine... not producing same sum: " + sumCombined + " vs  " + sumIndividualA + " : "
						+ sumIndividualB + "  abs error: " + Math.abs(sumCombined - sumIndividualA - sumIndividualB));
			}
			return ret;
		}
		catch(NotImplementedException e) {
			throw e;
		}
		catch(Exception e) {
			StringBuilder sb = new StringBuilder();
			sb.append("Failed to combine:\n\n");
			String as = a.toString();
			if(as.length() < 10000)
				sb.append(as);
			else {
				sb.append(as.substring(0, 10000));
				sb.append("...");
			}
			sb.append("\n\n");

			String bs = b.toString();
			if(bs.length() < 10000)
				sb.append(bs);
			else {
				sb.append(bs.substring(0, 10000));
				sb.append("...");
			}

			String rets = ret.toString();
			if(rets.length() < 10000)
				sb.append(rets);
			else {
				sb.append(rets.substring(0, 10000));
				sb.append("...");
			}

			throw new DMLCompressionException(sb.toString(), e);
		}

	}

	private static AColGroup combineCompressed(IColIndex combinedColumns, AColGroupCompressed ac,
		AColGroupCompressed bc) {
		final IEncode ae = ac.getEncoding();
		final IEncode be = bc.getEncoding();

		final Pair<IEncode, HashMapLongInt> cec = ae.combineWithMap(be);
		final IEncode ce = cec.getLeft();
		final HashMapLongInt filter = cec.getRight();

		if(ce instanceof EmptyEncoding) {
			return new ColGroupEmpty(combinedColumns);
		}
		else if(ce instanceof ConstEncoding) {
			IDictionary cd = DictionaryFactory.combineDictionaries(ac, bc, filter);
			return ColGroupConst.create(combinedColumns, cd);
		}
		else if(ce instanceof DenseEncoding) {
			DenseEncoding ced = (DenseEncoding) (ce);
			IDictionary cd = DictionaryFactory.combineDictionaries(ac, bc, filter);
			return ColGroupDDC.create(combinedColumns, cd, ced.getMap(), null);
		}
		else if(ce instanceof SparseEncoding) {
			SparseEncoding sed = (SparseEncoding) ce;
			IDictionary cd = DictionaryFactory.combineDictionariesSparse(ac, bc, filter);
			double[] defaultTuple = constructDefaultTuple(ac, bc);
			return ColGroupSDC.create(combinedColumns, sed.getNumRows(), cd, defaultTuple, sed.getOffsets(), sed.getMap(),
				null);
		}

		throw new NotImplementedException(
			"Not implemented combine for " + ac.getClass().getSimpleName() + " - " + bc.getClass().getSimpleName());

	}

	private static AColGroup combineUC(IColIndex combineColumns, AColGroup a, AColGroup b) {
		int nRow = 0;
		if(a instanceof ColGroupUncompressed) {
			nRow = ((ColGroupUncompressed) a).getData().getNumRows();
		}
		else if(b instanceof ColGroupUncompressed) {
			nRow = ((ColGroupUncompressed) b).getData().getNumRows();
		}
		else if(a instanceof ColGroupDDC) {
			nRow = ((ColGroupDDC) a).getMapToData().size();
		}
		else if(b instanceof ColGroupDDC) {
			nRow = ((ColGroupDDC) b).getMapToData().size();
		}
		else
			throw new NotImplementedException();

		return combineUC(combineColumns, a, b, nRow);
	}

	private static AColGroup combineUC(IColIndex combinedColumns, AColGroup a, AColGroup b, int nRow) {
		double sparsityCombined = (a.getSparsity() * a.getNumCols() + b.getSparsity() * b.getNumCols()) /
			combinedColumns.size();

		if(sparsityCombined < 0.4)
			return combineUCSparse(combinedColumns, a, b, nRow);
		else
			return combineUCDense(combinedColumns, a, b, nRow);
	}

	private static AColGroup combineUCSparse(IColIndex combinedColumns, AColGroup a, AColGroup b, int nRow) {
		MatrixBlock target = new MatrixBlock(nRow, combinedColumns.size(), true);
		target.allocateBlock();

		SparseBlock db = target.getSparseBlock();

		IColIndex aTempCols = ColIndexFactory.getColumnMapping(combinedColumns, a.getColIndices());
		a.copyAndSet(aTempCols).decompressToSparseBlock(db, 0, nRow, 0, 0);
		IColIndex bTempCols = ColIndexFactory.getColumnMapping(combinedColumns, b.getColIndices());
		b.copyAndSet(bTempCols).decompressToSparseBlock(db, 0, nRow, 0, 0);

		target.recomputeNonZeros();

		return ColGroupUncompressed.create(combinedColumns, target, false);
	}

	private static AColGroup combineUCDense(IColIndex combinedColumns, AColGroup a, AColGroup b, int nRow) {
		MatrixBlock target = new MatrixBlock(nRow, combinedColumns.size(), false);
		target.allocateBlock();
		DenseBlock db = target.getDenseBlock();

		IColIndex aTempCols = ColIndexFactory.getColumnMapping(combinedColumns, a.getColIndices());
		a.copyAndSet(aTempCols).decompressToDenseBlock(db, 0, nRow, 0, 0);
		IColIndex bTempCols = ColIndexFactory.getColumnMapping(combinedColumns, b.getColIndices());
		b.copyAndSet(bTempCols).decompressToDenseBlock(db, 0, nRow, 0, 0);

		target.recomputeNonZeros();

		return ColGroupUncompressed.create(combinedColumns, target, false);
	}

	public static double[] constructDefaultTuple(AColGroupCompressed ac, AColGroupCompressed bc) {
		final double[] ret = new double[ac.getNumCols() + bc.getNumCols()];
		final IIterate ai = ac.getColIndices().iterator();
		final IIterate bi = bc.getColIndices().iterator();
		final double[] defa = ((IContainDefaultTuple) ac).getDefaultTuple();
		final double[] defb = ((IContainDefaultTuple) bc).getDefaultTuple();

		int i = 0;
		while(ai.hasNext() && bi.hasNext()) {
			if(ai.v() < bi.v()) {
				ret[i++] = defa[ai.i()];
				ai.next();
			}
			else {
				ret[i++] = defb[bi.i()];
				bi.next();
			}
		}

		while(ai.hasNext()) {
			ret[i++] = defa[ai.i()];
			ai.next();
		}

		while(bi.hasNext()) {
			ret[i++] = defb[bi.i()];
			bi.next();
		}

		return ret;
	}

}

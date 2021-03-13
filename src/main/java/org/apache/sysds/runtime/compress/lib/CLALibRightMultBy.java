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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibRightMultBy {
	private static final Log LOG = LogFactory.getLog(CLALibRightMultBy.class.getName());

	public static MatrixBlock rightMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k, boolean allowOverlap){
		ret =  rightMultByMatrix(m1.getColGroups(), m2, ret, k, m1.getMaxNumValues(), allowOverlap);
		ret.recomputeNonZeros();
		return ret;
	}

	private static MatrixBlock rightMultByMatrix(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		Pair<Integer, int[]> v, boolean allowOverlap) {

		if(that instanceof CompressedMatrixBlock)
			LOG.warn("Decompression Right matrix");

		that = that instanceof CompressedMatrixBlock ? ((CompressedMatrixBlock) that).decompress(k) : that;

		MatrixBlock m = rightMultByMatrixOverlapping(colGroups, that, ret, k, v);
		if(m instanceof CompressedMatrixBlock)
			if(allowOverlappingOutput(colGroups, allowOverlap))
				return m;
			else 
				return ((CompressedMatrixBlock) m).decompress(k);
		else
			return m;
		

		// 	if(allowOverlappingOutput(colGroups, allowOverlap))
		// 	return rightMultByMatrixOverlapping(colGroups, that, ret, k, v);
		// else 
		// 	return rightMultByMatrixNonOverlapping(colGroups, that, ret, k, v);
	}

	private static boolean allowOverlappingOutput(List<AColGroup> colGroups, boolean allowOverlap) {
		if(!allowOverlap) {
			LOG.debug("Not Overlapping because it is not allowed");
			return false;
		}
		int distinctCount = 0;
		for(AColGroup g : colGroups) {
			if(g instanceof ColGroupValue) {
				distinctCount += ((ColGroupValue) g).getNumValues();
			}
			else {
				LOG.debug("Not Overlapping because there is an un-compressible column group");
				return false;
			}
		}
		int rl = colGroups.get(0).getNumRows();
		boolean allow = distinctCount <= rl / 2;
		if(!allow) {
			LOG.debug("Not Allowing Overlap because of number of distinct items in compression");
		}
		return allow;

	}

	// private static MatrixBlock rightMultByMatrixNonOverlapping(List<AColGroup> colGroups, MatrixBlock that,
	// 	MatrixBlock ret, int k, Pair<Integer, int[]> v) {

	// 	int rl = colGroups.get(0).getNumRows();
	// 	int cl = that.getNumColumns();
	// 	if(ret == null)
	// 		ret = new MatrixBlock(rl, cl, false, rl * cl);
	// 	else if(!(ret.getNumColumns() == cl && ret.getNumRows() == rl && ret.isAllocated()))
	// 		ret.reset(rl, cl, false, rl * cl);
	// 	ret.allocateDenseBlock();
	// 	ret = rightMultByMatrix(colGroups, that, ret, k, v);
	// 	ret.setNonZeros(ret.getNumColumns() * ret.getNumRows());
	// 	return ret;
	// }

	private static MatrixBlock rightMultByMatrixOverlapping(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret,
		int k, Pair<Integer, int[]> v) {
		int rl = colGroups.get(0).getNumRows();
		int cl = that.getNumColumns();
		// Create an overlapping compressed Matrix Block.
		ret = new CompressedMatrixBlock(true);
		ret.setNumColumns(cl);
		ret.setNumRows(rl);
		CompressedMatrixBlock retC = (CompressedMatrixBlock) ret;
		ret = rightMultByMatrixCompressed(colGroups, that, retC, k, v);
		return ret;
	}

	// /**
	//  * Multi-threaded version of rightMultByVector.
	//  * 
	//  * @param colGroups The Column groups used int the multiplication
	//  * @param vector    matrix block vector to multiply with
	//  * @param result    matrix block result to modify in the multiplication
	//  * @param k         number of threads to use
	//  * @param v         The Precalculated counts and Maximum number of tuple entries in the column groups
	//  */
	// public static void rightMultByVector(List<AColGroup> colGroups, MatrixBlock vector, MatrixBlock result, int k,
	// 	Pair<Integer, int[]> v) {
	// 	// initialize and allocate the result
	// 	result.allocateDenseBlock();
	// 	if(k <= 1) {
	// 		rightMultByVector(colGroups, vector, result, v);
	// 		return;
	// 	}

	// 	// multi-threaded execution of all groups
	// 	try {
	// 		// ColGroupUncompressed uc = getUncompressedColGroup();

	// 		// compute uncompressed column group in parallel
	// 		// if(uc != null)
	// 		// uc.rightMultByVector(vector, result, k);

	// 		// compute remaining compressed column groups in parallel
	// 		// note: OLE needs alignment to segment size, otherwise wrong entry
	// 		ExecutorService pool = CommonThreadPool.get(k);
	// 		int rlen = colGroups.get(0).getNumRows();
	// 		int seqsz = CompressionSettings.BITMAP_BLOCK_SZ;
	// 		int blklen = (int) (Math.ceil((double) rlen / k));
	// 		blklen += (blklen % seqsz != 0) ? seqsz - blklen % seqsz : 0;

	// 		ArrayList<RightMatrixVectorMultTask> tasks = new ArrayList<>();
	// 		for(int i = 0; i < k & i * blklen < rlen; i++) {
	// 			tasks.add(new RightMatrixVectorMultTask(colGroups, vector, result, i * blklen,
	// 				Math.min((i + 1) * blklen, rlen), v));
	// 		}

	// 		List<Future<Long>> ret = pool.invokeAll(tasks);
	// 		pool.shutdown();

	// 		// error handling and nnz aggregation
	// 		long lnnz = 0;
	// 		for(Future<Long> tmp : ret)
	// 			lnnz += tmp.get();
	// 		result.setNonZeros(lnnz);
	// 	}
	// 	catch(InterruptedException | ExecutionException e) {
	// 		throw new DMLRuntimeException(e);
	// 	}
	// }

	/**
	 * Multiply this matrix block by a column vector on the right.
	 * 
	 * @param vector right-hand operand of the multiplication
	 * @param result buffer to hold the result; must have the appropriate size already
	 * @param v      The Precalculated counts and Maximum number of tuple entries in the column groups.
	 */
	// private static void rightMultByVector(List<AColGroup> colGroups, MatrixBlock vector, MatrixBlock result,
	// 	Pair<Integer, int[]> v) {

	// 	// delegate matrix-vector operation to each column group
	// 	rightMultByVector(colGroups, vector, result, 0, result.getNumRows(), v);

	// 	// post-processing
	// 	result.recomputeNonZeros();
	// }

	// private static MatrixBlock rightMultByMatrix(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
	// 	Pair<Integer, int[]> v) {

	// 	double[] retV = ret.getDenseBlockValues();

	// 	for(AColGroup grp : colGroups) {
	// 		if(grp instanceof ColGroupUncompressed) {
	// 			((ColGroupUncompressed) grp).rightMultByMatrix(that, ret, 0, ret.getNumRows());
	// 		}
	// 	}

	// 	if(k == 1) {
	// 		for(int j = 0; j < colGroups.size(); j++) {
	// 			if(colGroups.get(j) instanceof ColGroupValue) {
	// 				Pair<int[], double[]> preAggregatedB = ((ColGroupValue) colGroups.get(j)).preaggValues(
	// 					v.getRight()[j],
	// 					that,
	// 					colGroups.get(j).getValues(),
	// 					0,
	// 					that.getNumColumns(),
	// 					that.getNumColumns());
	// 				int blklenRows = CompressionSettings.BITMAP_BLOCK_SZ;
	// 				for(int n = 0; n * blklenRows < ret.getNumRows(); n++) {
	// 					colGroups.get(j).rightMultByMatrix(preAggregatedB.getLeft(),
	// 						preAggregatedB.getRight(),
	// 						retV,
	// 						that.getNumColumns(),
	// 						n * blklenRows,
	// 						Math.min((n + 1) * blklenRows, ret.getNumRows()));
	// 				}
	// 			}

	// 		}

	// 	}
	// 	else {
	// 		ExecutorService pool = CommonThreadPool.get(k);
	// 		ArrayList<RightMatrixMultTask> tasks = new ArrayList<>();

	// 		final int blkz = CompressionSettings.BITMAP_BLOCK_SZ;
	// 		// int blklenRows = blkz * 8 / ret.getNumColumns();
	// 		int blklenRows = Math.max(blkz,  ret.getNumColumns() / k);

	// 		try {
	// 			List<Future<Pair<int[], double[]>>> ag = pool.invokeAll(preAggregate(colGroups, that, that, v));
			
	// 			for(int j = 0; j * blklenRows < ret.getNumRows(); j++) {
	// 				RightMatrixMultTask rmmt = new RightMatrixMultTask(colGroups, retV, ag, v, that.getNumColumns(),
	// 					j * blklenRows, Math.min((j + 1) * blklenRows, ret.getNumRows()));
	// 				tasks.add(rmmt);
	// 			}

	// 			for(Future<Object> future : pool.invokeAll(tasks))
	// 				future.get();
	// 			pool.shutdown();
	// 		}
	// 		catch(InterruptedException | ExecutionException e) {
	// 			throw new DMLRuntimeException(e);
	// 		}
	// 	}

	// 	return ret;
	// }

	private static MatrixBlock rightMultByMatrixCompressed(List<AColGroup> colGroups, MatrixBlock that,
		CompressedMatrixBlock ret, int k, Pair<Integer, int[]> v) {

		for(AColGroup grp : colGroups) 
			if(grp instanceof ColGroupUncompressed) 
				throw new DMLCompressionException(
					"Right Mult by dense with compressed output is not efficient to do with uncompressed Compressed ColGroups and therefore not supported.");
			

		List<AColGroup> retCg = new ArrayList<>();
		if(k == 1) {
			for(int j = 0; j < colGroups.size(); j++) {
				ColGroupValue g = (ColGroupValue) colGroups.get(j);
				Pair<int[], double[]> preAggregatedB = g
					.preaggValues(v.getRight()[j], that, g.getValues(), 0, that.getNumColumns(), that.getNumColumns());
				if(preAggregatedB.getLeft().length > 0)
					retCg.add(g.copyAndSet(preAggregatedB.getLeft(), preAggregatedB.getRight()));
			}
		}
		else {
			ExecutorService pool = CommonThreadPool.get(k);

			try {
				List<Future<Pair<int[], double[]>>> ag = pool.invokeAll(preAggregate(colGroups, that, that, v));

				for(int j = 0; j < colGroups.size(); j++) {
					Pair<int[], double[]> preAggregates = ag.get(j).get();
					if(preAggregates.getLeft().length > 0)
						retCg.add(((ColGroupValue) colGroups.get(j)).copyAndSet(preAggregates.getLeft(),
							preAggregates.getRight()));
				}
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}
		}
		ret.allocateColGroupList(retCg);
		if(retCg.size() > 1){
			ret.setOverlapping(true);
		}
		ret.setNonZeros(-1);
		return ret;
	}

	private static ArrayList<RightMatrixPreAggregateTask> preAggregate(List<AColGroup> colGroups, MatrixBlock b,
		MatrixBlock that, Pair<Integer, int[]> v) {
		ArrayList<RightMatrixPreAggregateTask> preTask = new ArrayList<>(colGroups.size());
		preTask.clear();
		for(int h = 0; h < colGroups.size(); h++) {
			RightMatrixPreAggregateTask pAggT = new RightMatrixPreAggregateTask((ColGroupValue) colGroups.get(h),
				v.getRight()[h], b, colGroups.get(h).getValues(), 0, that.getNumColumns(), that.getNumColumns());
			preTask.add(pAggT);
		}
		return preTask;
	}

	// private static void rightMultByVector(List<AColGroup> groups, MatrixBlock vect, MatrixBlock ret, int rl, int ru,
	// 	Pair<Integer, int[]> v) {
	// 	// + 1 to enable containing a single 0 value in the dictionary that was not materialized.
	// 	// This is to handle the case of a DDC dictionary not materializing the zero values.
	// 	// A fine tradeoff!
	// 	ColGroupValue.setupThreadLocalMemory(v.getLeft() + 1);

	// 	// boolean cacheDDC1 = ru - rl > CompressionSettings.BITMAP_BLOCK_SZ * 2;

	// 	// process uncompressed column group (overwrites output)
	// 	// if(inclUC) {
	// 	for(AColGroup grp : groups) {
	// 		if(grp instanceof ColGroupUncompressed)
	// 			((ColGroupUncompressed) grp).rightMultByVector(vect, ret, rl, ru);
	// 	}

	// 	// process cache-conscious DDC1 groups (adds to output)

	// 	// if(cacheDDC1) {
	// 	// ArrayList<ColGroupDDC1> tmp = new ArrayList<>();
	// 	// for(ColGroup grp : groups)
	// 	// if(grp instanceof ColGroupDDC1)
	// 	// tmp.add((ColGroupDDC1) grp);
	// 	// if(!tmp.isEmpty())
	// 	// ColGroupDDC1.rightMultByVector(tmp.toArray(new ColGroupDDC1[0]), vect, ret, rl, ru);
	// 	// }
	// 	// process remaining groups (adds to output)
	// 	double[] values = ret.getDenseBlockValues();
	// 	for(AColGroup grp : groups) {
	// 		if(!(grp instanceof ColGroupUncompressed)) {
	// 			grp.rightMultByVector(vect.getDenseBlockValues(), values, rl, ru, grp.getValues());
	// 		}
	// 	}

	// 	ColGroupValue.cleanupThreadLocalMemory();

	// }

	// private static class RightMatrixMultTask implements Callable<Object> {
	// 	private final List<AColGroup> _colGroups;
	// 	private final double[] _retV;
	// 	private final List<Future<Pair<int[], double[]>>> _aggB;
	// 	private final Pair<Integer, int[]> _v;
	// 	private final int _numColumns;

	// 	private final int _rl;
	// 	private final int _ru;

	// 	protected RightMatrixMultTask(List<AColGroup> groups, double[] retV, List<Future<Pair<int[], double[]>>> aggB,
	// 		Pair<Integer, int[]> v, int numColumns, int rl, int ru) {
	// 		_colGroups = groups;
	// 		_retV = retV;
	// 		_aggB = aggB;
	// 		_v = v;
	// 		_numColumns = numColumns;
	// 		_rl = rl;
	// 		_ru = ru;
	// 	}

	// 	@Override
	// 	public Object call() {
	// 		try {
	// 			ColGroupValue.setupThreadLocalMemory((_v.getLeft() + 1));
	// 			for(int j = 0; j < _colGroups.size(); j++) {
	// 				Pair<int[], double[]> aggb = _aggB.get(j).get();
	// 				_colGroups.get(j).rightMultByMatrix(aggb.getLeft(), aggb.getRight(), _retV, _numColumns, _rl, _ru);
	// 			}
	// 			return null;
	// 		}
	// 		catch(Exception e) {
	// 			e.printStackTrace();
	// 			throw new DMLRuntimeException(e);
	// 		}
	// 	}
	// }

	private static class RightMatrixPreAggregateTask implements Callable<Pair<int[], double[]>> {
		private final ColGroupValue _colGroup;
		private final int _numVals;
		private final MatrixBlock _b;
		private final double[] _dict;

		private final int _cl;
		private final int _cu;
		private final int _cut;

		protected RightMatrixPreAggregateTask(ColGroupValue colGroup, int numVals, MatrixBlock b, double[] dict, int cl,
			int cu, int cut) {
			_colGroup = colGroup;
			_numVals = numVals;
			_b = b;
			_dict = dict;
			_cl = cl;
			_cu = cu;
			_cut = cut;
		}

		@Override
		public Pair<int[], double[]> call() {
			try {
				return _colGroup.preaggValues(_numVals, _b, _dict, _cl, _cu, _cut);
			}
			catch(Exception e) {
				LOG.error(e);
				throw new DMLRuntimeException(e);
			}
		}
	}

	// private static class RightMatrixVectorMultTask implements Callable<Long> {
	// 	private final List<AColGroup> _groups;
	// 	private final MatrixBlock _vect;
	// 	private final MatrixBlock _ret;
	// 	private final int _rl;
	// 	private final int _ru;
	// 	private final Pair<Integer, int[]> _v;

	// 	protected RightMatrixVectorMultTask(List<AColGroup> groups, MatrixBlock vect, MatrixBlock ret, int rl, int ru,
	// 		Pair<Integer, int[]> v) {
	// 		_groups = groups;
	// 		_vect = vect;
	// 		_ret = ret;
	// 		_rl = rl;
	// 		_ru = ru;
	// 		_v = v;
	// 	}

	// 	@Override
	// 	public Long call() {
	// 		try {
	// 			rightMultByVector(_groups, _vect, _ret, _rl, _ru, _v);
	// 			return _ret.recomputeNonZeros(_rl, _ru - 1, 0, 0);
	// 		}
	// 		catch(Exception e) {
	// 			LOG.error(e);
	// 			throw new DMLRuntimeException(e);
	// 		}
	// 	}
	// }
}

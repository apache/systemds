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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroupCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibCompAgg {
	private static final Log LOG = LogFactory.getLog(CLALibCompAgg.class.getName());
	private static final long MIN_PAR_AGG_THRESHOLD = 8 * 1024;

	private static ThreadLocal<MatrixBlock> memPool = new ThreadLocal<MatrixBlock>();

	public static MatrixBlock aggregateUnary(CompressedMatrixBlock inputMatrix, MatrixBlock result,
		AggregateUnaryOperator op, int blen, MatrixIndexes indexesIn, boolean inCP) {

		if(!supported(op) || inputMatrix.isEmpty()) {
			return inputMatrix.getUncompressed("Unary aggregate " + op + " not supported yet.")
				.aggregateUnaryOperations(op, result, blen, indexesIn, inCP);
		}

		final int r = inputMatrix.getNumRows();
		final int c = inputMatrix.getNumColumns();
		final List<AColGroup> colGroups = inputMatrix.getColGroups();
		final boolean requireDecompress = requireDecompression(inputMatrix, op);

		if(requireDecompress) {
			LOG.trace("Require decompression in unaryAggregate");
			// Decide if we should use the cached decompressed Version, or we should decompress the full matrix, or we
			// should decompress blocks.
			final double denseSize = MatrixBlock.estimateSizeDenseInMemory(r, c);
			final double localMaxMemory = InfrastructureAnalyzer.getLocalMaxMemory();

			if(inputMatrix.getCachedDecompressed() != null || (colGroups.size() > 5 && denseSize <= localMaxMemory / 2)) {
				MatrixBlock uc = inputMatrix.getUncompressed("UnaryAggregate, because of overlapping state");
				return uc.aggregateUnaryOperations(op, result, blen, indexesIn, inCP);
			}
		}

		// prepare output dimensions
		final CellIndex tempCellIndex = new CellIndex(-1, -1);
		op.indexFn.computeDimension(r, c, tempCellIndex);

		// initialize and allocate the result
		if(result == null)
			result = new MatrixBlock(tempCellIndex.row, tempCellIndex.column, false);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, false);

		result.allocateDenseBlock();

		AggregateUnaryOperator opm = replaceKahnOperations(op);

		if(colGroups != null) {

			fillStart(inputMatrix, result, opm);
			if(requireDecompress)
				aggOverlapping(inputMatrix, result, opm, indexesIn, inCP);
			else
				agg(inputMatrix, result, opm, blen, indexesIn, inCP);
		}

		result.recomputeNonZeros();
		if(op.aggOp.existsCorrection() && !inCP) {
			result = addCorrection(result, op);
			if(op.aggOp.increOp.fn instanceof Mean)
				result = addCellCount(result, op, r, c);
		}
		return result;
	}

	private static boolean supported(AggregateUnaryOperator op) {
		final ValueFunction fn = op.aggOp.increOp.fn;
		if(fn instanceof Builtin) {
			final BuiltinCode b = ((Builtin) fn).getBuiltinCode();
			return b == BuiltinCode.MIN || b == BuiltinCode.MAX;
		}
		else
			return fn instanceof KahanPlus || fn instanceof KahanPlusSq || fn instanceof Mean ||
				(fn instanceof Multiply && op.indexFn instanceof ReduceAll);
	}

	private static boolean requireDecompression(CompressedMatrixBlock inputMatrix, AggregateUnaryOperator op) {
		if(inputMatrix.isOverlapping()) {
			final ValueFunction fn = op.aggOp.increOp.fn;
			if(fn instanceof Builtin) {
				final BuiltinCode b = ((Builtin) fn).getBuiltinCode();
				return b == BuiltinCode.MIN || b == BuiltinCode.MAX;
			}
			else
				return fn instanceof KahanPlusSq || fn instanceof Multiply;
		}
		return false;
	}

	private static MatrixBlock addCorrection(MatrixBlock ret, AggregateUnaryOperator op) {

		switch(op.aggOp.correction) {
			case LASTCOLUMN:
				MatrixBlock resWithCorrection = new MatrixBlock(ret.getNumRows(), ret.getNumColumns() + 1, false);
				resWithCorrection.allocateDenseBlock();
				for(int i = 0; i < ret.getNumRows(); i++)
					resWithCorrection.setValue(i, 0, ret.quickGetValue(i, 0));
				return resWithCorrection;
			case LASTROW:
				resWithCorrection = new MatrixBlock(ret.getNumRows() + 1, ret.getNumColumns(), false);
				resWithCorrection.allocateDenseBlock();
				for(int i = 0; i < ret.getNumColumns(); i++)
					resWithCorrection.setValue(0, i, ret.quickGetValue(0, i));
				return resWithCorrection;
			case LASTTWOCOLUMNS:
				resWithCorrection = new MatrixBlock(ret.getNumRows(), ret.getNumColumns() + 2, false);
				resWithCorrection.allocateDenseBlock();
				for(int i = 0; i < ret.getNumRows(); i++)
					resWithCorrection.setValue(i, 0, ret.quickGetValue(i, 0));
				return resWithCorrection;
			case LASTTWOROWS:
				resWithCorrection = new MatrixBlock(ret.getNumRows() + 2, ret.getNumColumns(), false);
				resWithCorrection.allocateDenseBlock();
				for(int i = 0; i < ret.getNumColumns(); i++)
					resWithCorrection.setValue(0, i, ret.quickGetValue(0, i));
				return resWithCorrection;
			case NONE:
				return ret;
			case LASTFOURCOLUMNS:
			case LASTFOURROWS:
			case INVALID:
			default:
				throw new NotImplementedException("Not implemented corrections of more than 2");
		}

	}

	private static MatrixBlock addCellCount(MatrixBlock ret, AggregateUnaryOperator op, int nRow, int nCol) {
		if(op.indexFn instanceof ReduceAll)
			ret.setValue(0, 1, (long) nRow * (long) nCol);
		else if(op.indexFn instanceof ReduceCol)
			for(int i = 0; i < nRow; i++)
				ret.setValue(i, 1, nCol);
		else // Reduce Row
			for(int i = 0; i < nCol; i++)
				ret.setValue(1, i, nRow);
		return ret;
	}

	private static AggregateUnaryOperator replaceKahnOperations(AggregateUnaryOperator op) {
		if(op.aggOp.increOp.fn instanceof KahanPlus)
			return new AggregateUnaryOperator(new AggregateOperator(0, Plus.getPlusFnObject()), op.indexFn,
				op.getNumThreads());
		return op;
	}

	private static void agg(CompressedMatrixBlock m, MatrixBlock o, AggregateUnaryOperator op, int blen,
		MatrixIndexes indexesIn, boolean inCP) {

		int k = op.getNumThreads();
		// replace mean operation with plus.
		AggregateUnaryOperator opm = (op.aggOp.increOp.fn instanceof Mean) ? new AggregateUnaryOperator(
			new AggregateOperator(0, Plus.getPlusFnObject()), op.indexFn) : op;

		if(isValidForParallelProcessing(m, op))
			aggregateInParallel(m, o, opm, k);
		else {
			final int nRows = m.getNumRows();
			if(op.indexFn instanceof ReduceCol)
				agg(opm, m.getColGroups(), o.getDenseBlockValues(), nRows, 0, nRows, m.getNumColumns(),
					getPreAgg(opm, m.getColGroups()));
			else
				agg(opm, m.getColGroups(), o.getDenseBlockValues(), nRows, 0, nRows, m.getNumColumns(), null);
		}

		postProcessAggregate(m, o, op);

	}

	private static boolean isValidForParallelProcessing(CompressedMatrixBlock m1, AggregateUnaryOperator op) {
		return op.getNumThreads() > 1 && m1.getExactSizeOnDisk() > MIN_PAR_AGG_THRESHOLD;
	}

	private static void aggregateInParallel(CompressedMatrixBlock m1, MatrixBlock ret, AggregateUnaryOperator op,
		int k) {

		final ExecutorService pool = CommonThreadPool.get(k);
		final ArrayList<UnaryAggregateTask> tasks = new ArrayList<>();

		final int r = m1.getNumRows();
		final int c = m1.getNumColumns();
		final List<AColGroup> colGroups = m1.getColGroups();

		try {
			// compute all compressed column groups
			if(op.indexFn instanceof ReduceCol) {
				final int blkz = CompressionSettings.BITMAP_BLOCK_SZ;
				final int blklen = Math.max((int) Math.ceil((double) r / (k * 2)), blkz);
				double[][] preAgg = getPreAgg(op, colGroups);
				for(int i = 0; i < r; i += blklen)
					tasks.add(new UnaryAggregateTask(colGroups, ret, r, i, Math.min(i + blklen, r), op, c, false, preAgg));
			}
			else
				for(List<AColGroup> grp : createTaskPartition(colGroups, k))
					tasks.add(new UnaryAggregateTask(grp, ret, r, 0, r, op, c, m1.isOverlapping(), null));

			List<Future<MatrixBlock>> futures = pool.invokeAll(tasks);
			pool.shutdown();

			// aggregate partial results
			if(op.indexFn instanceof ReduceAll)
				if(op.aggOp.increOp.fn instanceof Builtin)
					aggregateResults(ret, futures, op);
				else
					sumResults(ret, futures);
			else if(op.indexFn instanceof ReduceRow && m1.isOverlapping()) {
				if(op.aggOp.increOp.fn instanceof Builtin)
					aggregateResultVectors(ret, futures, op);
				else
					sumResultVectors(ret, futures);
			}
			else
				for(Future<MatrixBlock> f : futures)
					f.get();
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException("Aggregate In parallel failed.", e);
		}
	}

	private static double[][] getPreAgg(AggregateUnaryOperator opm, List<AColGroup> groups) {
		double[][] ret = new double[groups.size()][];
		for(int i = 0; i < groups.size(); i++) {
			AColGroup g = groups.get(i);
			if(g instanceof AColGroupCompressed) {
				ret[i] = ((AColGroupCompressed) g).preAggRows(opm);
			}
		}
		return ret;
	}

	private static void sumResults(MatrixBlock ret, List<Future<MatrixBlock>> futures)
		throws InterruptedException, ExecutionException {
		double val = ret.quickGetValue(0, 0);
		for(Future<MatrixBlock> rtask : futures) {
			double tmp = rtask.get().quickGetValue(0, 0);
			val = val + tmp;
		}
		ret.quickSetValue(0, 0, val);

	}

	private static void sumResultVectors(MatrixBlock ret, List<Future<MatrixBlock>> futures)
		throws InterruptedException, ExecutionException {

		double[] retVals = ret.getDenseBlockValues();
		for(Future<MatrixBlock> rtask : futures) {
			double[] taskResult = rtask.get().getDenseBlockValues();
			for(int i = 0; i < retVals.length; i++) {
				retVals[i] += taskResult[i];
			}
		}
		ret.setNonZeros(ret.getNumColumns());
	}

	private static void aggregateResults(MatrixBlock ret, List<Future<MatrixBlock>> futures, AggregateUnaryOperator op)
		throws InterruptedException, ExecutionException {
		double val = ret.quickGetValue(0, 0);
		for(Future<MatrixBlock> rtask : futures) {
			double tmp = rtask.get().quickGetValue(0, 0);
			val = op.aggOp.increOp.fn.execute(val, tmp);
		}
		ret.quickSetValue(0, 0, val);
	}

	private static void aggregateResultVectors(MatrixBlock ret, List<Future<MatrixBlock>> futures,
		AggregateUnaryOperator op) throws InterruptedException, ExecutionException {
		double[] retVals = ret.getDenseBlockValues();
		for(Future<MatrixBlock> rtask : futures) {
			double[] taskResult = rtask.get().getDenseBlockValues();
			for(int i = 0; i < retVals.length; i++) {
				retVals[i] = op.aggOp.increOp.fn.execute(retVals[i], taskResult[i]);
			}
		}
		ret.setNonZeros(ret.getNumColumns());
	}

	private static void divideByNumberOfCellsForMean(CompressedMatrixBlock m1, MatrixBlock ret, IndexFunction idxFn) {
		if(idxFn instanceof ReduceAll)
			divideByNumberOfCellsForMeanAll(m1, ret);
		else if(idxFn instanceof ReduceCol)
			divideByNumberOfCellsForMeanRows(m1, ret);
		else // ReduceRow
			divideByNumberOfCellsForMeanCols(m1, ret);
	}

	private static void divideByNumberOfCellsForMeanRows(CompressedMatrixBlock m1, MatrixBlock ret) {
		double[] values = ret.getDenseBlockValues();
		for(int i = 0; i < m1.getNumRows(); i++)
			values[i] = values[i] / m1.getNumColumns();
	}

	private static void divideByNumberOfCellsForMeanCols(CompressedMatrixBlock m1, MatrixBlock ret) {
		double[] values = ret.getDenseBlockValues();
		for(int i = 0; i < m1.getNumColumns(); i++)
			values[i] = values[i] / m1.getNumRows();

	}

	private static void divideByNumberOfCellsForMeanAll(CompressedMatrixBlock m1, MatrixBlock ret) {
		ret.quickSetValue(0, 0, ret.quickGetValue(0, 0) / ((long) m1.getNumColumns() * (long) m1.getNumRows()));
	}

	private static void postProcessAggregate(CompressedMatrixBlock m1, MatrixBlock ret, AggregateUnaryOperator op) {
		if(op.aggOp.increOp.fn instanceof Mean)
			divideByNumberOfCellsForMean(m1, ret, op.indexFn);

	}

	private static void aggOverlapping(CompressedMatrixBlock m1, MatrixBlock ret, AggregateUnaryOperator op,
		MatrixIndexes indexesIn, boolean inCP) {
		try {
			List<Future<MatrixBlock>> rtasks = generateUnaryAggregateOverlappingFutures(m1, ret, op);
			reduceOverlappingFutures(rtasks, ret, op);
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLCompressionException("Error in Compressed Unary Aggregate", e);
		}
	}

	private static void reduceOverlappingFutures(List<Future<MatrixBlock>> rtasks, MatrixBlock ret,
		AggregateUnaryOperator op) throws InterruptedException, ExecutionException {
		if(isReduceAll(ret, op.indexFn))
			reduceAllOverlappingFutures(rtasks, ret, op);
		else if(op.indexFn instanceof ReduceRow) {
			final BinaryOperator bop = (op.aggOp.increOp.fn instanceof KahanFunction) ? new BinaryOperator(
				Plus.getPlusFnObject()) : op.aggOp.increOp;
			for(Future<MatrixBlock> rtask : rtasks)
				LibMatrixBincell.bincellOpInPlace(ret, rtask.get(), bop);
		}
		else
			for(Future<MatrixBlock> rtask : rtasks)
				rtask.get();
	}

	private static boolean isReduceAll(MatrixBlock ret, IndexFunction idxFn) {
		return idxFn instanceof ReduceAll || (ret.getNumColumns() == 1 && ret.getNumRows() == 1);
	}

	private static void reduceAllOverlappingFutures(List<Future<MatrixBlock>> rtasks, MatrixBlock ret,
		AggregateUnaryOperator op) throws InterruptedException, ExecutionException {

		if(op.aggOp.increOp.fn instanceof KahanFunction) {
			KahanObject kbuff = new KahanObject(ret.quickGetValue(0, 0), 0);
			KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
			for(Future<MatrixBlock> rtask : rtasks) {
				double tmp = rtask.get().quickGetValue(0, 0);
				kplus.execute2(kbuff, tmp);
			}
			ret.quickSetValue(0, 0, kbuff._sum);
		}
		else {
			double val = ret.quickGetValue(0, 0);
			for(Future<MatrixBlock> rtask : rtasks) {
				double tmp = rtask.get().quickGetValue(0, 0);
				val = op.aggOp.increOp.fn.execute(val, tmp);
			}
			ret.quickSetValue(0, 0, val);
		}
	}

	private static List<Future<MatrixBlock>> generateUnaryAggregateOverlappingFutures(CompressedMatrixBlock m1,
		MatrixBlock ret, AggregateUnaryOperator op) throws InterruptedException {

		final ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
		final ArrayList<UAOverlappingTask> tasks = new ArrayList<>();
		final int nCol = m1.getNumColumns();
		final int nRow = m1.getNumRows();
		final int blklen = CompressionSettings.BITMAP_BLOCK_SZ / nCol * 5;
		final List<AColGroup> groups = m1.getColGroups();
		final boolean shouldFilter = CLALibUtils.shouldPreFilter(groups);
		if(shouldFilter) {
			final double[] constV = new double[nCol];
			final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(groups, constV);
			final AColGroup cRet = ColGroupFactory.genColGroupConst(constV);
			filteredGroups.add(cRet);
			for(int i = 0; i < nRow; i += blklen)
				tasks.add(new UAOverlappingTask(filteredGroups, ret, i, Math.min(i + blklen, nRow), op, nCol));
		}
		else {
			for(int i = 0; i < nRow; i += blklen)
				tasks.add(new UAOverlappingTask(groups, ret, i, Math.min(i + blklen, nRow), op, nCol));
		}

		List<Future<MatrixBlock>> futures = pool.invokeAll(tasks);
		pool.shutdown();
		return futures;
	}

	private static List<List<AColGroup>> createTaskPartition(List<AColGroup> colGroups, int k) {
		int numTasks = Math.min(k, colGroups.size());
		List<List<AColGroup>> grpParts = new ArrayList<>();
		for(int i = 0; i < numTasks; i++)
			grpParts.add(new ArrayList<>());

		int pos = 0;
		for(AColGroup grp : colGroups) {
			grpParts.get(pos).add(grp);
			pos = (pos + 1) % numTasks;
		}

		return grpParts;
	}

	private static void agg(AggregateUnaryOperator op, List<AColGroup> groups, double[] ret, int nRows, int rl, int ru,
		int numColumns, double[][] preAgg) {
		if(op.indexFn instanceof ReduceCol)
			aggRow(op, groups, ret, nRows, rl, ru, numColumns, preAgg);
		else
			aggColOrAll(op, groups, ret, nRows, rl, ru, numColumns);
	}

	private static void aggColOrAll(AggregateUnaryOperator op, List<AColGroup> groups, double[] ret, int nRows, int rl,
		int ru, int numColumns) {
		for(AColGroup grp : groups)
			grp.unaryAggregateOperations(op, ret, nRows, rl, ru);
	}

	private static void aggRow(AggregateUnaryOperator op, List<AColGroup> groups, double[] ret, int nRows, int rl,
		int ru, int numColumns, double[][] preAgg) {

		for(int i = 0; i < groups.size(); i++) {
			AColGroup grp = groups.get(i);
			if(grp instanceof AColGroupCompressed)
				((AColGroupCompressed) grp).unaryAggregateOperations(op, ret, nRows, rl, ru, preAgg[i]);
			else
				grp.unaryAggregateOperations(op, ret, nRows, rl, ru);
		}

	}

	private static void fillStart(MatrixBlock in, MatrixBlock ret, AggregateUnaryOperator op) {
		final ValueFunction fn = op.aggOp.increOp.fn;
		if(fn instanceof Builtin) {
			Double val = null;
			switch(((Builtin) fn).getBuiltinCode()) {
				case MAX:
					val = Double.NEGATIVE_INFINITY;
					break;
				case MIN:
					val = Double.POSITIVE_INFINITY;
					break;
				default:
					break;
			}
			if(val != null) {
				ret.getDenseBlock().set(val);
			}
		}
		if(fn instanceof Multiply) {
			long nnz = in.getNonZeros();
			long nc = (long) in.getNumRows() * in.getNumColumns();
			boolean containsZero = nnz != nc;
			if(op.indexFn instanceof ReduceAll)
				ret.setValue(0, 0, containsZero ? 0 : 1);
			else
				throw new NotImplementedException();
		}
	}

	protected static MatrixBlock genTmpReduceAllOrRow(MatrixBlock ret, AggregateUnaryOperator op) {
		final int c = ret.getNumColumns();
		MatrixBlock tmp = new MatrixBlock(1, c, false);
		tmp.allocateDenseBlock();
		if(op.aggOp.increOp.fn instanceof Builtin)
			System.arraycopy(ret.getDenseBlockValues(), 0, tmp.getDenseBlockValues(), 0, c);
		return tmp;
	}

	private static class UnaryAggregateTask implements Callable<MatrixBlock> {
		private final List<AColGroup> _groups;
		private final int _nRows;
		private final int _rl;
		private final int _ru;
		private final MatrixBlock _ret;
		private final int _numColumns;
		private final AggregateUnaryOperator _op;
		private final double[][] _preAgg;

		protected UnaryAggregateTask(List<AColGroup> groups, MatrixBlock ret, int nRows, int rl, int ru,
			AggregateUnaryOperator op, int numColumns, boolean overlapping, double[][] preAgg) {
			_groups = groups;
			_op = op;
			_nRows = nRows;
			_rl = rl;
			_ru = ru;
			_numColumns = numColumns;
			_preAgg = preAgg;
			if(_op.indexFn instanceof ReduceAll || (_op.indexFn instanceof ReduceRow && overlapping))
				_ret = genTmpReduceAllOrRow(ret, op);
			else // colSums / rowSums not overlapping
				_ret = ret;

		}

		@Override
		public MatrixBlock call() {
			agg(_op, _groups, _ret.getDenseBlockValues(), _nRows, _rl, _ru, _numColumns, _preAgg);
			return _ret;
		}
	}

	private static class UAOverlappingTask implements Callable<MatrixBlock> {
		private final List<AColGroup> _groups;
		private final int _rl;
		private final int _ru;
		private final MatrixBlock _ret;
		private final AggregateUnaryOperator _op;
		private final int _nCol;

		protected UAOverlappingTask(List<AColGroup> filteredGroups, MatrixBlock ret, int rl, int ru,
			AggregateUnaryOperator op, int nCol) {
			_groups = filteredGroups;
			_op = op;
			_rl = rl;
			_ru = ru;
			_ret = ret;
			_nCol = nCol;
		}

		private MatrixBlock getTmp() {
			MatrixBlock tmp = memPool.get();
			if(tmp == null) {
				memPool.set(new MatrixBlock(_ru - _rl, _nCol, false, -1).allocateBlock());
				tmp = memPool.get();
			}
			else
				tmp.reset(_ru - _rl, _nCol, false, -1);

			return tmp;
		}

		private MatrixBlock decompressToTemp() {
			MatrixBlock tmp = getTmp();

			for(AColGroup g : _groups)
				g.decompressToDenseBlock(tmp.getDenseBlock(), _rl, _ru, -_rl, 0);

			tmp.setNonZeros(_rl + _ru);
			return tmp;
		}

		@Override
		public MatrixBlock call() {
			MatrixBlock tmp = decompressToTemp();
			MatrixBlock outputBlock = tmp.prepareAggregateUnaryOutput(_op, null,
				Math.max(tmp.getNumColumns(), tmp.getNumRows()));
			LibMatrixAgg.aggregateUnaryMatrix(tmp, outputBlock, _op);
			outputBlock.dropLastRowsOrColumns(_op.aggOp.correction);

			if(_op.indexFn instanceof ReduceCol) {
				if(outputBlock.isEmpty())
					return null;
				else if(outputBlock.isInSparseFormat())
					throw new DMLCompressionException("Output block should never be sparse");
				else {
					double[] retValues = _ret.getDenseBlockValues();
					int currentIndex = _rl * _ret.getNumColumns();
					double[] outputBlockValues = outputBlock.getDenseBlockValues();
					System.arraycopy(outputBlockValues, 0, retValues, currentIndex, outputBlockValues.length);
					return null;
				}
			}
			else
				return outputBlock;
		}
	}
}

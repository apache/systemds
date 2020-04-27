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

package org.apache.sysds.runtime.compress;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.random.Well1024a;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.CTableMap;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.data.RandomMatrixGenerator;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateTernaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.COVOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.QuaternaryOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.SortUtils;

public abstract class AbstractCompressedMatrixBlock extends MatrixBlock {

	private static final Log LOG = LogFactory.getLog(AbstractCompressedMatrixBlock.class.getName());

	protected List<ColGroup> _colGroups;

	/**
	 * Constructor for building an empty Compressed Matrix block object.
	 */
	public AbstractCompressedMatrixBlock() {
		super();
	}

	/**
	 * Main constructor for building a block from scratch.
	 * 
	 * @param rl     number of rows in the block
	 * @param cl     number of columns
	 * @param sparse true if the UNCOMPRESSED representation of the block should be sparse
	 */
	public AbstractCompressedMatrixBlock(int rl, int cl, boolean sparse) {
		super(rl, cl, sparse);
	}

	/**
	 * "Copy" constructor to populate this compressed block with the uncompressed contents of a conventional block. Does
	 * <b>not</b> compress the block. Only creates a shallow copy, and only does deep copy on compression.
	 * 
	 * @param that matrix block
	 */
	public AbstractCompressedMatrixBlock(MatrixBlock that) {
		super(that.getNumRows(), that.getNumColumns(), that.isInSparseFormat());

		// shallow copy (deep copy on compression, prevents unnecessary copy)
		if(isInSparseFormat())
			sparseBlock = that.getSparseBlock();
		else
			denseBlock = that.getDenseBlock();
		nonZeros = that.getNonZeros();
	}

	public abstract boolean isCompressed();

	public abstract MatrixBlock decompress();

	@Override
	public boolean isEmptyBlock(boolean safe) {
		if(!isCompressed())
			return super.isEmptyBlock(safe);
		return(_colGroups == null || getNonZeros() == 0);
	}

	public static long estimateOriginalSizeInMemory(int nrows, int ncols, double sparsity) {
		// Estimate the original Size.
		// Unlike the other Estimation this one takes the original estimation
		// but also includes the small overhead of different arrays.
		// TODO: Make the original Memory estimates better for MatrixBlocks.
		long size = MatrixBlock.estimateSizeInMemory(nrows, ncols, sparsity);

		size += 4; // rlen
		size += 4; // clen
		size += 1; // a single boolean fills 8 bits !
		size += 8; // NonZeros.

		size += 8; // Object reference DenseBlock
		size += 8; // Object reference Sparse Block

		size += 4; // estimated NNzs Per Row

		if(size % 8 != 0)
			size += 8 - size % 8; // Add padding

		return size;
	}

	//////////////////////////////////////////
	// Graceful fallback to uncompressed linear algebra

	@Override
	public MatrixBlock unaryOperations(UnaryOperator op, MatrixValue result) {
		printDecompressWarning("unaryOperations");
		MatrixBlock tmp = isCompressed() ? decompress() : this;
		return tmp.unaryOperations(op, result);
	}

	@Override
	public MatrixBlock binaryOperations(BinaryOperator op, MatrixValue thatValue, MatrixValue result) {
		printDecompressWarning("binaryOperations", (MatrixBlock) thatValue);
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(thatValue);
		return left.binaryOperations(op, right, result);
	}

	@Override
	public void binaryOperationsInPlace(BinaryOperator op, MatrixValue thatValue) {
		printDecompressWarning("binaryOperationsInPlace", (MatrixBlock) thatValue);
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(thatValue);
		left.binaryOperationsInPlace(op, right);
	}

	@Override
	public void incrementalAggregate(AggregateOperator aggOp, MatrixValue correction, MatrixValue newWithCorrection,
		boolean deep) {
		throw new DMLRuntimeException("CompressedMatrixBlock: incrementalAggregate not supported.");
	}

	@Override
	public void incrementalAggregate(AggregateOperator aggOp, MatrixValue newWithCorrection) {
		throw new DMLRuntimeException("CompressedMatrixBlock: incrementalAggregate not supported.");
	}

	@Override
	public MatrixBlock reorgOperations(ReorgOperator op, MatrixValue ret, int startRow, int startColumn, int length) {
		printDecompressWarning("reorgOperations");
		MatrixBlock tmp = isCompressed() ? decompress() : this;
		return tmp.reorgOperations(op, ret, startRow, startColumn, length);
	}

	@Override
	public MatrixBlock append(MatrixBlock that, MatrixBlock ret, boolean cbind) {
		if(cbind) // use supported operation
			return append(that, ret);
		printDecompressWarning("append-rbind", that);
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(that);
		return left.append(right, ret, cbind);
	}

	@Override
	public void append(MatrixValue v2, ArrayList<IndexedMatrixValue> outlist, int blen, boolean cbind, boolean m2IsLast,
		int nextNCol) {
		printDecompressWarning("append", (MatrixBlock) v2);
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(v2);
		left.append(right, outlist, blen, cbind, m2IsLast, nextNCol);
	}

	@Override
	public void permutationMatrixMultOperations(MatrixValue m2Val, MatrixValue out1Val, MatrixValue out2Val) {
		permutationMatrixMultOperations(m2Val, out1Val, out2Val, 1);
	}

	@Override
	public void permutationMatrixMultOperations(MatrixValue m2Val, MatrixValue out1Val, MatrixValue out2Val, int k) {
		printDecompressWarning("permutationMatrixMultOperations", (MatrixBlock) m2Val);
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(m2Val);
		left.permutationMatrixMultOperations(right, out1Val, out2Val, k);
	}

	@Override
	public MatrixBlock leftIndexingOperations(MatrixBlock rhsMatrix, int rl, int ru, int cl, int cu, MatrixBlock ret,
		UpdateType update) {
		printDecompressWarning("leftIndexingOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(rhsMatrix);
		return left.leftIndexingOperations(right, rl, ru, cl, cu, ret, update);
	}

	@Override
	public MatrixBlock leftIndexingOperations(ScalarObject scalar, int rl, int cl, MatrixBlock ret, UpdateType update) {
		printDecompressWarning("leftIndexingOperations");
		MatrixBlock tmp = isCompressed() ? decompress() : this;
		return tmp.leftIndexingOperations(scalar, rl, cl, ret, update);
	}

	@Override
	public MatrixBlock slice(int rl, int ru, int cl, int cu, CacheBlock ret) {
		printDecompressWarning("slice");
		MatrixBlock tmp = isCompressed() ? decompress() : this;
		return tmp.slice(rl, ru, cl, cu, ret);
	}

	@Override
	public void slice(ArrayList<IndexedMatrixValue> outlist, IndexRange range, int rowCut, int colCut, int blen,
		int boundaryRlen, int boundaryClen) {
		printDecompressWarning("slice");
		try {
			MatrixBlock tmp = isCompressed() ? decompress() : this;
			tmp.slice(outlist, range, rowCut, colCut, blen, boundaryRlen, boundaryClen);
		}
		catch(DMLRuntimeException ex) {
			throw new RuntimeException(ex);
		}
	}

	@Override
	public MatrixBlock zeroOutOperations(MatrixValue result, IndexRange range, boolean complementary) {
		printDecompressWarning("zeroOutOperations");
		MatrixBlock tmp = isCompressed() ? decompress() : this;
		return tmp.zeroOutOperations(result, range, complementary);
	}

	@Override
	public CM_COV_Object cmOperations(CMOperator op) {
		printDecompressWarning("cmOperations");
		if(!isCompressed() || isEmptyBlock())
			return super.cmOperations(op);
		ColGroup grp = _colGroups.get(0);
		if(grp instanceof ColGroupUncompressed)
			return ((ColGroupUncompressed) grp).getData().cmOperations(op);

		ColGroupValue grpVal = (ColGroupValue) grp;
		MatrixBlock vals = grpVal.getValuesAsBlock();
		MatrixBlock counts = ColGroupValue.getCountsAsBlock(grpVal.getCounts(true));
		return vals.cmOperations(op, counts);
	}

	@Override
	public CM_COV_Object cmOperations(CMOperator op, MatrixBlock weights) {
		printDecompressWarning("cmOperations");
		MatrixBlock right = getUncompressed(weights);
		if(!isCompressed() || isEmptyBlock())
			return super.cmOperations(op, right);
		ColGroup grp = _colGroups.get(0);
		if(grp instanceof ColGroupUncompressed)
			return ((ColGroupUncompressed) grp).getData().cmOperations(op);
		return decompress().cmOperations(op, right);
	}

	@Override
	public CM_COV_Object covOperations(COVOperator op, MatrixBlock that) {
		printDecompressWarning("covOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(that);
		return left.covOperations(op, right);
	}

	@Override
	public CM_COV_Object covOperations(COVOperator op, MatrixBlock that, MatrixBlock weights) {
		printDecompressWarning("covOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right1 = getUncompressed(that);
		MatrixBlock right2 = getUncompressed(weights);
		return left.covOperations(op, right1, right2);
	}

	@Override
	public MatrixBlock sortOperations(MatrixValue weights, MatrixBlock result) {
		printDecompressWarning("sortOperations");
		MatrixBlock right = getUncompressed(weights);
		if(!isCompressed())
			return super.sortOperations(right, result);
		ColGroup grp = _colGroups.get(0);
		if(grp instanceof ColGroupUncompressed)
			return ((ColGroupUncompressed) grp).getData().sortOperations(right, result);

		if(right == null) {
			ColGroupValue grpVal = (ColGroupValue) grp;
			MatrixBlock vals = grpVal.getValuesAsBlock();
			int[] counts = grpVal.getCounts(true);
			double[] data = (vals.getDenseBlock() != null) ? vals.getDenseBlockValues() : null;
			SortUtils.sortByValue(0, vals.getNumRows(), data, counts);
			MatrixBlock counts2 = ColGroupValue.getCountsAsBlock(counts);
			return vals.sortOperations(counts2, result);
		}
		else
			return decompress().sortOperations(right, result);
	}

	@Override
	public MatrixBlock aggregateBinaryOperations(MatrixIndexes m1Index, MatrixBlock m1Value, MatrixIndexes m2Index,
		MatrixBlock m2Value, MatrixBlock result, AggregateBinaryOperator op) {
		printDecompressWarning("aggregateBinaryOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(m2Value);
		return left.aggregateBinaryOperations(m1Index, left, m2Index, right, result, op);
	}

	@Override
	public MatrixBlock aggregateTernaryOperations(MatrixBlock m1, MatrixBlock m2, MatrixBlock m3, MatrixBlock ret,
		AggregateTernaryOperator op, boolean inCP) {
		printDecompressWarning("aggregateTernaryOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right1 = getUncompressed(m2);
		MatrixBlock right2 = getUncompressed(m3);
		return left.aggregateTernaryOperations(left, right1, right2, ret, op, inCP);
	}

	@Override
	public MatrixBlock uaggouterchainOperations(MatrixBlock mbLeft, MatrixBlock mbRight, MatrixBlock mbOut,
		BinaryOperator bOp, AggregateUnaryOperator uaggOp) {
		printDecompressWarning("uaggouterchainOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(mbRight);
		return left.uaggouterchainOperations(left, right, mbOut, bOp, uaggOp);
	}

	@Override
	public MatrixBlock groupedAggOperations(MatrixValue tgt, MatrixValue wghts, MatrixValue ret, int ngroups,
		Operator op) {
		return groupedAggOperations(tgt, wghts, ret, ngroups, op, 1);
	}

	@Override
	public MatrixBlock groupedAggOperations(MatrixValue tgt, MatrixValue wghts, MatrixValue ret, int ngroups,
		Operator op, int k) {
		printDecompressWarning("groupedAggOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(wghts);
		return left.groupedAggOperations(left, right, ret, ngroups, op, k);
	}

	@Override
	public MatrixBlock removeEmptyOperations(MatrixBlock ret, boolean rows, boolean emptyReturn, MatrixBlock select) {
		printDecompressWarning("removeEmptyOperations");
		MatrixBlock tmp = isCompressed() ? decompress() : this;
		return tmp.removeEmptyOperations(ret, rows, emptyReturn, select);
	}

	@Override
	public MatrixBlock removeEmptyOperations(MatrixBlock ret, boolean rows, boolean emptyReturn) {
		printDecompressWarning("removeEmptyOperations");
		MatrixBlock tmp = isCompressed() ? decompress() : this;
		return tmp.removeEmptyOperations(ret, rows, emptyReturn);
	}

	@Override
	public MatrixBlock rexpandOperations(MatrixBlock ret, double max, boolean rows, boolean cast, boolean ignore,
		int k) {
		printDecompressWarning("rexpandOperations");
		MatrixBlock tmp = isCompressed() ? decompress() : this;
		return tmp.rexpandOperations(ret, max, rows, cast, ignore, k);
	}

	@Override
	public MatrixBlock replaceOperations(MatrixValue result, double pattern, double replacement) {
		printDecompressWarning("replaceOperations");
		MatrixBlock tmp = isCompressed() ? decompress() : this;
		return tmp.replaceOperations(result, pattern, replacement);
	}

	@Override
	public void ctableOperations(Operator op, double scalar, MatrixValue that, CTableMap resultMap,
		MatrixBlock resultBlock) {
		printDecompressWarning("ctableOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(that);
		left.ctableOperations(op, scalar, right, resultMap, resultBlock);
	}

	@Override
	public void ctableOperations(Operator op, double scalar, double scalar2, CTableMap resultMap,
		MatrixBlock resultBlock) {
		printDecompressWarning("ctableOperations");
		MatrixBlock tmp = isCompressed() ? decompress() : this;
		tmp.ctableOperations(op, scalar, scalar2, resultMap, resultBlock);
	}

	@Override
	public void ctableOperations(Operator op, MatrixIndexes ix1, double scalar, boolean left, int brlen,
		CTableMap resultMap, MatrixBlock resultBlock) {
		printDecompressWarning("ctableOperations");
		MatrixBlock tmp = isCompressed() ? decompress() : this;
		tmp.ctableOperations(op, ix1, scalar, left, brlen, resultMap, resultBlock);
	}

	@Override
	public void ctableOperations(Operator op, MatrixValue that, double scalar, boolean ignoreZeros, CTableMap resultMap,
		MatrixBlock resultBlock) {
		printDecompressWarning("ctableOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right = getUncompressed(that);
		left.ctableOperations(op, right, scalar, ignoreZeros, resultMap, resultBlock);
	}

	@Override
	public MatrixBlock ctableSeqOperations(MatrixValue that, double scalar, MatrixBlock resultBlock) {
		printDecompressWarning("ctableOperations");
		MatrixBlock right = getUncompressed(that);
		return this.ctableSeqOperations(right, scalar, resultBlock);
	}

	@Override
	public void ctableOperations(Operator op, MatrixValue that, MatrixValue that2, CTableMap resultMap) {
		printDecompressWarning("ctableOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right1 = getUncompressed(that);
		MatrixBlock right2 = getUncompressed(that2);
		left.ctableOperations(op, right1, right2, resultMap);
	}

	@Override
	public void ctableOperations(Operator op, MatrixValue that, MatrixValue that2, CTableMap resultMap,
		MatrixBlock resultBlock) {
		printDecompressWarning("ctableOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right1 = getUncompressed(that);
		MatrixBlock right2 = getUncompressed(that2);
		left.ctableOperations(op, right1, right2, resultMap, resultBlock);
	}

	@Override
	public MatrixBlock ternaryOperations(TernaryOperator op, MatrixBlock m2, MatrixBlock m3, MatrixBlock ret) {
		printDecompressWarning("ternaryOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right1 = getUncompressed(m2);
		MatrixBlock right2 = getUncompressed(m3);
		return left.ternaryOperations(op, right1, right2, ret);
	}

	@Override
	public MatrixBlock quaternaryOperations(QuaternaryOperator qop, MatrixBlock um, MatrixBlock vm, MatrixBlock wm,
		MatrixBlock out) {
		return quaternaryOperations(qop, um, vm, wm, out, 1);
	}

	@Override
	public MatrixBlock quaternaryOperations(QuaternaryOperator qop, MatrixBlock um, MatrixBlock vm, MatrixBlock wm,
		MatrixBlock out, int k) {
		printDecompressWarning("quaternaryOperations");
		MatrixBlock left = isCompressed() ? decompress() : this;
		MatrixBlock right1 = getUncompressed(um);
		MatrixBlock right2 = getUncompressed(vm);
		MatrixBlock right3 = getUncompressed(wm);
		return left.quaternaryOperations(qop, right1, right2, right3, out, k);
	}

	@Override
	public MatrixBlock randOperationsInPlace(RandomMatrixGenerator rgen, Well1024a bigrand, long bSeed) {
		throw new DMLRuntimeException("CompressedMatrixBlock: randOperationsInPlace not supported.");
	}

	@Override
	public MatrixBlock randOperationsInPlace(RandomMatrixGenerator rgen, Well1024a bigrand, long bSeed, int k) {
		throw new DMLRuntimeException("CompressedMatrixBlock: randOperationsInPlace not supported.");
	}

	@Override
	public MatrixBlock seqOperationsInPlace(double from, double to, double incr) {
		// output should always be uncompressed
		throw new DMLRuntimeException("CompressedMatrixBlock: seqOperationsInPlace not supported.");
	}

	private static boolean isCompressed(MatrixBlock mb) {
		return(mb instanceof CompressedMatrixBlock && ((CompressedMatrixBlock) mb).isCompressed());
	}

	private static MatrixBlock getUncompressed(MatrixValue mVal) {
		return isCompressed((MatrixBlock) mVal) ? ((CompressedMatrixBlock) mVal).decompress() : (MatrixBlock) mVal;
	}

	private void printDecompressWarning(String operation) {
		if(isCompressed()) {
			LOG.warn("Operation '" + operation + "' not supported yet - decompressing for ULA operations.");
		}
	}

	private void printDecompressWarning(String operation, MatrixBlock m2) {
		if(isCompressed() || isCompressed(m2)) {
			LOG.warn("Operation '" + operation + "' not supported yet - decompressing for ULA operations.");
		}
	}

}
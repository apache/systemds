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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.lang.ref.SoftReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Future;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.random.Well1024a;
import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.AColGroupValue;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupIO;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.lib.CLALibAppend;
import org.apache.sysds.runtime.compress.lib.CLALibBinaryCellOp;
import org.apache.sysds.runtime.compress.lib.CLALibCompAgg;
import org.apache.sysds.runtime.compress.lib.CLALibDecompress;
import org.apache.sysds.runtime.compress.lib.CLALibLeftMultBy;
import org.apache.sysds.runtime.compress.lib.CLALibMMChain;
import org.apache.sysds.runtime.compress.lib.CLALibMatrixMult;
import org.apache.sysds.runtime.compress.lib.CLALibReExpand;
import org.apache.sysds.runtime.compress.lib.CLALibScalar;
import org.apache.sysds.runtime.compress.lib.CLALibSlice;
import org.apache.sysds.runtime.compress.lib.CLALibSquash;
import org.apache.sysds.runtime.compress.lib.CLALibUnary;
import org.apache.sysds.runtime.compress.lib.CLALibUtils;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.functionobjects.TernaryValueFunction.ValueFunctionWithConstant;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.CTableMap;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysds.runtime.matrix.data.LibMatrixTercell;
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
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.utils.DMLCompressionStatistics;

public class CompressedMatrixBlock extends MatrixBlock {
	private static final Log LOG = LogFactory.getLog(CompressedMatrixBlock.class.getName());
	private static final long serialVersionUID = 73193720143154058L;

	/**
	 * Column groups
	 */
	protected transient List<AColGroup> _colGroups;

	/**
	 * Boolean specifying if the colGroups are overlapping each other. This happens after a right matrix multiplication.
	 */
	protected boolean overlappingColGroups = false;

	/**
	 * Soft reference to a decompressed version of this matrix block.
	 */
	protected transient SoftReference<MatrixBlock> decompressedVersion;

	public CompressedMatrixBlock() {
		super(true);
		sparse = false;
		nonZeros = -1;
	}

	/**
	 * Main constructor for building a block from scratch.
	 * 
	 * Use with caution, since it constructs an empty matrix block with nothing inside.
	 * 
	 * @param rl number of rows in the block
	 * @param cl number of columns
	 */
	public CompressedMatrixBlock(int rl, int cl) {
		super(true);
		rlen = rl;
		clen = cl;
		sparse = false;
		nonZeros = -1;
	}

	/**
	 * Copy constructor taking that CompressedMatrixBlock and populate this new compressedMatrixBlock with pointers to
	 * the same columnGroups.
	 * 
	 * @param that CompressedMatrixBlock to copy values from
	 */
	public CompressedMatrixBlock(CompressedMatrixBlock that) {
		super(true);
		rlen = that.getNumRows();
		clen = that.getNumColumns();
		this.copyCompressedMatrix(that);
	}

	/**
	 * Copy constructor taking an uncompressedMatrixBlock to copy metadata from also while copying metadata, a soft
	 * reference is constructed to the uncompressed matrixBlock, to allow quick decompressions if the program is not
	 * under memory pressure.
	 * 
	 * This method is used in the CompressionFactory.
	 * 
	 * @param uncompressedMatrixBlock An uncompressed Matrix to copy metadata from.
	 */
	protected CompressedMatrixBlock(MatrixBlock uncompressedMatrixBlock) {
		super(true);
		rlen = uncompressedMatrixBlock.getNumRows();
		clen = uncompressedMatrixBlock.getNumColumns();
		sparse = false;
		nonZeros = uncompressedMatrixBlock.getNonZeros();
		decompressedVersion = new SoftReference<>(uncompressedMatrixBlock);
	}

	@Override
	public void reset(int rl, int cl, boolean sp, long estnnz, double val) {
		throw new DMLCompressionException("Invalid to reset a Compressed MatrixBlock");
	}

	/**
	 * Allocate the given column group and remove all references to old column groups.
	 * 
	 * This is done by simply allocating a ned _colGroups list and adding the given column group
	 * 
	 * @param cg The column group to use after.
	 */
	public void allocateColGroup(AColGroup cg) {
		_colGroups = new ArrayList<>(1);
		_colGroups.add(cg);
	}

	/**
	 * Replace the column groups in this CompressedMatrixBlock with the given column groups
	 * 
	 * @param colGroups new ColGroups in the MatrixBlock
	 */
	public void allocateColGroupList(List<AColGroup> colGroups) {
		_colGroups = colGroups;
	}

	/**
	 * Get the column groups of this CompressedMatrixBlock
	 * 
	 * @return the column groups
	 */
	public List<AColGroup> getColGroups() {
		return _colGroups;
	}

	/**
	 * Decompress block into a MatrixBlock
	 * 
	 * @return a new uncompressed matrix block containing the contents of this block
	 */
	public MatrixBlock decompress() {
		return decompress(1);
	}

	/**
	 * Decompress block into a MatrixBlock
	 * 
	 * @param k degree of parallelism
	 * @return a new uncompressed matrix block containing the contents of this block
	 */
	public MatrixBlock decompress(int k) {
		// Early out if empty.
		if(isEmpty())
			return new MatrixBlock(rlen, clen, true, 0);

		// Early out if decompressed version already is cached
		MatrixBlock ret = getCachedDecompressed();
		if(ret != null)
			return ret;

		ret = CLALibDecompress.decompress(this, k);

		// Set soft reference to the decompressed version
		decompressedVersion = new SoftReference<>(ret);

		return ret;
	}

	@Override
	public void putInto(MatrixBlock target, int rowOffset, int colOffset, boolean sparseCopyShallow) {
		CLALibDecompress.decompressTo(this, target, rowOffset, colOffset, 1);
	}

	/**
	 * Get the cached decompressed matrix (if it exists otherwise null).
	 * 
	 * This in practice means that if some other instruction have materialized the decompressed version it can be
	 * accessed though this method with a guarantee that it did not go through the entire decompression phase.
	 * 
	 * @return The cached decompressed matrix, if it does not exist return null
	 */
	public MatrixBlock getCachedDecompressed() {
		if(decompressedVersion != null) {
			final MatrixBlock mb = decompressedVersion.get();
			if(mb != null) {
				DMLCompressionStatistics.addDecompressCacheCount();
				LOG.trace("Decompressed block was in soft reference.");
				return mb;
			}
		}
		return null;
	}

	public CompressedMatrixBlock squash(int k) {
		return CLALibSquash.squash(this, k);
	}

	@Override
	public long recomputeNonZeros() {
		if(isOverlapping())
			nonZeros = clen * rlen;
		else {
			long nnz = 0;
			for(AColGroup g : _colGroups)
				nnz += g.getNumberNonZeros(rlen);
			nonZeros = nnz;
		}

		if(nonZeros == 0) {
			ColGroupEmpty cg = ColGroupEmpty.generate(getNumColumns());
			allocateColGroup(cg);
		}

		return nonZeros;
	}

	@Override
	public long recomputeNonZeros(int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public long recomputeNonZeros(int rl, int ru, int cl, int cu) {
		throw new NotImplementedException();
	}

	@Override
	public long getInMemorySize() {
		return estimateCompressedSizeInMemory();
	}

	@Override
	public long estimateSizeInMemory() {
		return estimateCompressedSizeInMemory();
	}

	/**
	 * Obtain an upper bound on the memory used to store the compressed block.
	 * 
	 * @return an upper bound on the memory used to store this compressed block considering class overhead.
	 */
	public long estimateCompressedSizeInMemory() {
		long total = baseSizeInMemory();

		for(AColGroup grp : _colGroups)
			total += grp.estimateInMemorySize();

		return total;
	}

	public static long baseSizeInMemory() {
		long total = 16; // Object header

		total += getHeaderSize(); // Matrix Block elements
		total += 8; // Col Group Ref
		total += 8; // v reference
		total += 8; // soft reference to decompressed version
		total += 1 + 7; // Booleans plus padding

		total += 40; // Col Group Array List
		return total;
	}

	@Override
	public double quickGetValue(int r, int c) {
		if(isOverlapping()) {
			double v = 0.0;
			for(AColGroup group : _colGroups)
				v += group.get(r, c);
			return v;
		}
		else {
			for(AColGroup group : _colGroups) {
				final int idx = Arrays.binarySearch(group.getColIndices(), c);
				if(idx >= 0)
					return group.getIdx(r, idx);
			}
			return 0;
		}
	}

	@Override
	public long getExactSizeOnDisk() {
		// header information
		long ret = 4 + 4 + 8 + 1;
		ret += ColGroupIO.getExactSizeOnDisk(_colGroups);
		return ret;
	}

	@Override
	public long estimateSizeOnDisk() {
		return getExactSizeOnDisk();
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		// deserialize compressed block
		rlen = in.readInt();
		clen = in.readInt();
		nonZeros = in.readLong();
		overlappingColGroups = in.readBoolean();
		_colGroups = ColGroupIO.readGroups(in, rlen);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		if(getExactSizeOnDisk() > MatrixBlock.estimateSizeOnDisk(rlen, clen, nonZeros)) {
			// If the size of this matrixBlock is smaller in uncompressed format, then
			// decompress and save inside an uncompressed column group.
			MatrixBlock uncompressed = getUncompressed("for smaller serialization");
			ColGroupUncompressed cg = new ColGroupUncompressed(uncompressed);
			allocateColGroup(cg);
			nonZeros = cg.getNumberNonZeros(rlen);
			// clear the soft reference to the decompressed version, since the one column group is perfectly,
			// representing the decompressed version.
			clearSoftReferenceToDecompressed();
		}
		// serialize compressed matrix block
		out.writeInt(rlen);
		out.writeInt(clen);
		out.writeLong(nonZeros);
		out.writeBoolean(overlappingColGroups);
		ColGroupIO.writeGroups(out, _colGroups);
	}

	/**
	 * Redirects the default java serialization via externalizable to our default hadoop writable serialization for
	 * efficient broadcast/rdd de-serialization.
	 * 
	 * @param is object input
	 * @throws IOException if IOException occurs
	 */
	@Override
	public void readExternal(ObjectInput is) throws IOException {
		readFields(is);
	}

	/**
	 * Redirects the default java serialization via externalizable to our default hadoop writable serialization for
	 * efficient broadcast/rdd serialization.
	 * 
	 * @param os object output
	 * @throws IOException if IOException occurs
	 */
	@Override
	public void writeExternal(ObjectOutput os) throws IOException {
		write(os);
	}

	@Override
	public MatrixBlock scalarOperations(ScalarOperator sop, MatrixValue result) {
		return CLALibScalar.scalarOperations(sop, this, result);
	}

	@Override
	public MatrixBlock binaryOperations(BinaryOperator op, MatrixValue thatValue, MatrixValue result) {
		MatrixBlock that = thatValue == null ? null : (MatrixBlock) thatValue;
		MatrixBlock ret = result == null ? null : (MatrixBlock) result;
		return CLALibBinaryCellOp.binaryOperationsRight(op, this, that, ret);
	}

	public MatrixBlock binaryOperationsLeft(BinaryOperator op, MatrixValue thatValue, MatrixValue result) {
		MatrixBlock that = thatValue == null ? null : (MatrixBlock) thatValue;
		MatrixBlock ret = result == null ? null : (MatrixBlock) result;
		return CLALibBinaryCellOp.binaryOperationsLeft(op, this, that, ret);
	}

	@Override
	public MatrixBlock append(MatrixBlock[] that, MatrixBlock ret, boolean cbind) {
		if(cbind && that.length == 1)
			return CLALibAppend.append(this, that[0], InfrastructureAnalyzer.getLocalParallelism());
		else {
			MatrixBlock left = getUncompressed("append list or r-bind not supported in compressed");
			MatrixBlock[] thatUC = new MatrixBlock[that.length];
			for(int i = 0; i < that.length; i++)
				thatUC[i] = getUncompressed(that[i]);
			return left.append(thatUC, ret, cbind);
		}
	}

	@Override
	public void append(MatrixValue v2, ArrayList<IndexedMatrixValue> outlist, int blen, boolean cbind, boolean m2IsLast,
		int nextNCol) {
		MatrixBlock left = getUncompressed("append ArrayList");
		MatrixBlock right = getUncompressed(v2);
		left.append(right, outlist, blen, cbind, m2IsLast, nextNCol);
	}

	@Override
	public MatrixBlock chainMatrixMultOperations(MatrixBlock v, MatrixBlock w, MatrixBlock out, ChainType ctype, int k) {

		checkMMChain(ctype, v, w);
		// multi-threaded MMChain of single uncompressed ColGroup
		if(_colGroups != null && _colGroups.size() == 1 &&
			_colGroups.get(0).getCompType() == CompressionType.UNCOMPRESSED)
			return ((ColGroupUncompressed) _colGroups.get(0)).getData().chainMatrixMultOperations(v, w, out, ctype, k);

		return CLALibMMChain.mmChain(this, v, w, out, ctype, k);
	}

	@Override
	public MatrixBlock aggregateBinaryOperations(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, AggregateBinaryOperator op) {
		checkAggregateBinaryOperations(m1, m2, op);
		return CLALibMatrixMult.matrixMultiply(m1, m2, ret, op.getNumThreads(), false, false);
	}

	public MatrixBlock aggregateBinaryOperations(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		AggregateBinaryOperator op, boolean transposeLeft, boolean transposeRight) {
		checkAggregateBinaryOperations(m1, m2, op, transposeLeft, transposeRight);
		return CLALibMatrixMult.matrixMultiply(m1, m2, ret, op.getNumThreads(), transposeLeft, transposeRight);
	}

	@Override
	public MatrixBlock aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, int blen,
		MatrixIndexes indexesIn, boolean inCP) {
		MatrixBlock ret = (result == null) ? null : (MatrixBlock) result;
		return CLALibCompAgg.aggregateUnary(this, ret, op, blen, indexesIn, inCP);
	}

	@Override
	public MatrixBlock transposeSelfMatrixMultOperations(MatrixBlock out, MMTSJType tstype, int k) {
		// check for transpose type
		if(tstype == MMTSJType.LEFT) {
			if(isEmpty())
				return new MatrixBlock(clen, clen, true);
			// create output matrix block
			if(out == null)
				out = new MatrixBlock(clen, clen, false);
			else
				out.reset(clen, clen, false);
			out.allocateDenseBlock();
			CLALibLeftMultBy.leftMultByTransposeSelf(this, out, k);
			return out;
		}
		else {
			throw new DMLRuntimeException("Invalid MMTSJ type '" + tstype.toString() + "'.");
		}
	}

	@Override
	public MatrixBlock replaceOperations(MatrixValue result, double pattern, double replacement) {
		if(isOverlapping())
			return getUncompressed("replaceOperations " + pattern + " -> " + replacement).replaceOperations(result,
				pattern, replacement);
		else {
			CompressedMatrixBlock ret = new CompressedMatrixBlock(getNumRows(), getNumColumns());
			final List<AColGroup> prev = getColGroups();
			final int colGroupsLength = prev.size();
			final List<AColGroup> retList = new ArrayList<>(colGroupsLength);
			for(int i = 0; i < colGroupsLength; i++)
				retList.add(prev.get(i).replace(pattern, replacement));
			ret.allocateColGroupList(retList);
			ret.recomputeNonZeros();
			return ret;
		}
	}

	@Override
	public MatrixBlock reorgOperations(ReorgOperator op, MatrixValue ret, int startRow, int startColumn, int length) {
		// Allow transpose to be compressed output. In general we need to have a transposed flag on
		// the compressed matrix. https://issues.apache.org/jira/browse/SYSTEMDS-3025
		printDecompressWarning(op.getClass().getSimpleName() + " -- " + op.fn.getClass().getSimpleName());
		MatrixBlock tmp = decompress(op.getNumThreads());
		return tmp.reorgOperations(op, ret, startRow, startColumn, length);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("CompressedMatrixBlock:");
		sb.append("\nCols:" + getNumColumns() + " Rows:" + getNumRows() + " Overlapping: " + isOverlapping() + " nnz: "
			+ nonZeros);
		if(_colGroups != null)
			for(AColGroup cg : _colGroups) {
				sb.append("\n" + cg);
			}
		else
			sb.append("\nEmptyColGroups");
		return sb.toString();
	}

	public boolean isOverlapping() {
		return _colGroups.size() != 1 && overlappingColGroups;
	}

	public void setOverlapping(boolean overlapping) {
		overlappingColGroups = overlapping;
	}

	@Override
	public MatrixBlock slice(int rl, int ru, int cl, int cu, boolean deep, CacheBlock ret) {
		validateSliceArgument(rl, ru, cl, cu);
		return CLALibSlice.slice(this, rl, ru, cl, cu, deep);
	}

	@Override
	public void slice(ArrayList<IndexedMatrixValue> outlist, IndexRange range, int rowCut, int colCut, int blen,
		int boundaryRlen, int boundaryClen) {
		MatrixBlock tmp = getUncompressed(
			"slice for distribution to spark. (Could be implemented such that it does not decompress)");
		tmp.slice(outlist, range, rowCut, colCut, blen, boundaryRlen, boundaryClen);
	}

	@Override
	public MatrixBlock unaryOperations(UnaryOperator op, MatrixValue result) {
		return CLALibUnary.unaryOperations(this, op, result);
	}

	@Override
	public boolean containsValue(double pattern) {
		if(isOverlapping())
			return getUncompressed("ContainsValue").containsValue(pattern);
		else {
			for(AColGroup g : _colGroups)
				if(g.containsValue(pattern))
					return true;
			return false;
		}
	}

	@Override
	public double max() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator("uamax", 1);
		return aggregateUnaryOperations(op, null, 1000, null).getValue(0, 0);
	}

	@Override
	public double min() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator("uamin", 1);
		return aggregateUnaryOperations(op, null, 1000, null).getValue(0, 0);
	}

	@Override
	public double sum() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator("uak+", 1);
		return aggregateUnaryOperations(op, null, 1000, null).getValue(0, 0);
	}

	@Override
	public MatrixBlock colSum() {
		MatrixBlock res = new MatrixBlock(1, getNumColumns(), false);
		res.allocateDenseBlock();
		double[] resV = res.getDenseBlockValues();
		AColGroup.colSum(_colGroups, resV, getNumRows());
		res.recomputeNonZeros();
		return res;
	}

	@Override
	public double sumSq() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator("uasqk+", 1);
		return aggregateUnaryOperations(op, null, 1000, null).getValue(0, 0);
	}

	@Override
	public double prod() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator("ua*", 1);
		return aggregateUnaryOperations(op, null, 1000, null).getValue(0, 0);
	}

	@Override
	public double mean() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator("uamean", 1);
		return aggregateUnaryOperations(op, null, 1000, null).getValue(0, 0);
	}

	@Override
	public MatrixBlock rexpandOperations(MatrixBlock ret, double max, boolean rows, boolean cast, boolean ignore,
		int k) {
		if(rows) {
			printDecompressWarning("rexpandOperations");
			MatrixBlock tmp = getUncompressed();
			return tmp.rexpandOperations(ret, max, rows, cast, ignore, k);
		}
		else
			return CLALibReExpand.reExpand(this, ret, max, cast, ignore, k);
	}

	@Override
	public boolean isEmptyBlock(boolean safe) {
		final long nonZeros = getNonZeros();
		return _colGroups == null || nonZeros == 0 || (nonZeros == -1 && recomputeNonZeros() == 0);
	}

	@Override
	public MatrixBlock binaryOperationsInPlace(BinaryOperator op, MatrixValue thatValue) {
		printDecompressWarning("binaryOperationsInPlace", (MatrixBlock) thatValue);
		MatrixBlock left = new MatrixBlock();
		left.copy(getUncompressed());
		MatrixBlock right = getUncompressed(thatValue);
		left.binaryOperationsInPlace(op, right);
		return left;
	}

	@Override
	public void incrementalAggregate(AggregateOperator aggOp, MatrixValue correction, MatrixValue newWithCorrection,
		boolean deep) {
		printDecompressWarning("IncrementalAggregate not supported");
		MatrixBlock left = getUncompressed();
		MatrixBlock correctionMatrixBlock = getUncompressed(correction);
		MatrixBlock newWithCorrectionMatrixBlock = getUncompressed(newWithCorrection);

		left.incrementalAggregate(aggOp, correctionMatrixBlock, newWithCorrectionMatrixBlock, deep);
	}

	@Override
	public void incrementalAggregate(AggregateOperator aggOp, MatrixValue newWithCorrection) {
		printDecompressWarning("IncrementalAggregate not supported");
		MatrixBlock left = getUncompressed();
		MatrixBlock newWithCorrectionMatrixBlock = getUncompressed(newWithCorrection);
		left.incrementalAggregate(aggOp, newWithCorrectionMatrixBlock);
	}

	@Override
	public void permutationMatrixMultOperations(MatrixValue m2Val, MatrixValue out1Val, MatrixValue out2Val, int k) {
		printDecompressWarning("permutationMatrixMultOperations", (MatrixBlock) m2Val);
		MatrixBlock left = getUncompressed();
		MatrixBlock right = getUncompressed(m2Val);
		left.permutationMatrixMultOperations(right, out1Val, out2Val, k);
	}

	@Override
	public MatrixBlock leftIndexingOperations(MatrixBlock rhsMatrix, int rl, int ru, int cl, int cu, MatrixBlock ret,
		UpdateType update) {
		printDecompressWarning("leftIndexingOperations");
		MatrixBlock left = getUncompressed();
		MatrixBlock right = getUncompressed(rhsMatrix);
		return left.leftIndexingOperations(right, rl, ru, cl, cu, ret, update);
	}

	@Override
	public MatrixBlock leftIndexingOperations(ScalarObject scalar, int rl, int cl, MatrixBlock ret, UpdateType update) {
		printDecompressWarning("leftIndexingOperations");
		MatrixBlock tmp = getUncompressed();
		return tmp.leftIndexingOperations(scalar, rl, cl, ret, update);
	}

	@Override
	public MatrixBlock zeroOutOperations(MatrixValue result, IndexRange range, boolean complementary) {
		printDecompressWarning("zeroOutOperations");
		MatrixBlock tmp = getUncompressed();
		return tmp.zeroOutOperations(result, range, complementary);
	}

	@Override
	public CM_COV_Object cmOperations(CMOperator op) {
		if(isEmpty())
			return super.cmOperations(op);
		else if(_colGroups.size() == 1 && _colGroups.get(0) instanceof AColGroupValue) {
			AColGroupValue g = (AColGroupValue) _colGroups.get(0);
			MatrixBlock vals = g.getValuesAsBlock();
			MatrixBlock counts = getCountsAsBlock(g.getCounts());
			if(counts.isEmpty())
				return vals.cmOperations(op);
			return vals.cmOperations(op, counts);
		}
		else
			return getUncompressed("cmOperations").cmOperations(op);
	}

	private static MatrixBlock getCountsAsBlock(int[] counts) {
		if(counts != null) {
			MatrixBlock ret = new MatrixBlock(counts.length, 1, false);
			for(int i = 0; i < counts.length; i++)
				ret.quickSetValue(i, 0, counts[i]);
			return ret;
		}
		else
			return new MatrixBlock(1, 1, false);
	}

	@Override
	public CM_COV_Object cmOperations(CMOperator op, MatrixBlock weights) {
		printDecompressWarning("cmOperations");
		MatrixBlock right = getUncompressed(weights);
		if(isEmpty())
			return super.cmOperations(op, right);
		AColGroup grp = _colGroups.get(0);
		if(grp instanceof ColGroupUncompressed)
			return ((ColGroupUncompressed) grp).getData().cmOperations(op);
		return getUncompressed().cmOperations(op, right);
	}

	@Override
	public CM_COV_Object covOperations(COVOperator op, MatrixBlock that) {
		MatrixBlock right = getUncompressed(that);
		return getUncompressed("covOperations").covOperations(op, right);
	}

	@Override
	public CM_COV_Object covOperations(COVOperator op, MatrixBlock that, MatrixBlock weights) {
		MatrixBlock right1 = getUncompressed(that);
		MatrixBlock right2 = getUncompressed(weights);
		return getUncompressed("covOperations").covOperations(op, right1, right2);
	}

	@Override
	public MatrixBlock sortOperations(MatrixValue weights, MatrixBlock result) {
		MatrixBlock right = getUncompressed(weights);
		return getUncompressed("sortOperations").sortOperations(right, result);
	}

	@Override
	public MatrixBlock aggregateTernaryOperations(MatrixBlock m1, MatrixBlock m2, MatrixBlock m3, MatrixBlock ret,
		AggregateTernaryOperator op, boolean inCP) {
		boolean m1C = m1 instanceof CompressedMatrixBlock;
		boolean m2C = m2 instanceof CompressedMatrixBlock;
		boolean m3C = m3 instanceof CompressedMatrixBlock;
		printDecompressWarning("aggregateTernaryOperations " + op.aggOp.getClass().getSimpleName() + " "
			+ op.indexFn.getClass().getSimpleName() + " " + op.aggOp.increOp.fn.getClass().getSimpleName() + " "
			+ op.binaryFn.getClass().getSimpleName() + " m1,m2,m3 " + m1C + " " + m2C + " " + m3C);
		MatrixBlock left = getUncompressed(m1);
		MatrixBlock right1 = getUncompressed(m2);
		MatrixBlock right2 = getUncompressed(m3);
		ret = left.aggregateTernaryOperations(left, right1, right2, ret, op, inCP);
		if(ret.getNumRows() == 0 || ret.getNumColumns() == 0)
			throw new DMLCompressionException("Invalid output");
		return ret;
	}

	@Override
	public MatrixBlock uaggouterchainOperations(MatrixBlock mbLeft, MatrixBlock mbRight, MatrixBlock mbOut,
		BinaryOperator bOp, AggregateUnaryOperator uaggOp) {
		printDecompressWarning("uaggouterchainOperations");
		MatrixBlock left = getUncompressed();
		MatrixBlock right = getUncompressed(mbRight);
		return left.uaggouterchainOperations(left, right, mbOut, bOp, uaggOp);
	}

	@Override
	public MatrixBlock groupedAggOperations(MatrixValue tgt, MatrixValue wghts, MatrixValue ret, int ngroups,
		Operator op, int k) {
		printDecompressWarning("groupedAggOperations");
		MatrixBlock left = getUncompressed();
		MatrixBlock right = getUncompressed(wghts);
		return left.groupedAggOperations(left, right, ret, ngroups, op, k);
	}

	@Override
	public MatrixBlock removeEmptyOperations(MatrixBlock ret, boolean rows, boolean emptyReturn, MatrixBlock select) {
		printDecompressWarning("removeEmptyOperations");
		MatrixBlock tmp = getUncompressed();
		return tmp.removeEmptyOperations(ret, rows, emptyReturn, select);
	}

	@Override
	public void ctableOperations(Operator op, double scalar, MatrixValue that, CTableMap resultMap,
		MatrixBlock resultBlock) {
		printDecompressWarning("ctableOperations Var 1");
		MatrixBlock left = getUncompressed();
		MatrixBlock right = getUncompressed(that);
		left.ctableOperations(op, scalar, right, resultMap, resultBlock);
	}

	@Override
	public void ctableOperations(Operator op, double scalar, double scalar2, CTableMap resultMap,
		MatrixBlock resultBlock) {
		printDecompressWarning("ctableOperations Var 2");
		MatrixBlock tmp = getUncompressed();
		tmp.ctableOperations(op, scalar, scalar2, resultMap, resultBlock);
	}

	@Override
	public void ctableOperations(Operator op, MatrixIndexes ix1, double scalar, boolean left, int brlen,
		CTableMap resultMap, MatrixBlock resultBlock) {
		printDecompressWarning("ctableOperations Var 3");
		MatrixBlock tmp = getUncompressed();
		tmp.ctableOperations(op, ix1, scalar, left, brlen, resultMap, resultBlock);
	}

	@Override
	public void ctableOperations(Operator op, MatrixValue that, double scalar, boolean ignoreZeros, CTableMap resultMap,
		MatrixBlock resultBlock) {
		printDecompressWarning("ctableOperations Var 4");
		MatrixBlock left = getUncompressed();
		MatrixBlock right = getUncompressed(that);
		left.ctableOperations(op, right, scalar, ignoreZeros, resultMap, resultBlock);
	}

	@Override
	public MatrixBlock ctableSeqOperations(MatrixValue thatMatrix, double thatScalar, MatrixBlock resultBlock,
		boolean updateClen) {
		printDecompressWarning("ctableOperations Var 5");
		MatrixBlock left = getUncompressed();
		MatrixBlock right = getUncompressed(thatMatrix);
		return left.ctableSeqOperations(right, thatScalar, resultBlock, updateClen);
	}

	@Override
	public void ctableOperations(Operator op, MatrixValue that, MatrixValue that2, CTableMap resultMap,
		MatrixBlock resultBlock) {
		MatrixBlock left = getUncompressed("ctableOperations Var 7");
		MatrixBlock right1 = getUncompressed(that);
		MatrixBlock right2 = getUncompressed(that2);
		left.ctableOperations(op, right1, right2, resultMap, resultBlock);
	}

	@Override
	public MatrixBlock ternaryOperations(TernaryOperator op, MatrixBlock m2, MatrixBlock m3, MatrixBlock ret) {

		// prepare inputs
		final int r1 = getNumRows();
		final int r2 = m2.getNumRows();
		final int r3 = m3.getNumRows();
		final int c1 = getNumColumns();
		final int c2 = m2.getNumColumns();
		final int c3 = m3.getNumColumns();
		final boolean s1 = (r1 == 1 && c1 == 1);
		final boolean s2 = (r2 == 1 && c2 == 1);
		final boolean s3 = (r3 == 1 && c3 == 1);
		final double d1 = s1 ? quickGetValue(0, 0) : Double.NaN;
		final double d2 = s2 ? m2.quickGetValue(0, 0) : Double.NaN;
		final double d3 = s3 ? m3.quickGetValue(0, 0) : Double.NaN;
		final int m = Math.max(Math.max(r1, r2), r3);
		final int n = Math.max(Math.max(c1, c2), c3);

		ternaryOperationCheck(s1, s2, s3, m, r1, r2, r3, n, c1, c2, c3);

		final boolean PM_Or_MM = (op.fn instanceof PlusMultiply || op.fn instanceof MinusMultiply);
		if(PM_Or_MM && ((s2 && d2 == 0) || (s3 && d3 == 0))) {
			ret = new CompressedMatrixBlock();
			ret.copy(this);
			return ret;
		}

		if(m2 instanceof CompressedMatrixBlock)
			m2 = ((CompressedMatrixBlock) m2).getUncompressed("Ternary Operator arg2 " + op.fn.getClass().getSimpleName());
		if(m3 instanceof CompressedMatrixBlock)
			m3 = ((CompressedMatrixBlock) m3).getUncompressed("Ternary Operator arg3 " + op.fn.getClass().getSimpleName());

		if(s2 != s3 && (op.fn instanceof PlusMultiply || op.fn instanceof MinusMultiply)) {
			// SPECIAL CASE for sparse-dense combinations of common +* and -*
			BinaryOperator bop = ((ValueFunctionWithConstant) op.fn).setOp2Constant(s2 ? d2 : d3);
			bop.setNumThreads(op.getNumThreads());
			ret = CLALibBinaryCellOp.binaryOperationsRight(bop, this, s2 ? m3 : m2, ret);
		}
		else {
			final boolean sparseOutput = evalSparseFormatInMemory(m, n, (s1 ? m * n * (d1 != 0 ? 1 : 0) : getNonZeros()) +
				Math.min(s2 ? m * n : m2.getNonZeros(), s3 ? m * n : m3.getNonZeros()));
			ret.reset(m, n, sparseOutput);
			final MatrixBlock thisUncompressed = getUncompressed("Ternary Operation not supported");
			LibMatrixTercell.tercellOp(thisUncompressed, m2, m3, ret, op);
			ret.examSparsity();
		}
		return ret;
	}

	@Override
	public MatrixBlock quaternaryOperations(QuaternaryOperator qop, MatrixBlock um, MatrixBlock vm, MatrixBlock wm,
		MatrixBlock out, int k) {
		MatrixBlock left = getUncompressed("quaternaryOperations");
		MatrixBlock right1 = getUncompressed(um);
		MatrixBlock right2 = getUncompressed(vm);
		MatrixBlock right3 = getUncompressed(wm);
		return left.quaternaryOperations(qop, right1, right2, right3, out, k);
	}

	@Override
	public MatrixBlock randOperationsInPlace(RandomMatrixGenerator rgen, Well1024a bigrand, long bSeed) {
		LOG.info("Inplace rand ops not on CompressedMatrix");
		MatrixBlock ret = new MatrixBlock(getNumRows(), getNumColumns(), true);
		LibMatrixDatagen.generateRandomMatrix(ret, rgen, bigrand, bSeed);
		return ret;
	}

	@Override
	public MatrixBlock randOperationsInPlace(RandomMatrixGenerator rgen, Well1024a bigrand, long bSeed, int k) {
		LOG.info("Inplace rand ops not on CompressedMatrix");
		MatrixBlock ret = new MatrixBlock(getNumRows(), getNumColumns(), true);
		LibMatrixDatagen.generateRandomMatrix(ret, rgen, bigrand, bSeed, k);
		return ret;
	}

	@Override
	public MatrixBlock seqOperationsInPlace(double from, double to, double incr) {
		// output should always be uncompressed
		throw new DMLRuntimeException("CompressedMatrixBlock: seqOperationsInPlace not supported.");
	}

	private static boolean isCompressed(MatrixBlock mb) {
		return mb instanceof CompressedMatrixBlock;
	}

	public static MatrixBlock getUncompressed(MatrixValue mVal) {
		return isCompressed((MatrixBlock) mVal) ? ((CompressedMatrixBlock) mVal).getUncompressed() : (MatrixBlock) mVal;
	}

	public static MatrixBlock getUncompressed(MatrixValue mVal, String message) {
		return isCompressed((MatrixBlock) mVal) ? ((CompressedMatrixBlock) mVal)
			.getUncompressed(message) : (MatrixBlock) mVal;
	}

	public MatrixBlock getUncompressed() {
		MatrixBlock d_compressed = getCachedDecompressed();
		if(d_compressed != null)
			return d_compressed;
		if(isEmpty())
			return new MatrixBlock(getNumRows(), getNumColumns(), true);
		return this.decompress(InfrastructureAnalyzer.getLocalParallelism());
	}

	public MatrixBlock getUncompressed(String operation) {
		printDecompressWarning(operation);
		return getUncompressed();
	}

	private static void printDecompressWarning(String operation) {
		LOG.warn("Decompressing because: " + operation);
	}

	private static void printDecompressWarning(String operation, MatrixBlock m2) {
		if(isCompressed(m2))
			printDecompressWarning(operation);
	}

	@Override
	public boolean isShallowSerialize(boolean inclConvert) {
		return true;
	}

	@Override
	public void toShallowSerializeBlock() {
		// do nothing
	}

	@Override
	public void copy(MatrixValue thatValue) {
		copy(thatValue, false);
	}

	private static CompressedMatrixBlock checkType(MatrixValue thatValue) {
		if(thatValue == null || !(thatValue instanceof CompressedMatrixBlock))
			throw new DMLRuntimeException("Invalid call to copy, requre a compressed MatrixBlock to copy to");

		return (CompressedMatrixBlock) thatValue;
	}

	@Override
	public void copy(MatrixValue thatValue, boolean sp) {
		CompressedMatrixBlock that = checkType(thatValue);
		if(this == that) // prevent data loss (e.g., on sparse-dense conversion)
			throw new RuntimeException("Copy must not overwrite itself!");
		copyCompressedMatrix(that);
	}

	@Override
	public MatrixBlock copyShallow(MatrixBlock that) {
		if(that instanceof CompressedMatrixBlock)
			throw new NotImplementedException();
		else
			throw new DMLCompressionException(
				"Invalid copy shallow, since the matrixBlock given is not of type CompressedMatrixBLock");
	}

	@Override
	public void copy(int rl, int ru, int cl, int cu, MatrixBlock src, boolean awareDestNZ) {
		throw new DMLCompressionException("Invalid copy into CompressedMatrixBlock");
	}

	private void copyCompressedMatrix(CompressedMatrixBlock that) {
		this.rlen = that.getNumRows();
		this.clen = that.getNumColumns();
		this.sparseBlock = null;
		this.denseBlock = null;
		this.nonZeros = that.getNonZeros();

		this._colGroups = new ArrayList<>(that.getColGroups().size());
		for(AColGroup cg : that._colGroups)
			_colGroups.add(cg.copy());

		overlappingColGroups = that.overlappingColGroups;
	}

	public SoftReference<MatrixBlock> getSoftReferenceToDecompressed() {
		return decompressedVersion;
	}

	public void clearSoftReferenceToDecompressed() {
		decompressedVersion = null;
	}

	@Override
	public DenseBlock getDenseBlock() {
		throw new DMLCompressionException("Should not get DenseBlock on a compressed Matrix");
	}

	@Override
	public void setDenseBlock(DenseBlock dblock) {
		throw new DMLCompressionException("Should not set DenseBlock on a compressed Matrix");
	}

	@Override
	public double[] getDenseBlockValues() {
		throw new DMLCompressionException("Should not get DenseBlock values on a compressed Matrix");
	}

	@Override
	public SparseBlock getSparseBlock() {
		throw new DMLCompressionException("Should not get SparseBlock on a compressed Matrix");
	}

	@Override
	public void setSparseBlock(SparseBlock sblock) {
		throw new DMLCompressionException("Should not set SparseBlock on a compressed Matrix");
	}

	@Override
	public Iterator<IJV> getSparseBlockIterator() {
		throw new DMLCompressionException("Should not get SparseBlockIterator on a compressed Matrix");
	}

	@Override
	public Iterator<IJV> getSparseBlockIterator(int rl, int ru) {
		throw new DMLCompressionException("Should not get SparseBlockIterator on a compressed Matrix");
	}

	@Override
	public void quickSetValue(int r, int c, double v) {
		throw new DMLCompressionException("Should not set a value on a compressed Matrix");
	}

	@Override
	public double quickGetValueThreadSafe(int r, int c) {
		throw new DMLCompressionException("Thread safe execution does not work on Compressed Matrix");
	}

	@Override
	public double getValueDenseUnsafe(int r, int c) {
		throw new DMLCompressionException("Compressed Matrix does not have a dense matrix block");
	}

	@Override
	public void appendValue(int r, int c, double v) {
		throw new DMLCompressionException("Cant append value to compressed Matrix");
	}

	@Override
	public void appendValuePlain(int r, int c, double v) {
		throw new DMLCompressionException("Can't append value to compressed Matrix");
	}

	@Override
	public void appendRow(int r, SparseRow row, boolean deep) {
		throw new DMLCompressionException("Can't append row to compressed Matrix");
	}

	@Override
	public void appendRowToSparse(SparseBlock dest, MatrixBlock src, int i, int rowoffset, int coloffset, boolean deep) {
		throw new DMLCompressionException("Can't append row to compressed Matrix");
	}

	@Override
	public void sortSparseRows() {
		throw new DMLCompressionException("It does not make sense to sort the rows in a compressed matrix");
	}

	@Override
	public void sortSparseRows(int rl, int ru) {
		throw new DMLCompressionException("It does not make sense to sort the rows in a compressed matrix");
	}

	@Override
	public double minNonZero() {
		throw new NotImplementedException();
	}

	@Override
	public boolean isInSparseFormat() {
		return false;
	}

	@Override
	public boolean isUltraSparse() {
		return false;
	}

	@Override
	public boolean isUltraSparse(boolean checkNnz) {
		return false;
	}

	@Override
	public boolean isSparsePermutationMatrix() {
		return false;
	}

	@Override
	public boolean evalSparseFormatInMemory() {
		return false;
	}

	@Override
	public boolean evalSparseFormatOnDisk() {
		return false;
	}

	@Override
	public void examSparsity(boolean allowCSR) {
		// do nothing
	}

	@Override
	public void sparseToDense() {
		// do nothing
	}

	@Override
	public void merge(MatrixBlock that, boolean appendOnly, boolean par, boolean deep) {
		throw new NotImplementedException();
	}

	@Override
	public void compactEmptyBlock() {
		if(isEmptyBlock(false)) {
			cleanupBlock(true, true);
			CLALibUtils.combineConstColumns(this);
			overlappingColGroups = false;
			decompressedVersion = null;
		}
	}

	@Override
	public void dropLastRowsOrColumns(CorrectionLocationType correctionLocation) {
		throw new NotImplementedException();
	}

	@Override
	public double interQuartileMean() {
		return getUncompressed("interQuartileMean").interQuartileMean();
	}

	@Override
	public MatrixBlock pickValues(MatrixValue quantiles, MatrixValue ret) {
		return getUncompressed("pickValues").pickValues(quantiles, ret);
	}

	@Override
	public double pickValue(double quantile, boolean average) {
		return getUncompressed("pickValue").pickValue(quantile, average);
	}

	@Override
	public double sumWeightForQuantile() {
		return getUncompressed("sumWeightForQuantile").sumWeightForQuantile();
	}

	@Override
	public MatrixBlock extractTriangular(MatrixBlock ret, boolean lower, boolean diag, boolean values) {
		return getUncompressed("extractTriangular").extractTriangular(ret, lower, diag, values);
	}

	@Override
	public boolean isThreadSafe() {
		return false;
	}

	@Override
	public void checkNaN() {
		throw new NotImplementedException();
	}

	@Override
	public void init(double[][] arr, int r, int c) {
		throw new DMLCompressionException("Invalid to init on a compressed MatrixBlock");
	}

	@Override
	public void init(double[] arr, int r, int c) {
		throw new DMLCompressionException("Invalid to init on a compressed MatrixBlock");
	}

	@Override
	public boolean isAllocated() {
		return true;
	}

	@Override
	public Future<MatrixBlock> allocateBlockAsync() {
		throw new DMLCompressionException("Invalid to allocate dense block on a compressed MatrixBlock");
	}

	@Override
	public boolean allocateDenseBlock(boolean clearNNZ) {
		throw new DMLCompressionException("Invalid to allocate dense block on a compressed MatrixBlock");
	}

	@Override
	public boolean allocateSparseRowsBlock(boolean clearNNZ) {
		throw new DMLCompressionException("Invalid to allocate sparse block on a compressed MatrixBlock");
	}

	@Override
	public void allocateAndResetSparseBlock(boolean clearNNZ, SparseBlock.Type stype) {
		throw new DMLCompressionException("Invalid to allocate block on a compressed MatrixBlock");
	}
}

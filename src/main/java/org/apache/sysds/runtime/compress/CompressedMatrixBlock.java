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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConverter;
import org.apache.sysds.runtime.compress.colgroup.ColGroupIO;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.compress.colgroup.DenseRowIterator;
import org.apache.sysds.runtime.compress.colgroup.SparseRowIterator;
import org.apache.sysds.runtime.compress.lib.LibBinaryCellOp;
import org.apache.sysds.runtime.compress.lib.LibLeftMultBy;
import org.apache.sysds.runtime.compress.lib.LibRightMultBy;
import org.apache.sysds.runtime.compress.lib.LibScalar;
import org.apache.sysds.runtime.compress.utils.ColumnGroupIterator;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell.BinaryAccessType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.DMLCompressionStatistics;

public class CompressedMatrixBlock extends AbstractCompressedMatrixBlock {
	private static final Log LOG = LogFactory.getLog(CompressedMatrixBlock.class.getName());
	private static final long serialVersionUID = 7319372019143154058L;

	/** Threshold for when to parallelize the aggregation functions. */
	private static final long MIN_PAR_AGG_THRESHOLD = 8 * 1024 * 1024; // 8MB

	/**
	 * Constructor for building an empty Compressed Matrix block object.
	 * 
	 * OBS! Only to be used for serialization.
	 */
	public CompressedMatrixBlock() {
		super();
	}

	/**
	 * Create a base Compressed matrix block with overlapping column groups.
	 * 
	 * @param overLapping boolean specifier of the if the groups are overlapping.
	 */
	public CompressedMatrixBlock(boolean overLapping) {
		super(overLapping);
	}

	/**
	 * Main constructor for building a block from scratch.
	 * 
	 * @param rl     number of rows in the block
	 * @param cl     number of columns
	 * @param sparse true if the UNCOMPRESSED representation of the block should be sparse
	 */
	protected CompressedMatrixBlock(int rl, int cl, boolean sparse) {
		super(rl, cl, sparse);
	}

	/**
	 * "Copy" constructor to populate this compressed block with the uncompressed contents of a conventional block. Does
	 * <b>not</b> compress the block. Only creates a shallow copy, and only does deep copy on compression.
	 * 
	 * @param that matrix block
	 */
	protected CompressedMatrixBlock(MatrixBlock that) {
		super(that.getNumRows(), that.getNumColumns(), that.isInSparseFormat());

		// shallow copy (deep copy on compression, prevents unnecessary copy)
		if(isInSparseFormat())
			sparseBlock = that.getSparseBlock();
		else
			denseBlock = that.getDenseBlock();
		nonZeros = that.getNonZeros();
	}

	public boolean isSingleUncompressedGroup() {
		return(_colGroups != null && _colGroups.size() == 1 &&
			_colGroups.get(0).getCompType() == CompressionType.UNCOMPRESSED);
	}

	public void allocateColGroupList(List<ColGroup> colGroups) {
		_colGroups = colGroups;
	}

	public List<ColGroup> getColGroups() {
		return _colGroups;
	}

	/**
	 * Decompress block.
	 * 
	 * @return a new uncompressed matrix block containing the contents of this block
	 */
	public MatrixBlock decompress() {

		Timing time = new Timing(true);

		// preallocation sparse rows to avoid repeated reallocations
		MatrixBlock ret = (nonZeros == -1) ? new MatrixBlock(rlen, clen, false, -1)
			.allocateBlock() : new MatrixBlock(rlen, clen, sparse, nonZeros).allocateBlock();
		if(ret.isInSparseFormat()) {
			int[] rnnz = new int[rlen];
			for(ColGroup grp : _colGroups)
				grp.countNonZerosPerRow(rnnz, 0, rlen);
			ret.allocateSparseRowsBlock();
			SparseBlock rows = ret.getSparseBlock();
			for(int i = 0; i < rlen; i++)
				rows.allocate(i, rnnz[i]);
		}

		// core decompression (append if sparse)
		for(ColGroup grp : _colGroups)
			grp.decompressToBlock(ret, 0, rlen);

		// post-processing (for append in decompress)
		if(ret.getNonZeros() == -1 || nonZeros == -1) {
			ret.recomputeNonZeros();
		}
		else {
			ret.setNonZeros(nonZeros);
		}
		if(ret.isInSparseFormat())
			ret.sortSparseRows();

		if(DMLScript.STATISTICS || LOG.isDebugEnabled()) {
			double t = time.stop();
			LOG.debug("decompressed block w/ k=" + 1 + " in " + t + "ms.");
			DMLCompressionStatistics.addDecompressTime(t,1);
		}
		return ret;
	}

	/**
	 * Decompress block.
	 * 
	 * @param k degree of parallelism
	 * @return a new uncompressed matrix block containing the contents of this block
	 */
	public MatrixBlock decompress(int k) {

		if(k <= 1)
			return decompress();

		Timing time = new Timing(true);

		MatrixBlock ret = (nonZeros == -1) ? new MatrixBlock(rlen, clen, false, -1)
			.allocateBlock() : new MatrixBlock(rlen, clen, sparse, nonZeros).allocateBlock();
		// multi-threaded decompression
		nonZeros = 0;
		try {
			ExecutorService pool = CommonThreadPool.get(k);
			int rlen = getNumRows();
			final int blkz = CompressionSettings.BITMAP_BLOCK_SZ;
			int blklen = (int) Math.ceil((double) rlen / k);
			blklen += (blklen % blkz != 0) ? blkz - blklen % blkz : 0;
			ArrayList<DecompressTask> tasks = new ArrayList<>();
			for(int i = 0; i < k & i * blklen < getNumRows(); i++)
				tasks.add(new DecompressTask(_colGroups, ret, i * blklen, Math.min((i + 1) * blklen, rlen)));
			List<Future<Long>> rtasks = pool.invokeAll(tasks);
			pool.shutdown();
			for(Future<Long> rt : rtasks)
				nonZeros += rt.get(); // error handling
		}
		catch(InterruptedException | ExecutionException ex) {
			LOG.error("Parallel decompression failed defaulting to non parallel implementation " + ex.getMessage());
			nonZeros = -1;
			ex.printStackTrace();
			return decompress();
		}

		ret.setNonZeros(nonZeros);

		if(DMLScript.STATISTICS || LOG.isDebugEnabled()) {
			double t = time.stop();
			LOG.debug("decompressed block w/ k=" + k + " in " + time.stop() + "ms.");
			DMLCompressionStatistics.addDecompressTime(t, k);
		}
		return ret;
	}

	/**
	 * Obtain an upper bound on the memory used to store the compressed block.
	 * 
	 * @return an upper bound on the memory used to store this compressed block considering class overhead.
	 */
	public long estimateCompressedSizeInMemory() {
		long total = baseSizeInMemory();

		for(ColGroup grp : _colGroups)
			total += grp.estimateInMemorySize();

		return total;
	}

	public static long baseSizeInMemory() {
		long total = 16; // Object header

		total += 40; // Matrix Block elements
		total += 8; // Col Group Ref
		total += 2 + 6; // Booleans plus padding

		total += 40; // Col Group Array List
		return total;
	}

	@Override
	public double quickGetValue(int r, int c) {

		// TODO Optimize Quick Get Value, to located the correct column group without having to search for it
		double v = 0.0;
		for(ColGroup group : _colGroups) {
			if(Arrays.binarySearch(group.getColIndices(), c) >= 0) {
				v += group.get(r, c);
				if(!isOverlapping())
					break;
			}
		}

		// find row value
		return v;
	}

	//////////////////////////////////////////
	// Serialization / Deserialization

	@Override
	public long getExactSizeOnDisk() {
		// header information
		long ret = 20;
		for(ColGroup grp : _colGroups) {
			ret += 1; // type info
			ret += grp.getExactSizeOnDisk();
		}
		return ret;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		// deserialize compressed block
		rlen = in.readInt();
		clen = in.readInt();
		nonZeros = in.readLong();
		overlappingColGroups = in.readBoolean();
		_colGroups = ColGroupIO.readGroups(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
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

	public Iterator<IJV> getIterator(int rl, int ru, boolean inclZeros) {
		return getIterator(rl, ru, 0, _colGroups.size(), inclZeros);
	}

	public Iterator<IJV> getIterator(int rl, int ru, int cgl, int cgu, boolean inclZeros) {
		return new ColumnGroupIterator(rl, ru, cgl, cgu, inclZeros, _colGroups);
	}

	public Iterator<double[]> getDenseRowIterator(int rl, int ru) {
		return new DenseRowIterator(rl, ru, _colGroups, clen);
	}

	public Iterator<SparseRow> getSparseRowIterator(int rl, int ru) {
		return new SparseRowIterator(rl, ru, _colGroups, clen);
	}

	public int[] countNonZerosPerRow(int rl, int ru) {
		int[] rnnz = new int[ru - rl];
		for(ColGroup grp : _colGroups)
			grp.countNonZerosPerRow(rnnz, rl, ru);
		return rnnz;
	}

	@Override
	public MatrixBlock scalarOperations(ScalarOperator sop, MatrixValue result) {
		if(overlappingColGroups && !(sop.fn instanceof Multiply || sop.fn instanceof Plus || sop.fn instanceof Minus ||
			(sop instanceof LeftScalarOperator && sop.fn instanceof Power2))) {
			MatrixBlock m1d = decompress(sop.getNumThreads());
			result = m1d.scalarOperations(sop, result);
			return (MatrixBlock) result;
		}

		CompressedMatrixBlock ret = null;
		if(result == null || !(result instanceof CompressedMatrixBlock))
			ret = new CompressedMatrixBlock(getNumRows(), getNumColumns(), sparse);
		result = LibScalar.scalarOperations(sop, this, ret, overlappingColGroups);
		return (MatrixBlock) result;
	}

	@Override
	public MatrixBlock binaryOperations(BinaryOperator op, MatrixValue thatValue, MatrixValue result) {

		MatrixBlock that = getUncompressed(thatValue);
		if(!LibMatrixBincell.isValidDimensionsBinary(this, that)) {
			throw new DMLRuntimeException("Block sizes are not matched for binary " + "cell operations: " + this.rlen
				+ "x" + this.clen + " vs " + that.getNumRows() + "x" + that.getNumColumns());
		}

		if(LibMatrixBincell.getBinaryAccessType(this, that) == BinaryAccessType.MATRIX_COL_VECTOR ||
			(this.getNumColumns() == 1 && that.getNumColumns() == 1 && that.getNumRows() != 1) ||
			!(op.fn instanceof Multiply || op.fn instanceof Plus || op.fn instanceof Minus ||
				op.fn instanceof MinusMultiply || op.fn instanceof PlusMultiply)) {
			// case MATRIX_COL_VECTOR:
			// TODO make partial decompress and do operation.
			// TODO support more of the operations... since it is possible.
			MatrixBlock m2 = getUncompressed(this);
			MatrixBlock ret = m2.binaryOperations(op, thatValue, result);
			result = ret;
			return ret;
		}
		else {

			CompressedMatrixBlock ret = null;
			if(result == null || !(result instanceof CompressedMatrixBlock))
				ret = new CompressedMatrixBlock(getNumRows(), getNumColumns(), sparse);
			else {
				ret = (CompressedMatrixBlock) result;
				ret.reset(rlen, clen);
			}
			result = LibBinaryCellOp.bincellOp(this, that, ret, op);
			result = ret;
			return ret;
		}

	}

	@Override
	public MatrixBlock append(MatrixBlock that, MatrixBlock ret) {

		final int m = rlen;
		final int n = clen + that.getNumColumns();
		final long nnz = nonZeros + that.getNonZeros();

		// init result matrix
		CompressedMatrixBlock ret2 = null;
		if(ret == null || !(ret instanceof CompressedMatrixBlock)) {
			ret2 = new CompressedMatrixBlock(m, n, isInSparseFormat());
		}
		else {
			ret2 = (CompressedMatrixBlock) ret;
			ret2.reset(m, n);
		}

		// shallow copy of lhs column groups
		ret2.allocateColGroupList(new ArrayList<ColGroup>());
		ret2._colGroups.addAll(_colGroups);

		// copy of rhs column groups w/ col index shifting
		if(!(that instanceof CompressedMatrixBlock)) {
			that = CompressedMatrixBlockFactory.compress(that).getLeft();
		}

		List<ColGroup> inColGroups = ((CompressedMatrixBlock) that)._colGroups;
		for(ColGroup group : inColGroups) {
			ColGroup tmp = ColGroupConverter.copyColGroup(group);
			tmp.shiftColIndices(clen);
			ret2._colGroups.add(tmp);
		}

		// meta data maintenance
		ret2.setNonZeros(nnz);
		return ret2;
	}

	@Override
	public MatrixBlock chainMatrixMultOperations(MatrixBlock v, MatrixBlock w, MatrixBlock out, ChainType ctype) {
		return chainMatrixMultOperations(v, w, out, ctype, 1);
	}

	@Override
	public MatrixBlock chainMatrixMultOperations(MatrixBlock v, MatrixBlock w, MatrixBlock out, ChainType ctype,
		int k) {

		if(this.getNumColumns() != v.getNumRows())
			throw new DMLRuntimeException(
				"Dimensions mismatch on mmchain operation (" + this.getNumColumns() + " != " + v.getNumRows() + ")");
		if(v.getNumColumns() != 1)
			throw new DMLRuntimeException(
				"Invalid input vector (column vector expected, but ncol=" + v.getNumColumns() + ")");
		if(w != null && w.getNumColumns() != 1)
			throw new DMLRuntimeException(
				"Invalid weight vector (column vector expected, but ncol=" + w.getNumColumns() + ")");

		// multi-threaded MMChain of single uncompressed ColGroup
		if(isSingleUncompressedGroup()) {
			return ((ColGroupUncompressed) _colGroups.get(0)).getData().chainMatrixMultOperations(v, w, out, ctype, k);
		}

		// Timing time = LOG.isDebugEnabled() ? new Timing(true) : null;

		// prepare result
		if(out != null)
			out.reset(clen, 1, false);
		else
			out = new MatrixBlock(clen, 1, false);

		// empty block handling
		if(isEmptyBlock(false))
			return out;

		// compute matrix mult
		MatrixBlock tmp = new MatrixBlock(rlen, 1, false);
		tmp  = LibRightMultBy.rightMultByMatrix(_colGroups, v, tmp, k, getMaxNumValues(), false);
		if(ctype == ChainType.XtwXv) {
			BinaryOperator bop = new BinaryOperator(Multiply.getMultiplyFnObject());
			LibMatrixBincell.bincellOpInPlace(tmp, w, bop);
		}
		LibLeftMultBy.leftMultByVectorTranspose(_colGroups, tmp, out, true, k, getMaxNumValues(), isOverlapping());

		return out;
	}

	@Override
	public MatrixBlock aggregateBinaryOperations(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		AggregateBinaryOperator op) {

		// setup meta data (dimensions, sparsity)

		boolean right = (m1 == this);
		MatrixBlock that = right ? m2 : m1;
		if(!right && m2 != this) {
			throw new DMLRuntimeException(
				"Invalid inputs for aggregate Binary Operation which expect either m1 or m2 to be equal to the object calling");
		}

		int rl = m1.getNumRows();
		int cl = m2.getNumColumns();

		// create output matrix block
		if(right) {
			that = that instanceof CompressedMatrixBlock ? ((CompressedMatrixBlock) that).decompress() : that;
			return ret = LibRightMultBy
				.rightMultByMatrix(_colGroups, that, ret, op.getNumThreads(), getMaxNumValues(), true);
		}
		else {
			return LibLeftMultBy.leftMultByMatrix(_colGroups,
				that,
				ret,
				false,
				true,
				rl,
				cl,
				isOverlapping(),
				op.getNumThreads(),
				getMaxNumValues());
		}

	}

	@Override
	public MatrixBlock aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, int blen,
		MatrixIndexes indexesIn, boolean inCP) {

		// check for supported operations
		if(!(op.aggOp.increOp.fn instanceof KahanPlus || op.aggOp.increOp.fn instanceof KahanPlusSq ||
			op.aggOp.increOp.fn instanceof Mean ||
			(op.aggOp.increOp.fn instanceof Builtin &&
				(((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN ||
					((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAX)))) {
			throw new NotImplementedException("Unary aggregate " + op.aggOp.increOp.fn + " not supported yet.");
		}

		if(overlappingColGroups &&
			(op.aggOp.increOp.fn instanceof KahanPlusSq || (op.aggOp.increOp.fn instanceof Builtin &&
				(((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN ||
					((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAX)))) {
			MatrixBlock m1d = decompress(op.getNumThreads());
			return m1d.aggregateUnaryOperations(op, result, blen, indexesIn, inCP);
		}

		// prepare output dimensions
		CellIndex tempCellIndex = new CellIndex(-1, -1);
		op.indexFn.computeDimension(rlen, clen, tempCellIndex);
		// Correction no long exists
		if(op.aggOp.existsCorrection()) {
			switch(op.aggOp.correction) {
				case LASTROW:
					tempCellIndex.row++;
					break;
				case LASTCOLUMN:
					tempCellIndex.column++;
					break;
				case LASTTWOROWS:
					tempCellIndex.row += 2;
					break;
				case LASTTWOCOLUMNS:
					tempCellIndex.column += 2;
					break;
				default:
					throw new DMLRuntimeException("unrecognized correctionLocation: " + op.aggOp.correction);
			}
		}

		// initialize and allocate the result
		if(result == null)
			result = new MatrixBlock(tempCellIndex.row, tempCellIndex.column, false);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, false);
		MatrixBlock ret = (MatrixBlock) result;
		ret.allocateDenseBlock();

		if(op.aggOp.increOp.fn instanceof Builtin) {
			Double val = null;
			switch(((Builtin) op.aggOp.increOp.fn).getBuiltinCode()) {
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

		// core unary aggregate
		if(op.getNumThreads() > 1 && getExactSizeOnDisk() > MIN_PAR_AGG_THRESHOLD) {
			// multi-threaded execution of all groups
			ArrayList<ColGroup>[] grpParts = createStaticTaskPartitioning(_colGroups,
				(op.indexFn instanceof ReduceCol) ? 1 : op.getNumThreads(),
				false);
			ColGroupUncompressed uc = getUncompressedColGroup();

			try {
				// compute uncompressed column group in parallel (otherwise bottleneck)
				if(uc != null)
					uc.unaryAggregateOperations(op, ret);
				// compute all compressed column groups
				ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
				ArrayList<UnaryAggregateTask> tasks = new ArrayList<>();
				if(op.indexFn instanceof ReduceCol && grpParts.length > 0) {
					final int blkz = CompressionSettings.BITMAP_BLOCK_SZ;
					int blklen = (int) Math.ceil((double) rlen / op.getNumThreads());
					blklen += (blklen % blkz != 0) ? blkz - blklen % blkz : 0;
					for(int i = 0; i < op.getNumThreads() & i * blklen < rlen; i++) {
						tasks.add(
							new UnaryAggregateTask(grpParts[0], ret, i * blklen, Math.min((i + 1) * blklen, rlen), op));

					}
				}
				else
					for(ArrayList<ColGroup> grp : grpParts) {
						if(grp != null)
							tasks.add(new UnaryAggregateTask(grp, ret, 0, rlen, op));
					}
				List<Future<MatrixBlock>> rtasks = pool.invokeAll(tasks);
				pool.shutdown();

				// aggregate partial results
				if(op.indexFn instanceof ReduceAll) {
					if(op.aggOp.increOp.fn instanceof KahanFunction) {
						KahanObject kbuff = new KahanObject(ret.quickGetValue(0, 0), 0);
						for(Future<MatrixBlock> rtask : rtasks) {
							double tmp = rtask.get().quickGetValue(0, 0);
							((KahanFunction) op.aggOp.increOp.fn).execute2(kbuff, tmp);
						}
						ret.quickSetValue(0, 0, kbuff._sum);
					}
					else if(op.aggOp.increOp.fn instanceof Mean) {
						double val = ret.quickGetValue(0, 0);
						for(Future<MatrixBlock> rtask : rtasks) {
							double tmp = rtask.get().quickGetValue(0, 0);
							val = val + tmp;
						}
						ret.quickSetValue(0, 0, val);
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
			}
			catch(InterruptedException | ExecutionException e) {
				LOG.fatal("UnaryAggregate Exception: " + e.getMessage(), e);
				throw new DMLRuntimeException(e);
			}
		}
		else {
			if(_colGroups != null) {

				for(ColGroup grp : _colGroups)
					if(grp instanceof ColGroupUncompressed)
						((ColGroupUncompressed) grp).unaryAggregateOperations(op, ret);
				aggregateUnaryOperations(op, _colGroups, ret, 0, rlen);
			}
		}

		// special handling zeros for rowmins/rowmax
		if(op.indexFn instanceof ReduceCol && op.aggOp.increOp.fn instanceof Builtin) {
			int[] rnnz = new int[rlen];
			for(ColGroup grp : _colGroups)
				grp.countNonZerosPerRow(rnnz, 0, rlen);
			Builtin builtin = (Builtin) op.aggOp.increOp.fn;
			for(int i = 0; i < rlen; i++)
				if(rnnz[i] < clen)
					ret.quickSetValue(i, 0, builtin.execute(ret.quickGetValue(i, 0), 0));
		}

		// special handling of mean
		if(op.aggOp.increOp.fn instanceof Mean) {
			if(op.indexFn instanceof ReduceAll) {
				ret.quickSetValue(0, 0, ret.quickGetValue(0, 0) / (getNumColumns() * getNumRows()));
			}
			else if(op.indexFn instanceof ReduceCol) {
				for(int i = 0; i < getNumRows(); i++) {
					ret.quickSetValue(i, 0, ret.quickGetValue(i, 0) / getNumColumns());
				}
			}
			else if(op.indexFn instanceof ReduceRow)
				for(int i = 0; i < getNumColumns(); i++) {
					ret.quickSetValue(0, i, ret.quickGetValue(0, i) / getNumRows());
				}
		}

		// drop correction if necessary
		if(op.aggOp.existsCorrection() && inCP)
			ret.dropLastRowsOrColumns(op.aggOp.correction);

		// post-processing
		ret.recomputeNonZeros();

		return ret;
	}

	@Override
	public MatrixBlock aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, int blen,
		MatrixIndexes indexesIn) {
		return aggregateUnaryOperations(op, result, blen, indexesIn, false);
	}

	private static void aggregateUnaryOperations(AggregateUnaryOperator op, List<ColGroup> groups, MatrixBlock ret,
		int rl, int ru) {

		// note: UC group never passed into this function
		double[] c = ret.getDenseBlockValues();
		for(ColGroup grp : groups)
			if(grp != null && !(grp instanceof ColGroupUncompressed))
				grp.unaryAggregateOperations(op, c, rl, ru);

	}

	@Override
	public MatrixBlock transposeSelfMatrixMultOperations(MatrixBlock out, MMTSJType tstype) {
		return transposeSelfMatrixMultOperations(out, tstype, 1);
	}

	@Override
	public MatrixBlock transposeSelfMatrixMultOperations(MatrixBlock out, MMTSJType tstype, int k) {
		// check for transpose type
		if(tstype != MMTSJType.LEFT) // right not supported yet
			throw new DMLRuntimeException("Invalid MMTSJ type '" + tstype.toString() + "'.");

		// create output matrix block
		if(out == null)
			out = new MatrixBlock(clen, clen, false);
		else
			out.reset(clen, clen, false);
		out.allocateDenseBlock();

		if(!isEmptyBlock(false)) {
			// compute matrix mult
			LibLeftMultBy.leftMultByTransposeSelf(_colGroups,
				out,
				0,
				_colGroups.size(),
				k,
				getNumColumns(),
				getMaxNumValues(),
				isOverlapping());
			// post-processing
			out.setNonZeros(LinearAlgebraUtils.copyUpperToLowerTriangle(out));
		}
		return out;
	}

	@SuppressWarnings("unchecked")
	private static ArrayList<ColGroup>[] createStaticTaskPartitioning(List<ColGroup> colGroups, int k,
		boolean inclUncompressed) {
		// special case: single uncompressed col group
		if(colGroups.size() == 1 && colGroups.get(0) instanceof ColGroupUncompressed) {
			return new ArrayList[0];
		}

		// initialize round robin col group distribution
		// (static task partitioning to reduce mem requirements/final agg)
		int numTasks = Math.min(k, colGroups.size());
		ArrayList<ColGroup>[] grpParts = new ArrayList[numTasks];
		int pos = 0;
		for(ColGroup grp : colGroups) {
			if(grpParts[pos] == null)
				grpParts[pos] = new ArrayList<>();
			if(inclUncompressed || !(grp instanceof ColGroupUncompressed)) {
				grpParts[pos].add(grp);
				pos = (pos == numTasks - 1) ? 0 : pos + 1;
			}
		}

		return grpParts;
	}

	public boolean hasUncompressedColGroup() {
		return getUncompressedColGroup() != null;
	}

	private ColGroupUncompressed getUncompressedColGroup() {
		for(ColGroup grp : _colGroups)
			if(grp instanceof ColGroupUncompressed)
				return (ColGroupUncompressed) grp;

		return null;
	}

	public Pair<Integer, int[]> getMaxNumValues() {
		if(v == null) {

			int numVals = 1;
			int[] numValues = new int[_colGroups.size()];
			int nr;
			for(int i = 0; i < _colGroups.size(); i++)
				if(_colGroups.get(i) instanceof ColGroupValue) {
					nr = ((ColGroupValue) _colGroups.get(i)).getNumValues();
					numValues[i] = nr;
					numVals = Math.max(numVals, nr);
				}
				else {
					numValues[i] = -1;
				}
			v = new ImmutablePair<>(numVals, numValues);
			return v;
		}
		else {
			return v;
		}
	}

	private static class UnaryAggregateTask implements Callable<MatrixBlock> {
		private final List<ColGroup> _groups;
		private final int _rl;
		private final int _ru;
		private final MatrixBlock _ret;
		private final AggregateUnaryOperator _op;

		protected UnaryAggregateTask(List<ColGroup> groups, MatrixBlock ret, int rl, int ru,
			AggregateUnaryOperator op) {
			_groups = groups;
			_op = op;
			_rl = rl;
			_ru = ru;

			if(_op.indexFn instanceof ReduceAll) { // sum
				_ret = new MatrixBlock(ret.getNumRows(), ret.getNumColumns(), false);
				_ret.allocateDenseBlock();
				if(_op.aggOp.increOp.fn instanceof Builtin)
					System.arraycopy(ret.getDenseBlockValues(),
						0,
						_ret.getDenseBlockValues(),
						0,
						ret.getNumRows() * ret.getNumColumns());
			}
			else { // colSums
				_ret = ret;
			}
		}

		@Override
		public MatrixBlock call() {
			aggregateUnaryOperations(_op, _groups, _ret, _rl, _ru);
			return _ret;
		}
	}

	private static class DecompressTask implements Callable<Long> {
		private final List<ColGroup> _colGroups;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;

		protected DecompressTask(List<ColGroup> colGroups, MatrixBlock ret, int rl, int ru) {
			_colGroups = colGroups;
			_ret = ret;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Long call() {

			// preallocate sparse rows to avoid repeated alloc

			if(_ret.isInSparseFormat()) {
				int[] rnnz = new int[_ru - _rl];
				for(ColGroup grp : _colGroups)
					grp.countNonZerosPerRow(rnnz, _rl, _ru);
				SparseBlock rows = _ret.getSparseBlock();
				for(int i = _rl; i < _ru; i++)
					rows.allocate(i, rnnz[i - _rl]);
			}

			// decompress row partition
			for(ColGroup grp : _colGroups)
				grp.decompressToBlock(_ret, _rl, _ru);

			// post processing (sort due to append)
			if(_ret.isInSparseFormat())
				_ret.sortSparseRows(_rl, _ru);

			return _ret.recomputeNonZeros(_rl, _ru - 1);
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("\nCompressed Matrix:");
		sb.append("\nCols:" + getNumColumns() + " Rows:" + getNumRows());
		for(ColGroup cg : _colGroups) {
			sb.append("\n" + cg);
		}
		return sb.toString();
	}

	public boolean isOverlapping() {
		return overlappingColGroups;
	}

	public void setOverlapping(boolean overlapping) {
		overlappingColGroups = overlapping;
	}
}

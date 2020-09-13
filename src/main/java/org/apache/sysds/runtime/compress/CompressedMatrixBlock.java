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
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConverter;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupIO;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.compress.colgroup.DenseRowIterator;
import org.apache.sysds.runtime.compress.colgroup.SparseRowIterator;
import org.apache.sysds.runtime.compress.utils.ColumnGroupIterator;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell.BinaryAccessType;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

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
		MatrixBlock ret = new MatrixBlock(getNumRows(), getNumColumns(), isInSparseFormat(), getNonZeros());
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
		ret.setNonZeros(nonZeros);
		if(ret.isInSparseFormat())
			ret.sortSparseRows();

		if(LOG.isDebugEnabled())
			LOG.debug("decompressed block in " + time.stop() + "ms.");

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

		MatrixBlock ret = new MatrixBlock(rlen, clen, sparse, nonZeros).allocateBlock();

		// multi-threaded decompression
		try {
			ExecutorService pool = CommonThreadPool.get(k);
			int rlen = getNumRows();
			int blklen = getAlignedBlockSize((int) (Math.ceil((double) rlen / k)));
			ArrayList<DecompressTask> tasks = new ArrayList<>();
			for(int i = 0; i < k & i * blklen < getNumRows(); i++)
				tasks.add(new DecompressTask(_colGroups, ret, i * blklen, Math.min((i + 1) * blklen, rlen)));
			List<Future<Object>> rtasks = pool.invokeAll(tasks);
			pool.shutdown();
			for(Future<Object> rt : rtasks)
				rt.get(); // error handling
		}
		catch(InterruptedException | ExecutionException ex) {
			LOG.error("Parallel decompression failed defaulting to non parallel implementation " + ex.getMessage());
			return decompress();
		}

		// post-processing
		ret.setNonZeros(nonZeros);

		LOG.debug("decompressed block w/ k=" + k + " in " + time.stop() + "ms.");

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

		ColGroup grp = null;
		for(ColGroup group : _colGroups) {

			if(Arrays.binarySearch(group.getColIndices(), c) >= 0) {
				grp = group;
				break;
			}
		}

		// find row value
		return grp.get(r, c);
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
		_colGroups = ColGroupIO.readGroups(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		// serialize compressed matrix block
		out.writeInt(rlen);
		out.writeInt(clen);
		out.writeLong(nonZeros);
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

		// allocate the output matrix block
		CompressedMatrixBlock ret = null;
		if(result == null || !(result instanceof CompressedMatrixBlock))
			ret = new CompressedMatrixBlock(getNumRows(), getNumColumns(), sparse);
		else {
			ret = (CompressedMatrixBlock) result;
			ret.reset(rlen, clen);
		}

		int threads = OptimizerUtils.getConstrainedNumThreads(_colGroups.size());

		if(threads > 1) {
			ExecutorService pool = CommonThreadPool.get(sop.getNumThreads());
			ArrayList<ScalarTask> tasks = new ArrayList<>();

			ArrayList<ColGroup> small = new ArrayList<>();

			for(ColGroup grp : _colGroups) {
				if(grp instanceof ColGroupUncompressed) {
					ArrayList<ColGroup> uc = new ArrayList<>();
					uc.add(grp);
					tasks.add(new ScalarTask(uc, sop));
				}
				else {
					int nv = ((ColGroupValue) grp).getNumValues();
					if(nv < 256) {
						small.add(grp);
					}
					else {
						ArrayList<ColGroup> large = new ArrayList<>();
						large.add(grp);
						tasks.add(new ScalarTask(large, sop));

					}
				}
				if(small.size() > 10) {
					tasks.add(new ScalarTask(small, sop));
					small = new ArrayList<>();
				}
			}
			if(small.size() > 0) {
				tasks.add(new ScalarTask(small, sop));
			}
			try {
				List<Future<List<ColGroup>>> rtasks = pool.invokeAll(tasks);
				pool.shutdown();

				ArrayList<ColGroup> newColGroups = new ArrayList<>();
				for(Future<List<ColGroup>> f : rtasks) {
					for(ColGroup x : f.get()) {
						newColGroups.add(x);
					}
				}
				ret._colGroups = newColGroups;
				ret.setNonZeros(rlen * clen);
			}
			catch(InterruptedException | ExecutionException e) {
				LOG.fatal("UnaryAggregate Exception: " + e.getMessage(), e);
				throw new DMLRuntimeException(e);
			}
		}
		else {

			// Apply the operation to each of the column groups.
			// Most implementations will only modify metadata.
			ArrayList<ColGroup> newColGroups = new ArrayList<>();
			for(ColGroup grp : _colGroups) {
				newColGroups.add(grp.scalarOperation(sop));
			}
			ret._colGroups = newColGroups;
			ret.setNonZeros(rlen * clen);
		}

		return ret;

	}

	protected void binaryMV(MatrixBlock m2, CompressedMatrixBlock ret, BinaryOperator op, BinaryAccessType aType) {
		if(aType == BinaryAccessType.MATRIX_COL_VECTOR) {
			throw new NotImplementedException("Binary Matrix Col Vector operations are not implemented CLA");
		}
		else if(aType == BinaryAccessType.MATRIX_ROW_VECTOR) {
			// Apply the operation to each of the column groups.
			// Most implementations will only modify metadata.
			ArrayList<ColGroup> newColGroups = new ArrayList<>();

			for(ColGroup grp : _colGroups) {
				if(grp instanceof ColGroupUncompressed) {
					LOG.error("NOT HANDLING UNCOMPRESSED IN BINARY MV");
				}
				else {

					if(grp.getNumCols() == 1) {
						ScalarOperator sop = new LeftScalarOperator(op.fn, m2.getValue(0, grp.getColIndices()[0]), 1);
						newColGroups.add(grp.scalarOperation(sop));
					}
					else {
						throw new NotImplementedException("Cocoded columns (nr cols:" + grp.getNumCols()
							+ ") groupType: not implemented for Binary Matrix Row Vector operations");
					}
				}
			}
			ret._colGroups = newColGroups;
		}
	}

	protected void binaryVV(MatrixBlock m2, CompressedMatrixBlock ret, BinaryOperator op, BinaryAccessType aType) {
		throw new NotImplementedException("Binary Vector Vector operations are not implemented");
	}

	protected void binaryMM(MatrixBlock m2, CompressedMatrixBlock ret, BinaryOperator op) {
		throw new NotImplementedException("Binary Matrix Matrix operations are not implemented");
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

		if(this.getNumColumns() != v.getNumRows())
			throw new DMLRuntimeException(
				"Dimensions mismatch on mmchain operation (" + this.getNumColumns() + " != " + v.getNumRows() + ")");
		if(v.getNumColumns() != 1)
			throw new DMLRuntimeException(
				"Invalid input vector (column vector expected, but ncol=" + v.getNumColumns() + ")");
		if(w != null && w.getNumColumns() != 1)
			throw new DMLRuntimeException(
				"Invalid weight vector (column vector expected, but ncol=" + w.getNumColumns() + ")");

		// single-threaded MMChain of single uncompressed ColGroup
		if(isSingleUncompressedGroup()) {
			return ((ColGroupUncompressed) _colGroups.get(0)).getData().chainMatrixMultOperations(v, w, out, ctype);
		}

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
		rightMultByVector(v, tmp);
		if(ctype == ChainType.XtwXv) {
			BinaryOperator bop = new BinaryOperator(Multiply.getMultiplyFnObject());
			LibMatrixBincell.bincellOpInPlace(tmp, w, bop);
		}
		leftMultByVectorTranspose(_colGroups, tmp, out, true, true);

		return out;
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
		rightMultByVector(v, tmp, k);
		if(ctype == ChainType.XtwXv) {
			BinaryOperator bop = new BinaryOperator(Multiply.getMultiplyFnObject());
			LibMatrixBincell.bincellOpInPlace(tmp, w, bop);
		}
		leftMultByVectorTranspose(_colGroups, tmp, out, true, k);

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
		if(ret == null)
			ret = new MatrixBlock(rl, cl, false, rl * cl);
		else
			ret.reset(rl, cl, false, rl * cl);

		if(right) {
			if(that.getNumColumns() == 1) {
				// Right Matrix Vector Multiplication
				if(op.getNumThreads() > 1)
					rightMultByVector(that, ret, op.getNumThreads());
				else
					rightMultByVector(that, ret);
			}
			else {
				that = that instanceof CompressedMatrixBlock ? ((CompressedMatrixBlock) that).decompress() : that;
				ret = rightMultByMatrix(_colGroups, that, ret, op.getNumThreads(), getNumColumns());
			}
		}
		else{ // Left
			that = that instanceof CompressedMatrixBlock ? ((CompressedMatrixBlock) that).decompress() : that;
			if(that.getNumRows() == 1){
				if(op.getNumThreads() > 1)
					leftMultByVectorTranspose(_colGroups, that, ret, false, op.getNumThreads());
				else
					leftMultByVectorTranspose(_colGroups, that, ret, false, true);
			}else{
				leftMultByMatrix(_colGroups, that, ret, op.getNumThreads(), 1);
			}
		}

		return ret;
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
					int blklen = getAlignedBlockSize((int) (Math.ceil((double) rlen / op.getNumThreads())));
					for(int i = 0; i < op.getNumThreads() & i * blklen < rlen; i++)
						tasks.add(
							new UnaryAggregateTask(grpParts[0], ret, i * blklen, Math.min((i + 1) * blklen, rlen), op));
				}
				else
					for(ArrayList<ColGroup> grp : grpParts)
						tasks.add(new UnaryAggregateTask(grp, ret, 0, rlen, op));
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
			// process UC column group
			for(ColGroup grp : _colGroups)
				if(grp instanceof ColGroupUncompressed)
					((ColGroupUncompressed) grp).unaryAggregateOperations(op, ret);

			// process OLE/RLE column groups
			aggregateUnaryOperations(op, _colGroups, ret, 0, rlen);
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

		// Seems misplaced logic for when to use CacheDDC
		// boolean cacheDDC1 = false;
		// op.indexFn instanceof ReduceCol && op.aggOp.increOp.fn instanceof KahanPlus // rowSums
		// && ColGroupOffset.ALLOW_CACHE_CONSCIOUS_ROWSUMS && ru - rl > CompressionSettings.BITMAP_BLOCK_SZ;

		// process cache-conscious DDC1 groups (adds to output)
		// if(cacheDDC1) {
		// ArrayList<ColGroupDDC1> tmp = new ArrayList<>();
		// for(ColGroup grp : groups)
		// if(grp instanceof ColGroupDDC1)
		// tmp.add((ColGroupDDC1) grp);
		// if(!tmp.isEmpty())
		// ColGroupDDC1
		// .computeRowSums(tmp.toArray(new ColGroupDDC1[0]), ret, KahanPlus.getKahanPlusFnObject(), rl, ru);
		// }

		// process remaining groups (adds to output)
		// note: UC group never passed into this function
		double[] c = ret.getDenseBlockValues();
		if(c == null) {
			c = ret.getSparseBlock().values(0);
			// throw new RuntimeException("aggregateUnaryOperation failed to materialize matrix data");
		}
		for(ColGroup grp : groups)
			if(!(grp instanceof ColGroupUncompressed))
				grp.unaryAggregateOperations(op, c, rl, ru);
		// LOG.debug(Arrays.toString(c));
	}

	@Override
	public MatrixBlock transposeSelfMatrixMultOperations(MatrixBlock out, MMTSJType tstype) {

		Timing time = LOG.isDebugEnabled() ? new Timing(true) : null;

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
			leftMultByTransposeSelf(_colGroups, out, 0, _colGroups.size());

			// post-processing
			out.setNonZeros(LinearAlgebraUtils.copyUpperToLowerTriangle(out));
		}

		if(LOG.isDebugEnabled())
			LOG.debug("Compressed TSMM in " + time.stop());

		return out;
	}

	@Override
	public MatrixBlock transposeSelfMatrixMultOperations(MatrixBlock out, MMTSJType tstype, int k) {

		if(k <= 1) {
			return transposeSelfMatrixMultOperations(out, tstype);
		}

		Timing time = LOG.isDebugEnabled() ? new Timing(true) : null;

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
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<MatrixMultTransposeTask> tasks = new ArrayList<>();
				int numgrp = _colGroups.size();
				int blklen = (int) (Math.ceil((double) numgrp / (2 * k)));
				for(int i = 0; i < 2 * k & i * blklen < clen; i++)
					tasks.add(
						new MatrixMultTransposeTask(_colGroups, out, i * blklen, Math.min((i + 1) * blklen, numgrp)));
				List<Future<Object>> ret = pool.invokeAll(tasks);
				for(Future<Object> tret : ret)
					tret.get(); // check for errors
				pool.shutdown();
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}

			// post-processing
			out.setNonZeros(LinearAlgebraUtils.copyUpperToLowerTriangle(out));
		}

		if(LOG.isDebugEnabled())
			LOG.debug("Compressed TSMM k=" + k + " in " + time.stop());

		return out;
	}

	/**
	 * Multiply this matrix block by a column vector on the right.
	 * 
	 * @param vector right-hand operand of the multiplication
	 * @param result buffer to hold the result; must have the appropriate size already
	 */
	private void rightMultByVector(MatrixBlock vector, MatrixBlock result) {
		// initialize and allocate the result
		result.allocateDenseBlock();

		// delegate matrix-vector operation to each column group
		rightMultByVector(_colGroups, vector, result, 0, result.getNumRows());

		// post-processing
		result.recomputeNonZeros();
	}

	/**
	 * Multi-threaded version of rightMultByVector.
	 * 
	 * @param vector matrix block vector
	 * @param result matrix block result
	 * @param k      number of threads
	 */
	private void rightMultByVector(MatrixBlock vector, MatrixBlock result, int k) {
		// initialize and allocate the result
		result.allocateDenseBlock();

		// multi-threaded execution of all groups
		try {
			// ColGroupUncompressed uc = getUncompressedColGroup();

			// compute uncompressed column group in parallel
			// if(uc != null)
			// uc.rightMultByVector(vector, result, k);

			// compute remaining compressed column groups in parallel
			// note: OLE needs alignment to segment size, otherwise wrong entry
			ExecutorService pool = CommonThreadPool.get(k);
			int rlen = getNumRows();
			int seqsz = CompressionSettings.BITMAP_BLOCK_SZ;
			int blklen = (int) (Math.ceil((double) rlen / k));
			blklen += (blklen % seqsz != 0) ? seqsz - blklen % seqsz : 0;

			ArrayList<RightMatrixMultTask> tasks = new ArrayList<>();
			for(int i = 0; i < k & i * blklen < getNumRows(); i++) {
				tasks.add(
					new RightMatrixMultTask(_colGroups, vector, result, i * blklen, Math.min((i + 1) * blklen, rlen)));
			}

			List<Future<Long>> ret = pool.invokeAll(tasks);
			pool.shutdown();

			// error handling and nnz aggregation
			long lnnz = 0;
			for(Future<Long> tmp : ret)
				lnnz += tmp.get();
			result.setNonZeros(lnnz);
		}
		catch(Exception e) {
			LOG.error(e);
			throw new DMLRuntimeException(e);
		}
	}

	private static void rightMultByVector(List<ColGroup> groups, MatrixBlock vect, MatrixBlock ret, int rl, int ru) {
		// + 1 to enable containing a single 0 value in the dictionary that was not materialized.
		// This is to handle the case of a DDC dictionary not materializing the zero values.
		// A fine tradeoff!
		ColGroupValue.setupThreadLocalMemory(getMaxNumValues(groups).getLeft() + 1);

		// boolean cacheDDC1 = ru - rl > CompressionSettings.BITMAP_BLOCK_SZ * 2;

		// process uncompressed column group (overwrites output)
		// if(inclUC) {
		for(ColGroup grp : groups) {
			if(grp instanceof ColGroupUncompressed)
				((ColGroupUncompressed) grp).rightMultByVector(vect, ret, rl, ru);
		}

		// process cache-conscious DDC1 groups (adds to output)

		// if(cacheDDC1) {
		// ArrayList<ColGroupDDC1> tmp = new ArrayList<>();
		// for(ColGroup grp : groups)
		// if(grp instanceof ColGroupDDC1)
		// tmp.add((ColGroupDDC1) grp);
		// if(!tmp.isEmpty())
		// ColGroupDDC1.rightMultByVector(tmp.toArray(new ColGroupDDC1[0]), vect, ret, rl, ru);
		// }
		// process remaining groups (adds to output)
		double[] values = ret.getDenseBlockValues();
		for(ColGroup grp : groups) {
			if(!(grp instanceof ColGroupUncompressed)) {
				grp.rightMultByVector(vect.getDenseBlockValues(), values, rl, ru, grp.getValues());
			}
		}

		ColGroupValue.cleanupThreadLocalMemory();

	}

	/**
	 * Multiply this matrix block by the transpose of a column vector (i.e. t(v)%*%X)
	 * 
	 * @param colGroups   list of column groups
	 * @param vector      left-hand operand of the multiplication
	 * @param result      buffer to hold the result; must have the appropriate size already
	 * @param doTranspose if true, transpose vector
	 */
	private static void leftMultByVectorTranspose(List<ColGroup> colGroups, MatrixBlock vector, MatrixBlock result,
		boolean doTranspose, boolean allocTmp) {
		// transpose vector if required
		LOG.debug("Left Mult vector Transpose " + vector.getClass());
		MatrixBlock rowVector = vector;
		if(doTranspose) {
			rowVector = new MatrixBlock(1, vector.getNumRows(), false);
			LibMatrixReorg.transpose(vector, rowVector);
		}

		// initialize and allocate the result
		result.reset();
		result.allocateDenseBlock();

		// setup memory pool for reuse
		if(allocTmp) {
			Pair<Integer, List<Integer>> v = getMaxNumValues(colGroups);
			ColGroupValue.setupThreadLocalMemory(v.getLeft());
			for(int i = 0; i < colGroups.size(); i++) {
				colGroups.get(i).leftMultByRowVector(rowVector.getDenseBlockValues(),
					result.getDenseBlockValues(),
					v.getRight().get(i));
			}
		}
		else {
			for(ColGroup grp : colGroups) {
				grp.leftMultByRowVector(rowVector.getDenseBlockValues(), result.getDenseBlockValues(), -1);
			}
		}

		// delegate matrix-vector operation to each column group

		// post-processing
		if(allocTmp)
			ColGroupValue.cleanupThreadLocalMemory();
		result.recomputeNonZeros();
	}

	// private static void leftMultByVectorTranspose(List<ColGroup> colGroups, ColGroupDDC vector, MatrixBlock result) {
	// // initialize and allocate the result
	// result.reset();
	// // delegate matrix-vector operation to each column group
	// for(ColGroup grp : colGroups)
	// grp.leftMultByRowVector(vector, result);
	// // post-processing
	// result.recomputeNonZeros();
	// }

	/**
	 * Multi-thread version of leftMultByVectorTranspose.
	 * 
	 * @param colGroups   list of column groups
	 * @param vector      left-hand operand of the multiplication
	 * @param result      buffer to hold the result; must have the appropriate size already
	 * @param doTranspose if true, transpose vector
	 * @param k           number of threads
	 */
	private void leftMultByVectorTranspose(List<ColGroup> colGroups, MatrixBlock vector, MatrixBlock result,
		boolean doTranspose, int k) {
		// transpose vector if required
		MatrixBlock rowVector = vector;
		if(doTranspose) {
			rowVector = new MatrixBlock(1, vector.getNumRows(), false);
			LibMatrixReorg.transpose(vector, rowVector);
		}

		// initialize and allocate the result
		result.reset();
		result.allocateDenseBlock();

		// multi-threaded execution
		try {
			// compute uncompressed column group in parallel
			// ColGroupUncompressed uc = getUncompressedColGroup();
			// if(uc != null)
			// uc.leftMultByRowVector(rowVector, result, k);

			// compute remaining compressed column groups in parallel
			ExecutorService pool = CommonThreadPool.get(Math.min(colGroups.size(), k));
			ArrayList<ColGroup>[] grpParts = createStaticTaskPartitioning(_colGroups, 4 * k, true);
			ArrayList<LeftMatrixVectorMultTask> tasks = new ArrayList<>();
			for(ArrayList<ColGroup> groups : grpParts)
				tasks.add(new LeftMatrixVectorMultTask(groups, rowVector, result));
			List<Future<Object>> ret;

			ret = pool.invokeAll(tasks);

			pool.shutdown();
			for(Future<Object> tmp : ret)
				tmp.get();

		}
		catch(InterruptedException | ExecutionException e) {
			LOG.error(e);
			throw new DMLRuntimeException(e);
		}

		// post-processing
		result.recomputeNonZeros();
	}

	/**
	 * Multiply this matrix block by a matrix (i.e. v%*%X)
	 * 
	 * @param colGroups  List of column groups
	 * @param that       Left-hand operand of the multiplication
	 * @param ret        The result matrix to insert the results
	 * @param tmp        buffer to hold the result; must have the appropriate size already
	 * @param tmpIn      buffer to hold a since row of input.
	 * @param k          The number of threads used
	 * @param numColumns The number of columns in this colGroup
	 */
	private static void leftMultByMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		int numColumns) {

		ret.reset();
		ret.allocateDenseBlock();

		if(that.isInSparseFormat()) {
			LOG.warn("Inefficient materialization of sparse matrix for left compressed matrix mult.");

			// leftMultBySparseMatrix(colGroups, that, ret, k, numColumns);
			leftMultByDenseMatrix(colGroups, that.allocateDenseBlock(), ret, k, numColumns);
		}
		else {
			leftMultByDenseMatrix(colGroups, that, ret, k, numColumns);
		}

		ret.recomputeNonZeros();
	}

	private static MatrixBlock rightMultByMatrix(List<ColGroup> colGroups,
		MatrixBlock that, MatrixBlock ret, int k, int numColumns)
	{
		// Exchange with rightMultByDenseMatrix.

		ret.reset();
		ret.allocateDenseBlock();

		// transpose input to optimize column access.
		that = LibMatrixReorg
			.transpose(that, new MatrixBlock(that.getNumColumns(), that.getNumRows(), that.isInSparseFormat()), k);

		if(that.isInSparseFormat()) {
			LOG.warn("Inefficient materialization of sparse matrix for right compressed matrix mult.");
			rightMultByDenseMatrix(colGroups, that.allocateDenseBlock(), ret, k, numColumns);
		}
		else {
			rightMultByDenseMatrix(colGroups, that, ret, k, numColumns);
		}
		ret.recomputeNonZeros();
		return ret;

		// ret = LibMatrixReorg
		// .transpose(ret, new MatrixBlock(ret.getNumColumns(), ret.getNumRows(), ret.isInSparseFormat()), k);

		// MatrixBlock tmpIn = new MatrixBlock(1, that.getNumColumns(), false).allocateBlock();
		// MatrixBlock tmpOut = new MatrixBlock(this.getNumRows(), 1, false).allocateBlock();
		// for(int i = 0; i < that.getNumRows(); i++) { // on transpose
		// tmpIn = that.slice(i, i, 0, that.getNumColumns() - 1, tmpIn);
		// MatrixBlock tmpIn2 = LibMatrixReorg // meta data op
		// .transpose(tmpIn, new MatrixBlock(tmpIn.getNumColumns(), tmpIn.getNumRows(), false));
		// tmpOut.reset();
		// if(k > 1)
		// rightMultByVector(tmpIn2, tmpOut, k);
		// else
		// rightMultByVector(tmpIn2, tmpOut);
		// MatrixBlock tmpOutT = LibMatrixReorg // meta data op
		// .transpose(tmpOut, new MatrixBlock(tmpOut.getNumColumns(), tmpOut.getNumRows(), false), k);
		// ret.leftIndexingOperations(tmpOutT, i, i, 0, ret.getNumColumns() - 1, ret, UpdateType.INPLACE);
		// }
		// ret.checkNonZeros();
		// ret = LibMatrixReorg
		// .transpose(ret, new MatrixBlock(ret.getNumColumns(), ret.getNumRows(), ret.isInSparseFormat()), k);
		// return ret;
	}

	private static void leftMultByDenseMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		int numColumns) {
		DenseBlock db = that.getDenseBlock();
		double[] retV = ret.getDenseBlockValues();
		double[] thatV;
		int blockU;
		int blockL = 0;

		for(ColGroup grp : colGroups) {
			if(grp instanceof ColGroupUncompressed)
				((ColGroupUncompressed) grp).leftMultByMatrix(that, ret);
		}

		for(int b = 0; b <= db.numBlocks(); b++) {
			int blockSize = db.blockSize(b);
			blockU = Math.min(blockL + blockSize, ret.getNumRows());
			thatV = db.valuesAt(b);

			if(k == 1) {
				// OPTIMIZE dont allocate colGroups for the uncompressable columns.
				// Optimize allocate once the materialization for all the blocks.
				double[][] materialized = new double[colGroups.size()][];
				for(int i = 0; i < colGroups.size(); i++) {
					materialized[i] = colGroups.get(i).getValues();
				}
				Pair<Integer, List<Integer>> v = getMaxNumValues(colGroups);
				for(int j = 0; j < colGroups.size(); j++)
					colGroups.get(j).leftMultByMatrix(thatV,
						retV,
						v.getRight().get(j),
						materialized[j],
						that.getNumRows(),
						ret.getNumColumns(),
						blockL,
						blockU,
						0);
			}
			else {
				try {
					ExecutorService pool = CommonThreadPool.get(Math.min(colGroups.size(), k));
					// compute remaining compressed column groups in parallel
					ArrayList<LeftMatrixMatrixMultTask> tasks = new ArrayList<>();
					List<ColGroup>[] parts = createStaticTaskPartitioningForMatrixMult(colGroups, k, false);
					int rowBlockSize = 10;
					for(List<ColGroup> part : parts) {
						for(int blo = blockL; blo < blockU; blo += rowBlockSize) {
							tasks.add(new LeftMatrixMatrixMultTask(part, thatV, retV, that.getNumRows(), numColumns,
								blo, Math.min(blo + rowBlockSize, blockU), blo - blockL));

						}
					}

					List<Future<Object>> futures = pool.invokeAll(tasks);

					pool.shutdown();
					for(Future<Object> future : futures)
						future.get();
				}
				catch(InterruptedException | ExecutionException e) {
					LOG.error(e);
					throw new DMLRuntimeException(e);
				}
			}
			blockL += blockSize;
		}
	}

	// private static void leftMultBySparseMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
	// int numColumns) {
	// // SparseBlock sb = that.getSparseBlock();
	// throw new NotImplementedException("Sparse Block input not handled.");
	// }

	private static void rightMultByDenseMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		int numColumns) {
		DenseBlock db = that.getDenseBlock();
		double[] retV = ret.getDenseBlockValues();
		double[] thatV;
		int blockU;
		int blockL = 0;

		for(ColGroup grp : colGroups) {
			if(grp instanceof ColGroupUncompressed)
				((ColGroupUncompressed) grp).rightMultByMatrix(that, ret, 0, ret.getNumRows());
		}

		// OPTIMIZE dont allocate colGroups for the uncompressable columns.
		double[][] materialized = new double[colGroups.size()][];

		for(int b = 0; b <= db.numBlocks(); b++) {
			int blockSize = db.blockSize(b);
			blockU = Math.min(blockL + blockSize, ret.getNumRows());
			thatV = db.valuesAt(b);

			for(int i = 0; i < colGroups.size(); i++) {
				materialized[i] = colGroups.get(i).getValues();
			}
			Pair<Integer, List<Integer>> v = getMaxNumValues(colGroups);
			for(int j = 0; j < colGroups.size(); j++) {
				colGroups.get(j)
					.rightMultByMatrix(thatV, retV, v.getRight().get(j), materialized[j], blockL, blockU, 0);
			}
		}
	}

	private static void leftMultByTransposeSelf(List<ColGroup> groups, MatrixBlock result, int gl, int gu) {
		final int numRows = groups.get(0).getNumRows();
		final int numGroups = groups.size();
		// final boolean containsUC = containsUncompressedColGroup(groups);

		// preallocated dense tmp matrix blocks
		MatrixBlock lhs = new MatrixBlock(1, numRows, false);
		MatrixBlock tmpret = new MatrixBlock(1, result.getNumColumns(), false);
		lhs.allocateDenseBlock();
		tmpret.allocateDenseBlock();

		// setup memory pool for reuse
		ColGroupValue.setupThreadLocalMemory(getMaxNumValues(groups).getLeft());

		// approach: for each colgroup, extract uncompressed columns one at-a-time
		// vector-matrix multiplies against remaining col groups
		for(int i = gl; i < gu; i++) {
			// get current group and relevant col groups
			ColGroup group = groups.get(i);
			int[] ixgroup = group.getColIndices();
			List<ColGroup> tmpList = groups.subList(i, numGroups);

			// if(group instanceof ColGroupDDC // single DDC group
			// && ixgroup.length == 1 && !containsUC && numRows < CompressionSettings.BITMAP_BLOCK_SZ) {
			// // compute vector-matrix partial result
			// leftMultByVectorTranspose(tmpList, (ColGroupDDC) group, tmpret);

			// // write partial results (disjoint non-zeros)
			// LinearAlgebraUtils.copyNonZerosToUpperTriangle(result, tmpret, ixgroup[0]);
			// }
			// else {
			// for all uncompressed lhs columns vectors
			for(int j = 0; j < ixgroup.length; j++) {
				group.decompressToBlock(lhs, j);

				if(!lhs.isEmptyBlock(false)) {
					// compute vector-matrix partial result
					leftMultByVectorTranspose(tmpList, lhs, tmpret, false, false);

					// write partial results (disjoint non-zeros)
					LinearAlgebraUtils.copyNonZerosToUpperTriangle(result, tmpret, ixgroup[j]);
				}
			}
			// }
		}

		// post processing
		ColGroupValue.cleanupThreadLocalMemory();
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

	@SuppressWarnings("unchecked")
	private static List<ColGroup>[] createStaticTaskPartitioningForMatrixMult(List<ColGroup> colGroups, int k,
		boolean inclUncompressed) {
		int numTasks = Math.min(k, colGroups.size());
		List<ColGroup>[] grpParts = new ArrayList[numTasks];
		int pos = 0;
		for(int i = 0; i < numTasks; i++) {
			grpParts[pos++] = new ArrayList<>();
		}
		pos = 0;
		for(ColGroup grp : colGroups) {

			if(grp instanceof ColGroupDDC) {
				grpParts[pos].add((ColGroupDDC) grp);
				pos = (pos == numTasks - 1) ? 0 : pos + 1;
			}
		}
		for(ColGroup grp : colGroups) {
			if(!(grp instanceof ColGroupDDC) && (inclUncompressed || !(grp instanceof ColGroupUncompressed))) {
				grpParts[pos].add(grp);
				pos = (pos == numTasks - 1) ? 0 : pos + 1;
			}
		}

		return grpParts;
	}

	private static Pair<Integer, List<Integer>> getMaxNumValues(List<ColGroup> groups) {
		int numVals = 1;
		List<Integer> numValues = new ArrayList<>(groups.size());
		int nr;
		for(ColGroup grp : groups)
			if(grp instanceof ColGroupValue) {
				nr = ((ColGroupValue) grp).getNumValues();
				numValues.add(nr);
				numVals = Math.max(numVals, nr);
			}
			else {
				numValues.add(-1);
			}
		return new ImmutablePair<>(numVals, numValues);
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

	private static class LeftMatrixVectorMultTask implements Callable<Object> {
		private final ArrayList<ColGroup> _groups;
		private final MatrixBlock _vect;
		private final MatrixBlock _ret;

		protected LeftMatrixVectorMultTask(ArrayList<ColGroup> groups, MatrixBlock vect, MatrixBlock ret) {
			_groups = groups;
			_vect = vect;
			_ret = ret;
		}

		@Override
		public Object call() {
			// setup memory pool for reuse
			try {
				Pair<Integer, List<Integer>> v = getMaxNumValues(_groups);
				ColGroupValue.setupThreadLocalMemory(v.getLeft());
				for(int i = 0; i < _groups.size(); i++) {
					_groups.get(i).leftMultByRowVector(_vect.getDenseBlockValues(),
						_ret.getDenseBlockValues(),
						v.getRight().get(i));
				}

				ColGroupValue.cleanupThreadLocalMemory();
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	private static class LeftMatrixMatrixMultTask implements Callable<Object> {
		private final List<ColGroup> _group;
		private final double[] _that;
		private final double[] _ret;
		private final int _numRows;
		private final int _numCols;
		private final int _rl;
		private final int _ru;
		private final int _vOff;

		protected LeftMatrixMatrixMultTask(List<ColGroup> group, double[] that, double[] ret, int numRows, int numCols,
			int rl, int ru, int vOff) {
			_group = group;
			_that = that;
			_ret = ret;
			_numRows = numRows;
			_numCols = numCols;
			_rl = rl;
			_ru = ru;
			_vOff = vOff;
		}

		@Override
		public Object call() {
			// setup memory pool for reuse

			double[][] materialized = new double[_group.size()][];
			for(int i = 0; i < _group.size(); i++) {
				materialized[i] = _group.get(i).getValues();
			}
			Pair<Integer, List<Integer>> v = getMaxNumValues(_group);
			try {
				ColGroupValue.setupThreadLocalMemory(v.getLeft());
				for(int j = 0; j < _group.size(); j++) {
					_group.get(j).leftMultByMatrix(_that,
						_ret,
						v.getRight().get(j),
						materialized[j],
						_numRows,
						_numCols,
						_rl,
						_ru,
						_vOff);
				}
				ColGroupValue.cleanupThreadLocalMemory();

			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	private static class RightMatrixMultTask implements Callable<Long> {
		private final List<ColGroup> _groups;
		private final MatrixBlock _vect;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;

		protected RightMatrixMultTask(List<ColGroup> groups, MatrixBlock vect, MatrixBlock ret, int rl, int ru) {
			_groups = groups;
			_vect = vect;
			_ret = ret;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Long call() {
			try {
				rightMultByVector(_groups, _vect, _ret, _rl, _ru);
				return _ret.recomputeNonZeros(_rl, _ru - 1, 0, 0);
			}
			catch(Exception e) {
				LOG.error(e);
				throw new DMLRuntimeException(e);
			}
		}
	}

	private static class MatrixMultTransposeTask implements Callable<Object> {
		private final List<ColGroup> _groups;
		private final MatrixBlock _ret;
		private final int _gl;
		private final int _gu;

		protected MatrixMultTransposeTask(List<ColGroup> groups, MatrixBlock ret, int gl, int gu) {
			_groups = groups;
			_ret = ret;
			_gl = gl;
			_gu = gu;
		}

		@Override
		public Object call() {
			leftMultByTransposeSelf(_groups, _ret, _gl, _gu);
			return null;
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

	private static class DecompressTask implements Callable<Object> {
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
		public Object call() {

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

			return null;
		}
	}

	private static class ScalarTask implements Callable<List<ColGroup>> {
		private final List<ColGroup> _colGroups;
		private final ScalarOperator _sop;

		protected ScalarTask(List<ColGroup> colGroups, ScalarOperator sop) {
			_colGroups = colGroups;
			_sop = sop;
		}

		@Override
		public List<ColGroup> call() {
			List<ColGroup> res = new ArrayList<>();
			for(ColGroup x : _colGroups) {
				res.add(x.scalarOperation(_sop));
			}
			return res;
		}
	}

	/**
	 * Calculates the Aligned block size if the block is a certain length.
	 * 
	 * @param blklen The Entered block length
	 * @return The total size of aligned blocks rounded the entered value up to the next BITMAP_BLOCK_SZ
	 */
	private static int getAlignedBlockSize(int blklen) {
		return blklen + ((blklen % CompressionSettings.BITMAP_BLOCK_SZ != 0) ? CompressionSettings.BITMAP_BLOCK_SZ -
			blklen % CompressionSettings.BITMAP_BLOCK_SZ : 0);
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
}
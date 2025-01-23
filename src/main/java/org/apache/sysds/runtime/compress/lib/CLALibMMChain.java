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

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.utils.stats.Timing;

/**
 * Support compressed MM chain operation to fuse the following cases :
 * 
 * <p>
 * XtXv == (t(X) %*% (X %*% v))
 * </p>
 * 
 * <p>
 * XtwXv == (t(X) %*% (w * (X %*% v)))
 * </p>
 *
 * <p>
 * XtXvy == (t(X) %*% ((X %*% v) - y))
 * </p>
 */
public final class CLALibMMChain {
	static final Log LOG = LogFactory.getLog(CLALibMMChain.class.getName());

	/** Reusable cache intermediate double array for temporary decompression */
	private static ThreadLocal<double[]> cacheIntermediate = null;

	private CLALibMMChain() {
		// private constructor
	}

	/**
	 * Support compressed MM chain operation to fuse the following cases :
	 * 
	 * <p>
	 * XtXv == (t(X) %*% (X %*% v))
	 * </p>
	 * 
	 * <p>
	 * XtwXv == (t(X) %*% (w * (X %*% v)))
	 * </p>
	 *
	 * <p>
	 * XtXvy == (t(X) %*% ((X %*% v) - y))
	 * </p>
	 * 
	 * Note the point of this optimization is that v and w always are vectors. This means in practice the all the compute
	 * is faster if the intermediates are exploited.
	 * 
	 * 
	 * @param x     Is the X part of the chain optimized kernel
	 * @param v     Is the mandatory v part of the chain
	 * @param w     Is the optional w port of t the chain
	 * @param out   The output to put the result into. Can also be returned and in some cases will not be used.
	 * @param ctype either XtwXv, XtXv or XtXvy
	 * @param k     the parallelization degree
	 * @return The result either in the given output or a new allocation
	 */
	public static MatrixBlock mmChain(CompressedMatrixBlock x, MatrixBlock v, MatrixBlock w, MatrixBlock out,
		ChainType ctype, int k) {

		Timing t = new Timing();
		if(x.isEmpty())
			return returnEmpty(x, out);

		// Morph the columns to efficient types for the operation.
		x = filterColGroups(x);
		double preFilterTime = t.stop();

		// Allow overlapping intermediate if the intermediate is guaranteed not to be overlapping.
		final boolean allowOverlap = x.getColGroups().size() == 1 && isOverlappingAllowed();

		// Right hand side multiplication
		MatrixBlock tmp = CLALibRightMultBy.rightMultByMatrix(x, v, null, k, true);

		double rmmTime = t.stop();

		if(ctype == ChainType.XtwXv) { // Multiply intermediate with vector if needed
			tmp = binaryMultW(tmp, w, k);
		}

		if(!allowOverlap && tmp instanceof CompressedMatrixBlock) {
			tmp = decompressIntermediate((CompressedMatrixBlock) tmp, k);
		}

		double decompressTime = t.stop();

		if(tmp instanceof CompressedMatrixBlock)
			// Compressed Compressed Matrix Multiplication
			CLALibLeftMultBy.leftMultByMatrixTransposed(x, (CompressedMatrixBlock) tmp, out, k);
		else
			// LMM with Compressed - uncompressed multiplication.
			CLALibLeftMultBy.leftMultByMatrixTransposed(x, tmp, out, k);

		double lmmTime = t.stop();
		if(out.getNumColumns() != 1) // transpose the output to make it a row output if needed
			out = LibMatrixReorg.transposeInPlace(out, k);

		if(LOG.isDebugEnabled()) {
			StringBuilder sb = new StringBuilder("\n");
			sb.append("\nPreFilter Time      : " + preFilterTime);
			sb.append("\nChain RMM           : " + rmmTime);
			sb.append("\nChain RMM Decompress: " + decompressTime);
			sb.append("\nChain LMM           : " + lmmTime);
			sb.append("\nChain Transpose     : " + t.stop());
			LOG.debug(sb.toString());
		}

		return out;
	}

	private static MatrixBlock decompressIntermediate(CompressedMatrixBlock tmp, int k) {
		// cacheIntermediate
		final int rows = tmp.getNumRows();
		final int cols = tmp.getNumColumns();
		final int nCells = rows * cols;
		final double[] tmpArr;
		if(cacheIntermediate == null) {
			tmpArr = new double[nCells];
			cacheIntermediate = new ThreadLocal<>();
			cacheIntermediate.set(tmpArr);
		}
		else {
			double[] cachedArr = cacheIntermediate.get();
			if(cachedArr == null || cachedArr.length < nCells) {
				tmpArr = new double[nCells];
				cacheIntermediate.set(tmpArr);
			}
			else {
				tmpArr = cachedArr;
			}
		}

		final MatrixBlock tmpV = new MatrixBlock(tmp.getNumRows(), tmp.getNumColumns(), tmpArr);
		CLALibDecompress.decompressTo((CompressedMatrixBlock) tmp, tmpV, 0, 0, k, false, true);
		return tmpV;
	}

	private static boolean isOverlappingAllowed() {
		return ConfigurationManager.getDMLConfig().getBooleanValue(DMLConfig.COMPRESSED_OVERLAPPING);
	}

	private static MatrixBlock returnEmpty(CompressedMatrixBlock x, MatrixBlock out) {
		out = prepareReturn(x, out);
		return out;
	}

	private static MatrixBlock prepareReturn(CompressedMatrixBlock x, MatrixBlock out) {
		final int clen = x.getNumColumns();
		if(out != null)
			out.reset(clen, 1, false);
		else
			out = new MatrixBlock(clen, 1, false);
		return out;
	}

	private static MatrixBlock binaryMultW(MatrixBlock tmp, MatrixBlock w, int k) {
		final BinaryOperator bop = new BinaryOperator(Multiply.getMultiplyFnObject(), k);
		if(tmp instanceof CompressedMatrixBlock)
			tmp = CLALibBinaryCellOp.binaryOperationsRight(bop, (CompressedMatrixBlock) tmp, w);
		else
			LibMatrixBincell.bincellOpInPlace(tmp, w, bop);
		return tmp;
	}

	private static CompressedMatrixBlock filterColGroups(CompressedMatrixBlock x) {
		final List<AColGroup> groups = x.getColGroups();
		final boolean shouldFilter = CLALibUtils.shouldPreFilter(groups);
		if(shouldFilter) {
			if(CLALibUtils.alreadyPreFiltered(groups, x.getNumColumns()))
				return x;
			final int nCol = x.getNumColumns();
			final double[] constV = new double[nCol];
			final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(groups, constV);

			AColGroup c = ColGroupConst.create(constV);
			filteredGroups.add(c);
			x.allocateColGroupList(filteredGroups);
			return x;
		}
		else
			return x;
	}
}

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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

public class CLALibMatrixMult {
	private static final Log LOG = LogFactory.getLog(CLALibMatrixMult.class.getName());

	public static MatrixBlock matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) {
		return matrixMultiply(m1, m2, ret, k, false, false);
	}

	public static MatrixBlock matrixMultiply(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		int k, boolean transposeLeft, boolean transposeRight) {
		final Timing time = LOG.isTraceEnabled() ? new Timing(true) : null;

		if(m1 instanceof CompressedMatrixBlock && m2 instanceof CompressedMatrixBlock) {
			return doubleCompressedMatrixMultiply((CompressedMatrixBlock) m1, (CompressedMatrixBlock) m2, ret,
				k, transposeLeft, transposeRight);
		}

		boolean transposeOutput = false;
		if(transposeLeft || transposeRight) {

			if((m1 instanceof CompressedMatrixBlock && transposeLeft) ||
				(m2 instanceof CompressedMatrixBlock && transposeRight)) {
				// change operation from m1 %*% m2 -> t( t(m2) %*% t(m1) )
				transposeOutput = true;
				MatrixBlock tmp = m1;
				m1 = m2;
				m2 = tmp;
				boolean tmpLeft = transposeLeft;
				transposeLeft = !transposeRight;
				transposeRight = !tmpLeft;
			}

			if(!(m1 instanceof CompressedMatrixBlock) && transposeLeft) {
				m1 = LibMatrixReorg.transpose(m1, k);
				transposeLeft = false;
			}
			else if(!(m2 instanceof CompressedMatrixBlock) && transposeRight) {
				m2 = LibMatrixReorg.transpose(m2, k);
				transposeRight = false;
			}
		}

		final boolean right = (m1 instanceof CompressedMatrixBlock);
		final CompressedMatrixBlock c =(CompressedMatrixBlock) (right ? m1 : m2);
		final MatrixBlock that = right ? m2 : m1;

		// create output matrix block
		if(right)
			ret = CLALibRightMultBy.rightMultByMatrix(c, that, ret, k);
		else
			ret = CLALibLeftMultBy.leftMultByMatrix(c, that, ret, k);

		if(LOG.isTraceEnabled())
			LOG.trace("MM: Time block w/ sharedDim: " + m1.getNumColumns() + " rowLeft: " + m1.getNumRows() + " colRight:"
				+ m2.getNumColumns() + " in " + time.stop() + "ms.");

		if(transposeOutput) {
			if(ret instanceof CompressedMatrixBlock) {
				LOG.warn("Transposing decompression");
				ret = ((CompressedMatrixBlock) ret).decompress(k);
			}
			ret = LibMatrixReorg.transpose(ret, k);
		}

		return ret;
	}

	private static MatrixBlock doubleCompressedMatrixMultiply(CompressedMatrixBlock m1, CompressedMatrixBlock m2,
		MatrixBlock ret, int k, boolean transposeLeft, boolean transposeRight) {
		if(!transposeLeft && !transposeRight) {
			// If both are not transposed, decompress the right hand side. to enable
			// compressed overlapping output.
			LOG.warn("Matrix decompression from multiplying two compressed matrices.");
			return matrixMultiply(m1, CompressedMatrixBlock.getUncompressed(m2), ret, k, transposeLeft, transposeRight);
		}
		else if(transposeLeft && !transposeRight) {
			if(m1.getNumColumns() > m2.getNumColumns()) {
				ret = CLALibLeftMultBy.leftMultByMatrixTransposed(m1, m2, ret, k);
				ReorgOperator r_op = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), k);
				return ret.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0);
			}
			else
				return CLALibLeftMultBy.leftMultByMatrixTransposed(m2, m1, ret, k);

		}
		else if(!transposeLeft && transposeRight) {
			throw new DMLCompressionException("Not Implemented compressed Matrix Mult, to produce larger matrix");
			// worst situation since it blows up the result matrix in number of rows in
			// either compressed matrix.
		}
		else {
			ret = CLALibMatrixMult.matrixMult(m2, m1, ret, k);
			ReorgOperator r_op = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), k);
			return ret.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0);
		}
	}

}

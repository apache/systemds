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
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateTernaryOperator;

public final class CLALibAggTernaryOp {
	private static final Log LOG = LogFactory.getLog(CLALibAggTernaryOp.class.getName());

	private final MatrixBlock m1;
	private final MatrixBlock m2;
	private final MatrixBlock m3;
	private final MatrixBlock ret;
	private final AggregateTernaryOperator op;
	private final boolean inCP;

	private static boolean warned = false;

	private CLALibAggTernaryOp(MatrixBlock m1, MatrixBlock m2, MatrixBlock m3, MatrixBlock ret,
		AggregateTernaryOperator op, boolean inCP) {
		this.m1 = m1;
		this.m2 = m2;
		this.m3 = m3;
		this.ret = ret;
		this.op = op;
		this.inCP = inCP;
	}

	private MatrixBlock exec() {
		if(op.indexFn instanceof ReduceAll && op.aggOp.increOp.fn instanceof KahanPlus &&
			op.binaryFn instanceof Multiply) {
			// early abort if if anyEmpty.
			if(m1.isEmptyBlock(false) || m2.isEmptyBlock(false) || m3 != null && m3.isEmptyBlock(false))
				return ret;

			// if any is constant.
			if(isConst(m1)) {
				double v = m1.quickGetValue(0, 0);
				if(v == 1.0)
					return new CLALibAggTernaryOp(m2, m3, null, ret, op, inCP).exec();
			}
		}
		return fallBack();
	}

	private static boolean isConst(MatrixBlock m) {
		if(m != null && m instanceof CompressedMatrixBlock) {
			List<AColGroup> gs = ((CompressedMatrixBlock) m).getColGroups();
			return gs.size() == 1 && gs.get(0) instanceof ColGroupConst;
		}
		return false;
	}

	public static MatrixBlock agg(MatrixBlock m1, MatrixBlock m2, MatrixBlock m3, MatrixBlock ret,
		AggregateTernaryOperator op, boolean inCP) {
		return new CLALibAggTernaryOp(m1, m2, m3, ret, op, inCP).exec();
	}

	private MatrixBlock fallBack() {
		warnDecompression();
		MatrixBlock m1UC = CompressedMatrixBlock.getUncompressed(m1);
		MatrixBlock m2UC = CompressedMatrixBlock.getUncompressed(m2);
		MatrixBlock m3UC = CompressedMatrixBlock.getUncompressed(m3);

		MatrixBlock ret2 = m1UC.aggregateTernaryOperations(m1UC, m2UC, m3UC, ret, op, inCP);
		if(ret2.getNumRows() == 0 || ret2.getNumColumns() == 0)
			throw new DMLCompressionException("Invalid output");
		return ret2;
	}

	private void warnDecompression() {

		if(!warned) {

			boolean m1C = m1 instanceof CompressedMatrixBlock;
			boolean m2C = m2 instanceof CompressedMatrixBlock;
			boolean m3C = m3 instanceof CompressedMatrixBlock;
			StringBuilder sb = new StringBuilder(120);

			sb.append("aggregateTernaryOperations ");
			sb.append(op.aggOp.getClass().getSimpleName());
			sb.append(" ");
			sb.append(op.indexFn.getClass().getSimpleName());
			sb.append(" ");
			sb.append(op.aggOp.increOp.fn.getClass().getSimpleName());
			sb.append(" ");
			sb.append(op.binaryFn.getClass().getSimpleName());
			sb.append(" m1,m2,m3 ");
			sb.append(m1C);
			sb.append(" ");
			sb.append(m2C);
			sb.append(" ");
			sb.append(m3C);

			LOG.warn(sb.toString());
			warned = true;
		}
	}

}

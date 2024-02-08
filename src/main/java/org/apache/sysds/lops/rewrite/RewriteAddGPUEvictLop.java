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

package org.apache.sysds.lops.rewrite;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.lops.BinaryScalar;
import org.apache.sysds.lops.Data;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.OperatorOrderingUtils;
import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.lops.UnaryCP;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.VariableSet;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;

import java.util.ArrayList;
import java.util.List;

public class RewriteAddGPUEvictLop extends LopRewriteRule
{
	@Override
	public List<StatementBlock> rewriteLOPinStatementBlock(StatementBlock sb) {
		// TODO: Move this as a Statement block rewrite
		if (!ConfigurationManager.isAutoEvictionEnabled())
			return List.of(sb);

		if (sb == null || !(sb instanceof ForStatementBlock)
			|| !DMLScript.USE_ACCELERATOR || LineageCacheConfig.ReuseCacheType.isNone())
			return List.of(sb);

		// Collect the LOPs
		StatementBlock csb = ((ForStatement) sb.getStatement(0)).getBody().get(0);
		ArrayList<Lop> lops = OperatorOrderingUtils.getLopList(csb);

		// Check if this loop is for mini-batch processing
		boolean isMiniBatch = findMiniBatchSlicing(lops);

		// Insert statement block with _evict instruction before the loop
		ArrayList<StatementBlock> ret = new ArrayList<>();
		if (isMiniBatch) {
			int evictFrac = 100;
			StatementBlock sb0 = new StatementBlock();
			sb0.setDMLProg(sb.getDMLProg());
			sb0.setParseInfo(sb);
			sb0.setLiveIn(new VariableSet());
			sb0.setLiveOut(new VariableSet());
			// Create both lops and hops (hops for recompilation)
			// TODO: Add another input for the backend (GPU/CPU/Spark)
			ArrayList<Lop> newlops = new ArrayList<>();
			ArrayList<Hop> newhops = new ArrayList<>();
			Lop fr = Data.createLiteralLop(Types.ValueType.INT64, Integer.toString(evictFrac));
			fr.getOutputParameters().setDimensions(0, 0, 0, -1);
			UnaryCP evict = new UnaryCP(fr, Types.OpOp1._EVICT, fr.getDataType(), fr.getValueType(), Types.ExecType.CP);
			Hop in = new LiteralOp(evictFrac);
			Hop evictHop = new UnaryOp("tmp", Types.DataType.SCALAR, Types.ValueType.INT64, Types.OpOp1._EVICT, in);
			newlops.add(evict);
			newhops.add(evictHop);
			sb0.setLops(newlops);
			sb0.setHops(newhops);
			ret.add(sb0);
		}
		ret.add(sb);

		return ret;
	}

	@Override
	public List<StatementBlock> rewriteLOPinStatementBlocks(List<StatementBlock> sbs) {
		return sbs;
	}

	// To verify mini-batch processing, match the below pattern
	// beg = ((i-1) * batch_size) %% N + 1;
	// end = min(N, beg+batch_size-1);
	// X_batch = X[beg:end];
	private boolean findMiniBatchSlicing(ArrayList<Lop> lops) {
		for (Lop l : lops) {
			if (l instanceof RightIndex) {
				ArrayList<Lop> inputs = l.getInputs();
				if (inputs.get(0) instanceof Data && ((Data) inputs.get(0)).isTransientRead()
					&& inputs.get(0).getInputs().size() == 0		//input1 is the dataset
					&& inputs.get(1) instanceof BinaryScalar		//input2 is beg
					&& ((BinaryScalar) inputs.get(1)).getOperationType() == Types.OpOp2.PLUS
					&& inputs.get(2) instanceof BinaryScalar		//input3 is end
					&& ((BinaryScalar) inputs.get(2)).getOperationType() == Types.OpOp2.MIN)
					return true;
			}
		}
		return false;
	}
}

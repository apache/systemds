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

package org.apache.sysds.runtime.einsum;

import org.apache.commons.logging.Log;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.functionobjects.DiagIndex;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceDiag;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

import java.util.List;

public class EOpNodeUnary extends EOpNode {
	private final EUnaryOperand eUnaryOperand;
	public EOpNode child;

	public enum EUnaryOperand {
		DIAG, TRACE, SUM, SUM_COLS, SUM_ROWS
	}
	public EOpNodeUnary(Character c1, Character c2, Integer dim1, Integer dim2, EOpNode child, EUnaryOperand eUnaryOperand) {
		super(c1, c2, dim1, dim2);
		this.child = child;
		this.eUnaryOperand = eUnaryOperand;
	}

	@Override
	public List<EOpNode> getChildren() {
		return List.of(child);
	}
	@Override
	public String toString() {
		return this.getClass().getSimpleName()+" ("+eUnaryOperand.toString()+") "+this.getOutputString();
	}

	@Override
	public MatrixBlock computeEOpNode(List<MatrixBlock> inputs, int numOfThreads, Log LOG) {
		MatrixBlock mb = child.computeEOpNode(inputs, numOfThreads, LOG);
		return switch(eUnaryOperand) {
			case DIAG->{
				ReorgOperator op = new ReorgOperator(DiagIndex.getDiagIndexFnObject());
				yield mb.reorgOperations(op, new MatrixBlock(),0,0,0);
			}
			case TRACE -> {
				AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), Types.CorrectionLocationType.LASTCOLUMN);
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceDiag.getReduceDiagFnObject(), numOfThreads);
				MatrixBlock res = new MatrixBlock(10, 10, false);
				mb.aggregateUnaryOperations(aggun, res,0,null);
				yield res;
			}
			case SUM->{
				AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numOfThreads);
				MatrixBlock res = new MatrixBlock(1, 1, false);
				mb.aggregateUnaryOperations(aggun, res, 0, null);
				yield res;
			}
			case SUM_COLS ->{
				AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numOfThreads);
				MatrixBlock res = new MatrixBlock(mb.getNumColumns(), 1, false);
				mb.aggregateUnaryOperations(aggun, res, 0, null);
				yield res;
			}
			case SUM_ROWS ->{
				AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numOfThreads);
				MatrixBlock res = new MatrixBlock(mb.getNumRows(), 1, false);
				mb.aggregateUnaryOperations(aggun, res, 0, null);
				yield res;
			}
		};
	}

	@Override
	public EOpNode reorderChildrenAndOptimize(EOpNode parent, Character outChar1, Character outChar2) {
		return this;
	}
}

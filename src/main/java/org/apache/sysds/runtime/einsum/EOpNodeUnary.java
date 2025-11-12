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
import org.apache.sysds.runtime.functionobjects.DiagIndex;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

import java.util.ArrayList;

public class EOpNodeUnary extends EOpNode {
	private final EUnaryOperand eUnaryOperand;
	public EOpNode child;

	public enum EUnaryOperand {
		DIAG, SUM, SUM_ROWS, SUM_COLS
	}
	public EOpNodeUnary(Character c1, Character c2, EOpNode child, EUnaryOperand eUnaryOperand) {
		super(c1, c2);
		this.child = child;
		this.eUnaryOperand = eUnaryOperand;
    }

	@Override
	public String[] recursivePrintString() {
		String[] childResult = child.recursivePrintString();
		String[] res = new String[1+childResult.length];
		res[0] = this.getClass().getSimpleName()+" ("+eUnaryOperand.toString()+") "+this.toString();
		for (int i=0; i<childResult.length; i++) {
			res[i+1] = "   " +childResult[i];
		}
		return res;
	}

    @Override
    public MatrixBlock computeEOpNode(ArrayList<MatrixBlock> inputs, int numOfThreads, Log LOG) {
		MatrixBlock mb = child.computeEOpNode(inputs, numOfThreads, LOG);
		return switch(eUnaryOperand) {
			case DIAG->{
				ReorgOperator op = new ReorgOperator(DiagIndex.getDiagIndexFnObject());
				yield mb.reorgOperations(op, new MatrixBlock(),0,0,0);
			}
			case SUM->{
				AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), numOfThreads);
				MatrixBlock res = new MatrixBlock(1, 1, false);
				mb.aggregateUnaryOperations(aggun, res, 0, null);
				yield res;
			}
			case SUM_ROWS->{
				AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numOfThreads);
				MatrixBlock res = new MatrixBlock(mb.getNumColumns(), 1, false);
				mb.aggregateUnaryOperations(aggun, res, 0, null);
				yield res;
			}
			case SUM_COLS->{
				AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), numOfThreads);
				MatrixBlock res = new MatrixBlock(mb.getNumRows(), 1, false);
				mb.aggregateUnaryOperations(aggun, res, 0, null);
				yield res;
			}
		};
	}

    @Override
    public void reorderChildren(Character outChar1, Character outChar2) {

    }
}

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

package org.apache.sysds.hops.cost;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DnnOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.NaryOp;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;

/**
 * Class with methods estimating compute costs of operations.
 */
public class ComputeCost {
	private static final Log LOG = LogFactory.getLog(ComputeCost.class.getName());

	/**
	 * Get compute cost for given HOP based on the number of floating point operations per output cell
	 * and the total number of output cells.
	 * @param currentHop for which compute cost is returned
	 * @return compute cost of currentHop as number of floating point operations
	 */
	public static double getHOPComputeCost(Hop currentHop){
		double costs = 1;
		if( currentHop instanceof UnaryOp) {
			switch( ((UnaryOp)currentHop).getOp() ) {
				case ABS:
				case ROUND:
				case CEIL:
				case FLOOR:
				case SIGN:    costs = 1; break;
				case SPROP:
				case SQRT:    costs = 2; break;
				case EXP:     costs = 18; break;
				case SIGMOID: costs = 21; break;
				case LOG:
				case LOG_NZ:  costs = 32; break;
				case NCOL:
				case NROW:
				case PRINT:
				case ASSERT:
				case CAST_AS_BOOLEAN:
				case CAST_AS_DOUBLE:
				case CAST_AS_INT:
				case CAST_AS_MATRIX:
				case CAST_AS_SCALAR: costs = 1; break;
				case SIN:     costs = 18; break;
				case COS:     costs = 22; break;
				case TAN:     costs = 42; break;
				case ASIN:    costs = 93; break;
				case ACOS:    costs = 103; break;
				case ATAN:    costs = 40; break;
				case SINH:    costs = 93; break; // TODO:
				case COSH:    costs = 103; break;
				case TANH:    costs = 40; break;
				case CUMSUM:
				case CUMMIN:
				case CUMMAX:
				case CUMPROD: costs = 1; break;
				case CUMSUMPROD: costs = 2; break;
				default:
					LOG.warn("Cost model not "
						+ "implemented yet for: "+((UnaryOp)currentHop).getOp());
			}
		}
		else if( currentHop instanceof BinaryOp) {
			switch( ((BinaryOp)currentHop).getOp() ) {
				case MULT:
				case PLUS:
				case MINUS:
				case MIN:
				case MAX:
				case AND:
				case OR:
				case EQUAL:
				case NOTEQUAL:
				case LESS:
				case LESSEQUAL:
				case GREATER:
				case GREATEREQUAL:
				case CBIND:
				case RBIND:   costs = 1; break;
				case INTDIV:  costs = 6; break;
				case MODULUS: costs = 8; break;
				case DIV:     costs = 22; break;
				case LOG:
				case LOG_NZ:  costs = 32; break;
				case POW:     costs = (HopRewriteUtils.isLiteralOfValue(
					currentHop.getInput().get(1), 2) ? 1 : 16); break;
				case MINUS_NZ:
				case MINUS1_MULT: costs = 2; break;
				case MOMENT:
					int type = (int) (currentHop.getInput().get(1) instanceof LiteralOp ?
						HopRewriteUtils.getIntValueSafe((LiteralOp)currentHop.getInput().get(1)) : 2);
					switch( type ) {
						case 0: costs = 1; break; //count
						case 1: costs = 8; break; //mean
						case 2: costs = 16; break; //cm2
						case 3: costs = 31; break; //cm3
						case 4: costs = 51; break; //cm4
						case 5: costs = 16; break; //variance
					}
					break;
				case COV: costs = 23; break;
				default:
					LOG.warn("Cost model not "
						+ "implemented yet for: "+((BinaryOp)currentHop).getOp());
			}
		}
		else if( currentHop instanceof TernaryOp) {
			switch( ((TernaryOp)currentHop).getOp() ) {
				case IFELSE:
				case PLUS_MULT:
				case MINUS_MULT: costs = 2; break;
				case CTABLE:     costs = 3; break;
				case MOMENT:
					int type = (int) (currentHop.getInput().get(1) instanceof LiteralOp ?
						HopRewriteUtils.getIntValueSafe((LiteralOp)currentHop.getInput().get(1)) : 2);
					switch( type ) {
						case 0: costs = 2; break; //count
						case 1: costs = 9; break; //mean
						case 2: costs = 17; break; //cm2
						case 3: costs = 32; break; //cm3
						case 4: costs = 52; break; //cm4
						case 5: costs = 17; break; //variance
					}
					break;
				case COV: costs = 23; break;
				default:
					LOG.warn("Cost model not "
						+ "implemented yet for: "+((TernaryOp)currentHop).getOp());
			}
		}
		else if( currentHop instanceof NaryOp) {
			costs = HopRewriteUtils.isNary(currentHop, Types.OpOpN.MIN, Types.OpOpN.MAX, Types.OpOpN.PLUS) ?
				currentHop.getInput().size() : 1;
		}
		else if( currentHop instanceof ParameterizedBuiltinOp) {
			costs = 1;
		}
		else if( currentHop instanceof IndexingOp) {
			costs = 1;
		}
		else if( currentHop instanceof ReorgOp) {
			costs = 1;
		}
		else if( currentHop instanceof DnnOp) {
			switch( ((DnnOp)currentHop).getOp() ) {
				case BIASADD:
				case BIASMULT:
					costs = 2;
				default:
					LOG.warn("Cost model not "
						+ "implemented yet for: "+((DnnOp)currentHop).getOp());
			}
		}
		else if( currentHop instanceof AggBinaryOp) {
			//outer product template w/ matrix-matrix
			//or row template w/ matrix-vector or matrix-matrix
			costs = 2 * currentHop.getInput().get(0).getDim2();
			if( currentHop.getInput().get(0).dimsKnown(true) )
				costs *= currentHop.getInput().get(0).getSparsity();
		}
		else if( currentHop instanceof AggUnaryOp) {
			switch(((AggUnaryOp)currentHop).getOp()) {
				case SUM:    costs = 4; break;
				case SUM_SQ: costs = 5; break;
				case MIN:
				case MAX:    costs = 1; break;
				default:
					LOG.warn("Cost model not "
						+ "implemented yet for: "+((AggUnaryOp)currentHop).getOp());
			}
			switch(((AggUnaryOp)currentHop).getDirection()) {
				case Col: costs *= Math.max(currentHop.getInput().get(0).getDim1(),1); break;
				case Row: costs *= Math.max(currentHop.getInput().get(0).getDim2(),1); break;
				case RowCol: costs *= getSize(currentHop.getInput().get(0)); break;
			}
		}

		//scale by current output size in order to correctly reflect
		//a mix of row and cell operations in the same fused operator
		//(e.g., row template with fused column vector operations)
		costs *= getSize(currentHop);
		return costs;
	}

	/**
	 * Get number of output cells of given hop.
	 * @param hop for which the number of output cells are found
	 * @return number of output cells of given hop
	 */
	private static long getSize(Hop hop) {
		return Math.max(hop.getDim1(),1)
			* Math.max(hop.getDim2(),1);
	}
}

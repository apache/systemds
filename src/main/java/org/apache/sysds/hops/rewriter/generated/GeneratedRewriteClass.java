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

package org.apache.sysds.hops.rewriter.generated;

import java.util.ArrayList;
import java.util.function.Function;

import org.apache.sysds.utils.Statistics;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewriter.dml.DMLExecutor;
import org.apache.sysds.hops.rewriter.RewriterRuntimeUtils;

public class GeneratedRewriteClass implements Function {

	@Override
	public Object apply( Object _hi ) {
		if ( _hi == null )
			return null;

		Hop hi = (Hop) _hi;

		if ( hi.getDataType() == Types.DataType.SCALAR ) {
			hi = _applyRewrite0(hi); // *(0.0,a) => 0.0
			hi = _applyRewrite1(hi); // *(a,0.0) => 0.0
			hi = _applyRewrite23(hi); // sum(/(tmp83271,tmp60732)) => /(sum(tmp83271),tmp60732)
			hi = _applyRewrite27(hi); // sum(*(*(tmp8790,tmp30390),tmp97178)) => *(tmp30390,sum(*(tmp97178,tmp8790)))
		} else if ( hi.getDataType() == Types.DataType.MATRIX ) {
			if ( hi instanceof BinaryOp ) {
				if ( (( BinaryOp ) hi ).getOp() == Types.OpOp2.PLUS ) {
					if ( hi.getInput().size() == 2 ) {
						Hop hi_0 = hi.getInput(0);
						Hop hi_1 = hi.getInput(1);
						if ( hi_0.getDataType() == Types.DataType.MATRIX ) {
							if ( hi_0 instanceof BinaryOp ) {
								if ( (( BinaryOp ) hi_0 ).getOp() == Types.OpOp2.MINUS ) {
									if ( hi_0.getInput().size() == 2 ) {
										Hop hi_0_0 = hi_0.getInput(0);
										Hop hi_0_1 = hi_0.getInput(1);
										hi = _applyRewrite2(hi); // +(A,0.0) => A
										hi = _applyRewrite7(hi); // +(-(0.0,A),B) => -(B,A)
										hi = _applyRewrite10(hi); // +(-(A,a),b) => +(A,-(b,a))
										hi = _applyRewrite12(hi); // +(-(a,A),b) => -(+(a,b),A)
										hi = _applyRewrite20(hi); // +(-(tmp80035,f12880),tmp63699) => -(+(tmp63699,tmp80035),f12880)
										hi = _applyRewrite31(hi); // +(-(a,tmp98488),tmp82242) => +(-(tmp82242,tmp98488),a)
										hi = _applyRewrite37(hi); // +(-(*(C,b),d),A) => -(+*(A,b,C),d)
										hi = _applyRewrite38(hi); // +(-(*(D,c),B),A) => -(A,-*(B,c,D))
										hi = _applyRewrite39(hi); // +(M9347,*(K,*(M40316,f32765))) => +*(M9347,f32765,*(K,M40316))
										hi = _applyRewrite41(hi); // +(-(f45081,A),B) => +(f45081,-(B,A))
										hi = _applyRewrite42(hi); // +(M9347,*(*(f32765,K),M40316)) => +*(M9347,f32765,*(K,M40316))
										hi = _applyRewrite46(hi); // +(-(b,%*%(C,D)),A) => +(b,-(A,%*%(C,D)))
										hi = _applyRewrite54(hi); // +(-(C,d),%*%(A,B)) => -(+(C,%*%(A,B)),d)
									} else {
										hi = _applyRewrite2(hi); // +(A,0.0) => A
										hi = _applyRewrite39(hi); // +(M9347,*(K,*(M40316,f32765))) => +*(M9347,f32765,*(K,M40316))
										hi = _applyRewrite42(hi); // +(M9347,*(*(f32765,K),M40316)) => +*(M9347,f32765,*(K,M40316))
									}
								} else if ( (( BinaryOp ) hi_0 ).getOp() == Types.OpOp2.MULT ) {
									if ( hi_0.getInput().size() == 2 ) {
										Hop hi_0_0 = hi_0.getInput(0);
										Hop hi_0_1 = hi_0.getInput(1);
										hi = _applyRewrite2(hi); // +(A,0.0) => A
										hi = _applyRewrite18(hi); // +(*(*(y_corr,-(float599,is_zero_y_corr)),tmp8608),*(tmp20367,+(tmp23071,tmp55180))) => +(*(*(tmp8608,y_corr),-(float599,is_zero_y_corr)),*(tmp20367,+(tmp55180,tmp23071)))
										hi = _applyRewrite32(hi); // +(*(tmp99142,missing_mask_Y),*(tmp58606,missing_mask_Y)) => *(missing_mask_Y,+(tmp99142,tmp58606))
										hi = _applyRewrite39(hi); // +(M9347,*(K,*(M40316,f32765))) => +*(M9347,f32765,*(K,M40316))
										hi = _applyRewrite42(hi); // +(M9347,*(*(f32765,K),M40316)) => +*(M9347,f32765,*(K,M40316))
										hi = _applyRewrite43(hi); // +(*(*(K,f32765),M40316),M9347) => +*(M9347,f32765,*(K,M40316))
									} else {
										hi = _applyRewrite2(hi); // +(A,0.0) => A
										hi = _applyRewrite39(hi); // +(M9347,*(K,*(M40316,f32765))) => +*(M9347,f32765,*(K,M40316))
										hi = _applyRewrite42(hi); // +(M9347,*(*(f32765,K),M40316)) => +*(M9347,f32765,*(K,M40316))
									}
								} else {
									hi = _applyRewrite2(hi); // +(A,0.0) => A
									hi = _applyRewrite39(hi); // +(M9347,*(K,*(M40316,f32765))) => +*(M9347,f32765,*(K,M40316))
									hi = _applyRewrite42(hi); // +(M9347,*(*(f32765,K),M40316)) => +*(M9347,f32765,*(K,M40316))
								}
							} else {
								hi = _applyRewrite2(hi); // +(A,0.0) => A
								hi = _applyRewrite39(hi); // +(M9347,*(K,*(M40316,f32765))) => +*(M9347,f32765,*(K,M40316))
								hi = _applyRewrite42(hi); // +(M9347,*(*(f32765,K),M40316)) => +*(M9347,f32765,*(K,M40316))
							}
						} else if ( hi_0.getDataType() == Types.DataType.SCALAR ) {
							hi = _applyRewrite3(hi); // +(0.0,A) => A
							hi = _applyRewrite11(hi); // +(a,-(A,b)) => +(A,-(a,b))
							hi = _applyRewrite13(hi); // +(a,-(b,A)) => -(+(a,b),A)
						}
					}
				} else if ( (( BinaryOp ) hi ).getOp() == Types.OpOp2.MINUS ) {
					if ( hi.getInput().size() == 2 ) {
						Hop hi_0 = hi.getInput(0);
						Hop hi_1 = hi.getInput(1);
						if ( hi_0.getDataType() == Types.DataType.MATRIX ) {
							if ( hi_0 instanceof BinaryOp ) {
								if ( (( BinaryOp ) hi_0 ).getOp() == Types.OpOp2.MINUS ) {
									if ( hi_0.getInput().size() == 2 ) {
										Hop hi_0_0 = hi_0.getInput(0);
										Hop hi_0_1 = hi_0.getInput(1);
										hi = _applyRewrite4(hi); // -(A,0.0) => A
										hi = _applyRewrite14(hi); // -(-(A,a),b) => -(A,+(b,a))
										hi = _applyRewrite16(hi); // -(-(a,A),b) => -(-(a,b),A)
										hi = _applyRewrite29(hi); // -(obj,+(tmp6500,tmp26035)) => -(-(obj,tmp6500),tmp26035)
										hi = _applyRewrite30(hi); // -(-(tmp68530,tmp73960),tmp29113) => -(tmp68530,+(tmp73960,tmp29113))
										hi = _applyRewrite40(hi); // -(y,+(%*%(X,B),intercept)) => -(-(y,%*%(X,B)),intercept)
										hi = _applyRewrite47(hi); // -(-(f43240,A),f67634) => -(-(f43240,f67634),A)
										hi = _applyRewrite51(hi); // -(M22650,*(*(f97734,M67673),M97683)) => -*(M22650,f97734,*(M97683,M67673))
										hi = _applyRewrite52(hi); // -(-(f75306,M67233),*(A,M350)) => -(f75306,+(*(A,M350),M67233))
										hi = _applyRewrite53(hi); // -(-(f75306,*(A,M350)),M67233) => -(f75306,+(*(A,M350),M67233))
									} else {
										hi = _applyRewrite4(hi); // -(A,0.0) => A
										hi = _applyRewrite29(hi); // -(obj,+(tmp6500,tmp26035)) => -(-(obj,tmp6500),tmp26035)
										hi = _applyRewrite40(hi); // -(y,+(%*%(X,B),intercept)) => -(-(y,%*%(X,B)),intercept)
										hi = _applyRewrite51(hi); // -(M22650,*(*(f97734,M67673),M97683)) => -*(M22650,f97734,*(M97683,M67673))
									}
								} else if ( (( BinaryOp ) hi_0 ).getOp() == Types.OpOp2.PLUS ) {
									hi = _applyRewrite28(hi); // -(+(a,tmp82242),tmp98488) => +(-(tmp82242,tmp98488),a)
									hi = _applyRewrite4(hi); // -(A,0.0) => A
									hi = _applyRewrite29(hi); // -(obj,+(tmp6500,tmp26035)) => -(-(obj,tmp6500),tmp26035)
									hi = _applyRewrite40(hi); // -(y,+(%*%(X,B),intercept)) => -(-(y,%*%(X,B)),intercept)
									hi = _applyRewrite51(hi); // -(M22650,*(*(f97734,M67673),M97683)) => -*(M22650,f97734,*(M97683,M67673))
								} else {
									hi = _applyRewrite4(hi); // -(A,0.0) => A
									hi = _applyRewrite29(hi); // -(obj,+(tmp6500,tmp26035)) => -(-(obj,tmp6500),tmp26035)
									hi = _applyRewrite40(hi); // -(y,+(%*%(X,B),intercept)) => -(-(y,%*%(X,B)),intercept)
									hi = _applyRewrite51(hi); // -(M22650,*(*(f97734,M67673),M97683)) => -*(M22650,f97734,*(M97683,M67673))
								}
							} else {
								hi = _applyRewrite4(hi); // -(A,0.0) => A
								hi = _applyRewrite29(hi); // -(obj,+(tmp6500,tmp26035)) => -(-(obj,tmp6500),tmp26035)
								hi = _applyRewrite40(hi); // -(y,+(%*%(X,B),intercept)) => -(-(y,%*%(X,B)),intercept)
								hi = _applyRewrite51(hi); // -(M22650,*(*(f97734,M67673),M97683)) => -*(M22650,f97734,*(M97683,M67673))
							}
						} else if ( hi_0.getDataType() == Types.DataType.SCALAR ) {
							hi = _applyRewrite8(hi); // -(0.0,-(B,A)) => -(A,B)
							hi = _applyRewrite15(hi); // -(a,-(A,b)) => -(+(a,b),A)
							hi = _applyRewrite17(hi); // -(a,-(b,A)) => +(-(a,b),A)
							hi = _applyRewrite21(hi); // -(tmp66496,cast.MATRIX(tmp91996)) => cast.MATRIX(-(tmp66496,tmp91996))
						}
					}
				} else if ( (( BinaryOp ) hi ).getOp() == Types.OpOp2.MULT ) {
					if ( hi.getInput().size() == 2 ) {
						Hop hi_0 = hi.getInput(0);
						Hop hi_1 = hi.getInput(1);
						if ( hi_0.getDataType() == Types.DataType.MATRIX ) {
							if ( hi_0 instanceof BinaryOp ) {
								if ( (( BinaryOp ) hi_0 ).getOp() == Types.OpOp2.DIV ) {
									if ( hi_0.getInput().size() == 2 ) {
										Hop hi_0_0 = hi_0.getInput(0);
										Hop hi_0_1 = hi_0.getInput(1);
										hi = _applyRewrite5(hi); // *(A,0.0) => const(A,0.0)
										hi = _applyRewrite9(hi); // *(A,/(1.0,B)) => /(A,B)
										hi = _applyRewrite19(hi); // *(/(1.0,tmp5995),tmp41945) => /(tmp41945,tmp5995)
										hi = _applyRewrite34(hi); // *(/(1.0,B),a) => /(a,B)
										hi = _applyRewrite44(hi); // *(/(1.0,M13119),A) => /(A,M13119)
										hi = _applyRewrite49(hi); // *(A,/(1.0,M13119)) => /(A,M13119)
									} else {
										hi = _applyRewrite5(hi); // *(A,0.0) => const(A,0.0)
										hi = _applyRewrite9(hi); // *(A,/(1.0,B)) => /(A,B)
										hi = _applyRewrite49(hi); // *(A,/(1.0,M13119)) => /(A,M13119)
									}
								} else if ( (( BinaryOp ) hi_0 ).getOp() == Types.OpOp2.MULT ) {
									hi = _applyRewrite25(hi); // *(*(y_corr,-(float599,is_zero_y_corr)),tmp8608) => *(*(y_corr,tmp8608),-(float599,is_zero_y_corr))
									hi = _applyRewrite5(hi); // *(A,0.0) => const(A,0.0)
									hi = _applyRewrite9(hi); // *(A,/(1.0,B)) => /(A,B)
									hi = _applyRewrite49(hi); // *(A,/(1.0,M13119)) => /(A,M13119)
								} else {
									hi = _applyRewrite5(hi); // *(A,0.0) => const(A,0.0)
									hi = _applyRewrite9(hi); // *(A,/(1.0,B)) => /(A,B)
									hi = _applyRewrite49(hi); // *(A,/(1.0,M13119)) => /(A,M13119)
								}
							} else if ( hi_0 instanceof AggBinaryOp ) {
								hi = _applyRewrite26(hi); // *(%*%(scale_lambda,parsertemp150455),tmp43267) => {%*%(*(tmp43267,scale_lambda),parsertemp150455)}
								hi = _applyRewrite5(hi); // *(A,0.0) => const(A,0.0)
								hi = _applyRewrite9(hi); // *(A,/(1.0,B)) => /(A,B)
								hi = _applyRewrite49(hi); // *(A,/(1.0,M13119)) => /(A,M13119)
							} else {
								hi = _applyRewrite5(hi); // *(A,0.0) => const(A,0.0)
								hi = _applyRewrite9(hi); // *(A,/(1.0,B)) => /(A,B)
								hi = _applyRewrite49(hi); // *(A,/(1.0,M13119)) => /(A,M13119)
							}
						} else if ( hi_0.getDataType() == Types.DataType.SCALAR ) {
							hi = _applyRewrite6(hi); // *(0.0,A) => const(A,0.0)
							hi = _applyRewrite33(hi); // *(tmp43267,%*%(scale_lambda,parsertemp150455)) => {%*%(*(tmp43267,scale_lambda),parsertemp150455)}
							hi = _applyRewrite36(hi); // *(a,cast.MATRIX(b)) => cast.MATRIX(*(a,b))
							hi = _applyRewrite50(hi); // *(f68833,-(0.0,M48693)) => *(M48693,-(0.0,f68833))
						}
					}
				} else if ( (( BinaryOp ) hi ).getOp() == Types.OpOp2.DIV ) {
					hi = _applyRewrite35(hi); // /(a,cast.MATRIX(b)) => cast.MATRIX(/(a,b))
					hi = _applyRewrite45(hi); // /(M43656,2.0) => *(0.5,M43656)
					hi = _applyRewrite48(hi); // /(M62235,2000.0) => *(5.0E-4,M62235)
				}
			} else if ( hi instanceof ReorgOp ) {
				hi = _applyRewrite22(hi); // t(==(key_unique,t(key))) => ==(key,t(key_unique))
			} else if ( hi instanceof AggBinaryOp ) {
				hi = _applyRewrite24(hi); // %*%(t(X_batch),tmp92007) => {t(%*%(t(tmp92007),X_batch))}
			}
		}
		return hi;
	}

	// Implementation of the rule *(0.0,a) => 0.0
	private static Hop _applyRewrite0(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: 0.0

		Hop newRoot = hi_0;
		if ( hi_0.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: *(0.0,a) => 0.0");
		return newRoot;
	}

	// Implementation of the rule *(a,0.0) => 0.0
	private static Hop _applyRewrite1(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new HOP-DAG: 0.0

		Hop newRoot = hi_1;
		if ( hi_1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: *(a,0.0) => 0.0");
		return newRoot;
	}

	// Implementation of the rule +(A,0.0) => A
	private static Hop _applyRewrite2(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new HOP-DAG: A

		Hop newRoot = hi_0;
		if ( hi_0.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: +(A,0.0) => A");
		return newRoot;
	}

	// Implementation of the rule +(0.0,A) => A
	private static Hop _applyRewrite3(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: A

		Hop newRoot = hi_1;
		if ( hi_1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: +(0.0,A) => A");
		return newRoot;
	}

	// Implementation of the rule -(A,0.0) => A
	private static Hop _applyRewrite4(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new HOP-DAG: A

		Hop newRoot = hi_0;
		if ( hi_0.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: -(A,0.0) => A");
		return newRoot;
	}

	// Implementation of the rule *(A,0.0) => const(A,0.0)
	private static Hop _applyRewrite5(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new HOP-DAG: const(A,0.0)
		DataGenOp v1 = ((DataGenOp) HopRewriteUtils.createDataGenOpFromDims(HopRewriteUtils.createUnary(hi_0, Types.OpOp1.NROW),HopRewriteUtils.createUnary(hi_0, Types.OpOp1.NCOL),0.0D));

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);

		DMLExecutor.println("Applying rewrite: *(A,0.0) => const(A,0.0)");
		return newRoot;
	}

	// Implementation of the rule *(0.0,A) => const(A,0.0)
	private static Hop _applyRewrite6(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: const(A,0.0)
		DataGenOp v1 = ((DataGenOp) HopRewriteUtils.createDataGenOpFromDims(HopRewriteUtils.createUnary(hi_1, Types.OpOp1.NROW),HopRewriteUtils.createUnary(hi_1, Types.OpOp1.NCOL),0.0D));

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);

		DMLExecutor.println("Applying rewrite: *(0.0,A) => const(A,0.0)");
		return newRoot;
	}

	// Implementation of the rule +(-(0.0,A),B) => -(B,A)
	private static Hop _applyRewrite7(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(B,A)
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1, hi_0_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		DMLExecutor.println("Applying rewrite: +(-(0.0,A),B) => -(B,A)");
		return newRoot;
	}

	// Implementation of the rule -(0.0,-(B,A)) => -(A,B)
	private static Hop _applyRewrite8(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(A,B)
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1_1, hi_1_0) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1_1, hi_1_0, Types.OpOp2.MINUS);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: -(0.0,-(B,A)) => -(A,B)");
		return newRoot;
	}

	// Implementation of the rule *(A,/(1.0,B)) => /(A,B)
	private static Hop _applyRewrite9(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR|| !l_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: /(A,B)
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0, hi_1_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, hi_1_1, Types.OpOp2.DIV);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		DMLExecutor.println("Applying rewrite: *(A,/(1.0,B)) => /(A,B)");
		return newRoot;
	}

	// Implementation of the rule +(-(A,a),b) => +(A,-(b,a))
	private static Hop _applyRewrite10(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: +(A,-(b,a))
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, v1, Types.OpOp2.PLUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: +(-(A,a),b) => +(A,-(b,a))");
		return newRoot;
	}

	// Implementation of the rule +(a,-(A,b)) => +(A,-(a,b))
	private static Hop _applyRewrite11(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: +(A,-(a,b))
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_1_0, v1, Types.OpOp2.PLUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: +(a,-(A,b)) => +(A,-(a,b))");
		return newRoot;
	}

	// Implementation of the rule +(-(a,A),b) => -(+(a,b),A)
	private static Hop _applyRewrite12(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(+(a,b),A)
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_0_1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: +(-(a,A),b) => -(+(a,b),A)");
		return newRoot;
	}

	// Implementation of the rule +(a,-(b,A)) => -(+(a,b),A)
	private static Hop _applyRewrite13(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(+(a,b),A)
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_1_1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: +(a,-(b,A)) => -(+(a,b),A)");
		return newRoot;
	}

	// Implementation of the rule -(-(A,a),b) => -(A,+(b,a))
	private static Hop _applyRewrite14(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(A,+(b,a))
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, hi_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, v1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: -(-(A,a),b) => -(A,+(b,a))");
		return newRoot;
	}

	// Implementation of the rule -(a,-(A,b)) => -(+(a,b),A)
	private static Hop _applyRewrite15(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(+(a,b),A)
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_1_0, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: -(a,-(A,b)) => -(+(a,b),A)");
		return newRoot;
	}

	// Implementation of the rule -(-(a,A),b) => -(-(a,b),A)
	private static Hop _applyRewrite16(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(-(a,b),A)
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_0_1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: -(-(a,A),b) => -(-(a,b),A)");
		return newRoot;
	}

	// Implementation of the rule -(a,-(b,A)) => +(-(a,b),A)
	private static Hop _applyRewrite17(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: +(-(a,b),A)
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_1_1, Types.OpOp2.PLUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: -(a,-(b,A)) => +(-(a,b),A)");
		return newRoot;
	}

	// Implementation of the rule +(*(*(y_corr,-(float599,is_zero_y_corr)),tmp8608),*(tmp20367,+(tmp23071,tmp55180))) => +(*(*(tmp8608,y_corr),-(float599,is_zero_y_corr)),*(tmp20367,+(tmp55180,tmp23071)))
	private static Hop _applyRewrite18(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if (hi_0_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0_1 = (BinaryOp) hi_0_0_1;

		if ( c_hi_0_0_1.getOp() != Types.OpOp2.MINUS || !c_hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1_0 = hi_0_0_1.getInput(0);

		if ( hi_0_0_1_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1_1 = hi_0_0_1.getInput(1);

		if ( hi_0_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: +(*(*(tmp8608,y_corr),-(float599,is_zero_y_corr)),*(tmp20367,+(tmp55180,tmp23071)))
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0_1, hi_0_0_0) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_1, hi_0_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0_1_0, hi_0_0_1_1, Types.OpOp2.MINUS);
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(v1, v2) )
			return hi;
		BinaryOp v3 = HopRewriteUtils.createAutoGeneratedBinary(v1, v2, Types.OpOp2.MULT);
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1_1_1, hi_1_1_0) )
			return hi;
		BinaryOp v4 = HopRewriteUtils.createAutoGeneratedBinary(hi_1_1_1, hi_1_1_0, Types.OpOp2.PLUS);
		BinaryOp v5 = HopRewriteUtils.createAutoGeneratedBinary(hi_1_0, v4, Types.OpOp2.MULT);
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(v3, v5) )
			return hi;
		BinaryOp v6 = HopRewriteUtils.createAutoGeneratedBinary(v3, v5, Types.OpOp2.PLUS);

		Hop newRoot = v6;
		if ( v6.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		DMLExecutor.println("Applying rewrite: +(*(*(y_corr,-(float599,is_zero_y_corr)),tmp8608),*(tmp20367,+(tmp23071,tmp55180))) => +(*(*(tmp8608,y_corr),-(float599,is_zero_y_corr)),*(tmp20367,+(tmp55180,tmp23071)))");
		return newRoot;
	}

	// Implementation of the rule *(/(1.0,tmp5995),tmp41945) => /(tmp41945,tmp5995)
	private static Hop _applyRewrite19(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: /(tmp41945,tmp5995)
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, hi_0_1, Types.OpOp2.DIV);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		DMLExecutor.println("Applying rewrite: *(/(1.0,tmp5995),tmp41945) => /(tmp41945,tmp5995)");
		return newRoot;
	}

	// Implementation of the rule +(-(tmp80035,f12880),tmp63699) => -(+(tmp63699,tmp80035),f12880)
	private static Hop _applyRewrite20(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(+(tmp63699,tmp80035),f12880)
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1, hi_0_0) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, hi_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_0_1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: +(-(tmp80035,f12880),tmp63699) => -(+(tmp63699,tmp80035),f12880)");
		return newRoot;
	}

	// Implementation of the rule -(tmp66496,cast.MATRIX(tmp91996)) => cast.MATRIX(-(tmp66496,tmp91996))
	private static Hop _applyRewrite21(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof UnaryOp) )
			return hi;

		UnaryOp c_hi_1 = (UnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp1.CAST_AS_MATRIX || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: cast.MATRIX(-(tmp66496,tmp91996))
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, hi_1_0, Types.OpOp2.MINUS);
		UnaryOp v2 = HopRewriteUtils.createUnary(v1, Types.OpOp1.CAST_AS_MATRIX);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: -(tmp66496,cast.MATRIX(tmp91996)) => cast.MATRIX(-(tmp66496,tmp91996))");
		return newRoot;
	}

	// Implementation of the rule t(==(key_unique,t(key))) => ==(key,t(key_unique))
	private static Hop _applyRewrite22(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.EQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: ==(key,t(key_unique))
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0_1_0, v1) )
			return hi;
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_1_0, v1, Types.OpOp2.EQUAL);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		DMLExecutor.println("Applying rewrite: t(==(key_unique,t(key))) => ==(key,t(key_unique))");
		return newRoot;
	}

	// Implementation of the rule sum(/(tmp83271,tmp60732)) => /(sum(tmp83271),tmp60732)
	private static Hop _applyRewrite23(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.RowCol) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: /(sum(tmp83271),tmp60732)
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0, Types.AggOp.SUM, Types.Direction.RowCol);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_0_1, Types.OpOp2.DIV);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: sum(/(tmp83271,tmp60732)) => /(sum(tmp83271),tmp60732)");
		return newRoot;
	}

	// Implementation of the rule %*%(t(X_batch),tmp92007) => {t(%*%(t(tmp92007),X_batch))}
	private static Hop _applyRewrite24(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		if ( hi_1.getDim2() == -1 || hi_1.getNnz() == -1 || hi_0_0.getNnz() == -1 || hi_0_0.getDim2() == -1 || hi_1.getDim1() == -1 )
			return hi;


		double[] costs = new double[2];
		costs[0] = (hi_0_0.getNnz() + (Math.min(hi_0_0.getNnz(), hi_1.getNnz()) * hi_1.getDim1() * 3.0) + 20020.0);
		costs[1] = (hi_1.getNnz() + (Math.min(hi_1.getNnz(), hi_0_0.getNnz()) * hi_1.getDim1() * 3.0) + (Math.min((hi_1.getNnz() * (1.0 / hi_1.getDim2())), 1.0) * Math.min((hi_0_0.getNnz() * (1.0 / hi_0_0.getDim2())), 1.0) * hi_1.getDim2() * hi_0_0.getDim2()) + 30030.0);
		int minIdx = minIdx(costs);

		switch( minIdx ) {
			case 1: {
				// Now, we start building the new HOP-DAG: t(%*%(t(tmp92007),X_batch))
				ReorgOp v1 = HopRewriteUtils.createTranspose(hi_1);
				AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_0);
				ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

				Hop newRoot = v3;
				if ( v3.getValueType() != hi.getValueType() ) {
					newRoot = castIfNecessary(newRoot, hi);
					if ( newRoot == null )
						return hi;
				}

				ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

				for ( Hop p : parents )
					HopRewriteUtils.replaceChildReference(p, hi, newRoot);

				// Remove old unreferenced Hops
				HopRewriteUtils.cleanupUnreferenced(hi);
				HopRewriteUtils.cleanupUnreferenced(hi_0);

				DMLExecutor.println("Applying rewrite: %*%(t(X_batch),tmp92007) => {t(%*%(t(tmp92007),X_batch))}");
				return newRoot;
			}
		}
		return hi;
	}

	// Implementation of the rule *(*(y_corr,-(float599,is_zero_y_corr)),tmp8608) => *(*(y_corr,tmp8608),-(float599,is_zero_y_corr))
	private static Hop _applyRewrite25(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: *(*(y_corr,tmp8608),-(float599,is_zero_y_corr))
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0_0, hi_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_1_0, hi_0_1_1, Types.OpOp2.MINUS);
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(v1, v2) )
			return hi;
		BinaryOp v3 = HopRewriteUtils.createAutoGeneratedBinary(v1, v2, Types.OpOp2.MULT);

		Hop newRoot = v3;
		if ( v3.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		DMLExecutor.println("Applying rewrite: *(*(y_corr,-(float599,is_zero_y_corr)),tmp8608) => *(*(y_corr,tmp8608),-(float599,is_zero_y_corr))");
		return newRoot;
	}

	// Implementation of the rule *(%*%(scale_lambda,parsertemp150455),tmp43267) => {%*%(*(tmp43267,scale_lambda),parsertemp150455)}
	private static Hop _applyRewrite26(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		if ( hi_0_0.getDim1() == -1 || hi_0_1.getDim2() == -1 || hi_0_1.getNnz() == -1 || hi_0_0.getNnz() == -1 || hi_0_1.getDim1() == -1 )
			return hi;


		double[] costs = new double[2];
		costs[0] = ((Math.min(hi_0_0.getNnz(), hi_0_1.getNnz()) * hi_0_1.getDim1() * 3.0) + (2.0 * (Math.min((hi_0_0.getNnz() * (1.0 / hi_0_0.getDim1())), 1.0) * Math.min((hi_0_1.getNnz() * (1.0 / hi_0_1.getDim2())), 1.0) * hi_0_0.getDim1() * hi_0_1.getDim2())) + 20020.0);
		costs[1] = ((2.0 * hi_0_0.getNnz()) + (Math.min(hi_0_0.getNnz(), hi_0_1.getNnz()) * hi_0_1.getDim1() * 3.0) + 20020.0);
		int minIdx = minIdx(costs);

		switch( minIdx ) {
			case 1: {
				// Now, we start building the new HOP-DAG: %*%(*(tmp43267,scale_lambda),parsertemp150455)
				BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, hi_0_0, Types.OpOp2.MULT);
				AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_1);

				Hop newRoot = v2;
				if ( v2.getValueType() != hi.getValueType() ) {
					newRoot = castIfNecessary(newRoot, hi);
					if ( newRoot == null )
						return hi;
				}

				ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

				for ( Hop p : parents )
					HopRewriteUtils.replaceChildReference(p, hi, newRoot);

				// Remove old unreferenced Hops
				HopRewriteUtils.cleanupUnreferenced(hi);
				HopRewriteUtils.cleanupUnreferenced(hi_0);

				DMLExecutor.println("Applying rewrite: *(%*%(scale_lambda,parsertemp150455),tmp43267) => {%*%(*(tmp43267,scale_lambda),parsertemp150455)}");
				return newRoot;
			}
		}
		return hi;
	}

	// Implementation of the rule sum(*(*(tmp8790,tmp30390),tmp97178)) => *(tmp30390,sum(*(tmp97178,tmp8790)))
	private static Hop _applyRewrite27(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.RowCol) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: *(tmp30390,sum(*(tmp97178,tmp8790)))
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0_1, hi_0_0_0) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_1, hi_0_0_0, Types.OpOp2.MULT);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.RowCol);
		BinaryOp v3 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0_1, v2, Types.OpOp2.MULT);

		Hop newRoot = v3;
		if ( v3.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		DMLExecutor.println("Applying rewrite: sum(*(*(tmp8790,tmp30390),tmp97178)) => *(tmp30390,sum(*(tmp97178,tmp8790)))");
		return newRoot;
	}

	// Implementation of the rule -(+(a,tmp82242),tmp98488) => +(-(tmp82242,tmp98488),a)
	private static Hop _applyRewrite28(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: +(-(tmp82242,tmp98488),a)
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0_1, hi_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_0_0, Types.OpOp2.PLUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: -(+(a,tmp82242),tmp98488) => +(-(tmp82242,tmp98488),a)");
		return newRoot;
	}

	// Implementation of the rule -(obj,+(tmp6500,tmp26035)) => -(-(obj,tmp6500),tmp26035)
	private static Hop _applyRewrite29(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(-(obj,tmp6500),tmp26035)
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0, hi_1_0) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_1_1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: -(obj,+(tmp6500,tmp26035)) => -(-(obj,tmp6500),tmp26035)");
		return newRoot;
	}

	// Implementation of the rule -(-(tmp68530,tmp73960),tmp29113) => -(tmp68530,+(tmp73960,tmp29113))
	private static Hop _applyRewrite30(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(tmp68530,+(tmp73960,tmp29113))
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0_1, hi_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, v1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: -(-(tmp68530,tmp73960),tmp29113) => -(tmp68530,+(tmp73960,tmp29113))");
		return newRoot;
	}

	// Implementation of the rule +(-(a,tmp98488),tmp82242) => +(-(tmp82242,tmp98488),a)
	private static Hop _applyRewrite31(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: +(-(tmp82242,tmp98488),a)
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1, hi_0_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_0_0, Types.OpOp2.PLUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: +(-(a,tmp98488),tmp82242) => +(-(tmp82242,tmp98488),a)");
		return newRoot;
	}

	// Implementation of the rule +(*(tmp99142,missing_mask_Y),*(tmp58606,missing_mask_Y)) => *(missing_mask_Y,+(tmp99142,tmp58606))
	private static Hop _applyRewrite32(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_0_1 != hi_1_1 )
			return hi;


		// Now, we start building the new HOP-DAG: *(missing_mask_Y,+(tmp99142,tmp58606))
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_1, v1, Types.OpOp2.MULT);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: +(*(tmp99142,missing_mask_Y),*(tmp58606,missing_mask_Y)) => *(missing_mask_Y,+(tmp99142,tmp58606))");
		return newRoot;
	}

	// Implementation of the rule *(tmp43267,%*%(scale_lambda,parsertemp150455)) => {%*%(*(tmp43267,scale_lambda),parsertemp150455)}
	private static Hop _applyRewrite33(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		if ( hi_1_0.getNnz() == -1 || hi_1_1.getDim2() == -1 || hi_1_0.getDim1() == -1 || hi_1_0.getDim2() == -1 || hi_1_1.getNnz() == -1 )
			return hi;


		double[] costs = new double[2];
		costs[0] = ((Math.min(hi_1_0.getNnz(), hi_1_1.getNnz()) * hi_1_0.getDim2() * 3.0) + (2.0 * (Math.min((hi_1_0.getNnz() * (1.0 / hi_1_0.getDim1())), 1.0) * Math.min((hi_1_1.getNnz() * (1.0 / hi_1_1.getDim2())), 1.0) * hi_1_0.getDim1() * hi_1_1.getDim2())) + 20020.0);
		costs[1] = ((2.0 * hi_1_0.getNnz()) + (Math.min(hi_1_0.getNnz(), hi_1_1.getNnz()) * hi_1_0.getDim2() * 3.0) + 20020.0);
		int minIdx = minIdx(costs);

		switch( minIdx ) {
			case 1: {
				// Now, we start building the new HOP-DAG: %*%(*(tmp43267,scale_lambda),parsertemp150455)
				BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, hi_1_0, Types.OpOp2.MULT);
				AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_1_1);

				Hop newRoot = v2;
				if ( v2.getValueType() != hi.getValueType() ) {
					newRoot = castIfNecessary(newRoot, hi);
					if ( newRoot == null )
						return hi;
				}

				ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

				for ( Hop p : parents )
					HopRewriteUtils.replaceChildReference(p, hi, newRoot);

				// Remove old unreferenced Hops
				HopRewriteUtils.cleanupUnreferenced(hi);
				HopRewriteUtils.cleanupUnreferenced(hi_1);

				DMLExecutor.println("Applying rewrite: *(tmp43267,%*%(scale_lambda,parsertemp150455)) => {%*%(*(tmp43267,scale_lambda),parsertemp150455)}");
				return newRoot;
			}
		}
		return hi;
	}

	// Implementation of the rule *(/(1.0,B),a) => /(a,B)
	private static Hop _applyRewrite34(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: /(a,B)
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, hi_0_1, Types.OpOp2.DIV);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		DMLExecutor.println("Applying rewrite: *(/(1.0,B),a) => /(a,B)");
		return newRoot;
	}

	// Implementation of the rule /(a,cast.MATRIX(b)) => cast.MATRIX(/(a,b))
	private static Hop _applyRewrite35(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof UnaryOp) )
			return hi;

		UnaryOp c_hi_1 = (UnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp1.CAST_AS_MATRIX || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: cast.MATRIX(/(a,b))
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, hi_1_0, Types.OpOp2.DIV);
		UnaryOp v2 = HopRewriteUtils.createUnary(v1, Types.OpOp1.CAST_AS_MATRIX);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: /(a,cast.MATRIX(b)) => cast.MATRIX(/(a,b))");
		return newRoot;
	}

	// Implementation of the rule *(a,cast.MATRIX(b)) => cast.MATRIX(*(a,b))
	private static Hop _applyRewrite36(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof UnaryOp) )
			return hi;

		UnaryOp c_hi_1 = (UnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp1.CAST_AS_MATRIX || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: cast.MATRIX(*(a,b))
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, hi_1_0, Types.OpOp2.MULT);
		UnaryOp v2 = HopRewriteUtils.createUnary(v1, Types.OpOp1.CAST_AS_MATRIX);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: *(a,cast.MATRIX(b)) => cast.MATRIX(*(a,b))");
		return newRoot;
	}

	// Implementation of the rule +(-(*(C,b),d),A) => -(+*(A,b,C),d)
	private static Hop _applyRewrite37(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(+*(A,b,C),d)
		if ( !RewriterRuntimeUtils.hasMatchingDims(hi_1, hi_0_0_0) )
			return hi;
		TernaryOp v1 = HopRewriteUtils.createTernary(hi_1, hi_0_0_1, hi_0_0_0,Types.OpOp3.PLUS_MULT);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_0_1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		DMLExecutor.println("Applying rewrite: +(-(*(C,b),d),A) => -(+*(A,b,C),d)");
		return newRoot;
	}

	// Implementation of the rule +(-(*(D,c),B),A) => -(A,-*(B,c,D))
	private static Hop _applyRewrite38(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(A,-*(B,c,D))
		if ( !RewriterRuntimeUtils.hasMatchingDims(hi_0_1, hi_0_0_0) )
			return hi;
		TernaryOp v1 = HopRewriteUtils.createTernary(hi_0_1, hi_0_0_1, hi_0_0_0,Types.OpOp3.MINUS_MULT);
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1, v1) )
			return hi;
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, v1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		DMLExecutor.println("Applying rewrite: +(-(*(D,c),B),A) => -(A,-*(B,c,D))");
		return newRoot;
	}

	// Implementation of the rule +(M9347,*(K,*(M40316,f32765))) => +*(M9347,f32765,*(K,M40316))
	private static Hop _applyRewrite39(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MULT || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: +*(M9347,f32765,*(K,M40316))
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1_0, hi_1_1_0) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1_0, hi_1_1_0, Types.OpOp2.MULT);
		if ( !RewriterRuntimeUtils.hasMatchingDims(hi_0, v1) )
			return hi;
		TernaryOp v2 = HopRewriteUtils.createTernary(hi_0, hi_1_1_1, v1,Types.OpOp3.PLUS_MULT);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		DMLExecutor.println("Applying rewrite: +(M9347,*(K,*(M40316,f32765))) => +*(M9347,f32765,*(K,M40316))");
		return newRoot;
	}

	// Implementation of the rule -(y,+(%*%(X,B),intercept)) => -(-(y,%*%(X,B)),intercept)
	private static Hop _applyRewrite40(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1_0) )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(-(y,%*%(X,B)),intercept)
		AggBinaryOp v1 = HopRewriteUtils.createMatrixMultiply(hi_1_0_0, hi_1_0_1);
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0, v1) )
			return hi;
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createAutoGeneratedBinary(v2, hi_1_1, Types.OpOp2.MINUS);

		Hop newRoot = v3;
		if ( v3.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		DMLExecutor.println("Applying rewrite: -(y,+(%*%(X,B),intercept)) => -(-(y,%*%(X,B)),intercept)");
		return newRoot;
	}

	// Implementation of the rule +(-(f45081,A),B) => +(f45081,-(B,A))
	private static Hop _applyRewrite41(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: +(f45081,-(B,A))
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1, hi_0_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, v1, Types.OpOp2.PLUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: +(-(f45081,A),B) => +(f45081,-(B,A))");
		return newRoot;
	}

	// Implementation of the rule +(M9347,*(*(f32765,K),M40316)) => +*(M9347,f32765,*(K,M40316))
	private static Hop _applyRewrite42(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: +*(M9347,f32765,*(K,M40316))
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1_0_1, hi_1_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1_0_1, hi_1_1, Types.OpOp2.MULT);
		if ( !RewriterRuntimeUtils.hasMatchingDims(hi_0, v1) )
			return hi;
		TernaryOp v2 = HopRewriteUtils.createTernary(hi_0, hi_1_0_0, v1,Types.OpOp3.PLUS_MULT);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		DMLExecutor.println("Applying rewrite: +(M9347,*(*(f32765,K),M40316)) => +*(M9347,f32765,*(K,M40316))");
		return newRoot;
	}

	// Implementation of the rule +(*(*(K,f32765),M40316),M9347) => +*(M9347,f32765,*(K,M40316))
	private static Hop _applyRewrite43(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: +*(M9347,f32765,*(K,M40316))
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0_0_0, hi_0_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MULT);
		if ( !RewriterRuntimeUtils.hasMatchingDims(hi_1, v1) )
			return hi;
		TernaryOp v2 = HopRewriteUtils.createTernary(hi_1, hi_0_0_1, v1,Types.OpOp3.PLUS_MULT);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		DMLExecutor.println("Applying rewrite: +(*(*(K,f32765),M40316),M9347) => +*(M9347,f32765,*(K,M40316))");
		return newRoot;
	}

	// Implementation of the rule *(/(1.0,M13119),A) => /(A,M13119)
	private static Hop _applyRewrite44(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: /(A,M13119)
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1, hi_0_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, hi_0_1, Types.OpOp2.DIV);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		DMLExecutor.println("Applying rewrite: *(/(1.0,M13119),A) => /(A,M13119)");
		return newRoot;
	}

	// Implementation of the rule /(M43656,2.0) => *(0.5,M43656)
	private static Hop _applyRewrite45(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1.getDoubleValue() != 2.0 )
			return hi;


		// Now, we start building the new HOP-DAG: *(0.5,M43656)
		LiteralOp l1 = new LiteralOp( 0.5 );
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(l1, hi_0, Types.OpOp2.MULT);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: /(M43656,2.0) => *(0.5,M43656)");
		return newRoot;
	}

	// Implementation of the rule +(-(b,%*%(C,D)),A) => +(b,-(A,%*%(C,D)))
	private static Hop _applyRewrite46(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0_1) )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: +(b,-(A,%*%(C,D)))
		AggBinaryOp v1 = HopRewriteUtils.createMatrixMultiply(hi_0_1_0, hi_0_1_1);
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1, v1) )
			return hi;
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		Hop newRoot = v3;
		if ( v3.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		DMLExecutor.println("Applying rewrite: +(-(b,%*%(C,D)),A) => +(b,-(A,%*%(C,D)))");
		return newRoot;
	}

	// Implementation of the rule -(-(f43240,A),f67634) => -(-(f43240,f67634),A)
	private static Hop _applyRewrite47(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(-(f43240,f67634),A)
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_0_1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		DMLExecutor.println("Applying rewrite: -(-(f43240,A),f67634) => -(-(f43240,f67634),A)");
		return newRoot;
	}

	// Implementation of the rule /(M62235,2000.0) => *(5.0E-4,M62235)
	private static Hop _applyRewrite48(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1.getDoubleValue() != 2000.0 )
			return hi;


		// Now, we start building the new HOP-DAG: *(5.0E-4,M62235)
		LiteralOp l1 = new LiteralOp( 5.0E-4 );
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(l1, hi_0, Types.OpOp2.MULT);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: /(M62235,2000.0) => *(5.0E-4,M62235)");
		return newRoot;
	}

	// Implementation of the rule *(A,/(1.0,M13119)) => /(A,M13119)
	private static Hop _applyRewrite49(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR|| !l_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: /(A,M13119)
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0, hi_1_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0, hi_1_1, Types.OpOp2.DIV);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		DMLExecutor.println("Applying rewrite: *(A,/(1.0,M13119)) => /(A,M13119)");
		return newRoot;
	}

	// Implementation of the rule *(f68833,-(0.0,M48693)) => *(M48693,-(0.0,f68833))
	private static Hop _applyRewrite50(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR|| !l_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: *(M48693,-(0.0,f68833))
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1_0, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_1_1, v1, Types.OpOp2.MULT);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: *(f68833,-(0.0,M48693)) => *(M48693,-(0.0,f68833))");
		return newRoot;
	}

	// Implementation of the rule -(M22650,*(*(f97734,M67673),M97683)) => -*(M22650,f97734,*(M97683,M67673))
	private static Hop _applyRewrite51(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -*(M22650,f97734,*(M97683,M67673))
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1_1, hi_1_0_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1_1, hi_1_0_1, Types.OpOp2.MULT);
		if ( !RewriterRuntimeUtils.hasMatchingDims(hi_0, v1) )
			return hi;
		TernaryOp v2 = HopRewriteUtils.createTernary(hi_0, hi_1_0_0, v1,Types.OpOp3.MINUS_MULT);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		DMLExecutor.println("Applying rewrite: -(M22650,*(*(f97734,M67673),M97683)) => -*(M22650,f97734,*(M97683,M67673))");
		return newRoot;
	}

	// Implementation of the rule -(-(f75306,M67233),*(A,M350)) => -(f75306,+(*(A,M350),M67233))
	private static Hop _applyRewrite52(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(f75306,+(*(A,M350),M67233))
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_1_0, hi_1_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_1_0, hi_1_1, Types.OpOp2.MULT);
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(v1, hi_0_1) )
			return hi;
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_0_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		Hop newRoot = v3;
		if ( v3.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: -(-(f75306,M67233),*(A,M350)) => -(f75306,+(*(A,M350),M67233))");
		return newRoot;
	}

	// Implementation of the rule -(-(f75306,*(A,M350)),M67233) => -(f75306,+(*(A,M350),M67233))
	private static Hop _applyRewrite53(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MULT || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(f75306,+(*(A,M350),M67233))
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0_1_0, hi_0_1_1) )
			return hi;
		BinaryOp v1 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_1_0, hi_0_1_1, Types.OpOp2.MULT);
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(v1, hi_1) )
			return hi;
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(v1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		Hop newRoot = v3;
		if ( v3.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		DMLExecutor.println("Applying rewrite: -(-(f75306,*(A,M350)),M67233) => -(f75306,+(*(A,M350),M67233))");
		return newRoot;
	}

	// Implementation of the rule +(-(C,d),%*%(A,B)) => -(+(C,%*%(A,B)),d)
	private static Hop _applyRewrite54(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new HOP-DAG: -(+(C,%*%(A,B)),d)
		AggBinaryOp v1 = HopRewriteUtils.createMatrixMultiply(hi_1_0, hi_1_1);
		if ( !RewriterRuntimeUtils.validateBinaryBroadcasting(hi_0_0, v1) )
			return hi;
		BinaryOp v2 = HopRewriteUtils.createAutoGeneratedBinary(hi_0_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createAutoGeneratedBinary(v2, hi_0_1, Types.OpOp2.MINUS);

		Hop newRoot = v3;
		if ( v3.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		DMLExecutor.println("Applying rewrite: +(-(C,d),%*%(A,B)) => -(+(C,%*%(A,B)),d)");
		return newRoot;
	}

	private static Hop castIfNecessary(Hop newRoot, Hop oldRoot) {
		Types.OpOp1 cast = null;
		switch ( oldRoot.getValueType().toExternalString() ) {
			case "DOUBLE":
				cast = Types.OpOp1.CAST_AS_DOUBLE;
				break;
			case "INT":
				cast = Types.OpOp1.CAST_AS_INT;
				break;
			case "BOOLEAN":
				cast = Types.OpOp1.CAST_AS_BOOLEAN;
				break;
			default:
				return null;
		}

		return new UnaryOp("tmp", oldRoot.getDataType(), oldRoot.getValueType(), cast, newRoot);
	}
	private static int minIdx(double[] l) {
		double minValue = Double.MAX_VALUE;
		int minIdx = -1;

		for (int i = 0; i < l.length; i++) {
			if (l[i] < minValue) {
				minValue = l[i];
				minIdx = i;
			}
		}

		return minIdx;
	}
}
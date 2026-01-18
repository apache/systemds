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

package org.apache.sysds.runtime.functionobjects;

import java.util.HashMap;

import org.apache.commons.math3.util.FastMath;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.DMLScriptException;


/**
 *  Class with pre-defined set of objects. This class can not be instantiated elsewhere.
 *  
 *  Notes on commons.math FastMath:
 *  * FastMath uses lookup tables and interpolation instead of native calls.
 *  * The memory overhead for those tables is roughly 48KB in total (acceptable)
 *  * Micro and application benchmarks showed significantly (30%-3x) performance improvements
 *    for most operations; without loss of accuracy.
 *  * atan / sqrt were 20% slower in FastMath and hence, we use Math there
 *  * round / abs were equivalent in FastMath and hence, we use Math there
 *  * Finally, there is just one argument against FastMath - The comparison heavily depends
 *    on the JVM. For example, currently the IBM JDK JIT compiles to HW instructions for sqrt
 *    which makes this operation very efficient; as soon as other operations like log/exp are
 *    similarly compiled, we should rerun the micro benchmarks, and switch back if necessary.
 *  
 */
public class Builtin extends ValueFunction 
{
	private static final long serialVersionUID = 3836744687789840574L;
	
	public enum BuiltinCode { AUTODIFF, SIN, COS, TAN, SINH, COSH, TANH, ASIN, ACOS, ATAN, LOG, LOG_NZ, MIN,
		MAX, ABS, SIGN, SQRT, EXP, PLOGP, PRINT, PRINTF, NROW, NCOL, LENGTH, LINEAGE, ROUND, MAXINDEX, MININDEX,
		STOP, CEIL, FLOOR, CUMSUM, ROWCUMSUM, CUMPROD, CUMMIN, CUMMAX, CUMSUMPROD, INVERSE, SPROP, SIGMOID, EVAL, LIST,
		TYPEOF, APPLY_SCHEMA, DETECTSCHEMA, ISNA, ISNAN, ISINF, DROP_INVALID_TYPE, 
		DROP_INVALID_LENGTH, VALUE_SWAP, FRAME_ROW_REPLICATE,
		MAP, COUNT_DISTINCT, COUNT_DISTINCT_APPROX, UNIQUE}


	public BuiltinCode bFunc;
	
	private static final boolean FASTMATH = true;
	
	static public HashMap<String, BuiltinCode> String2BuiltinCode;
	static {
		String2BuiltinCode = new HashMap<>();
		String2BuiltinCode.put( "autoDiff"    , BuiltinCode.AUTODIFF);
		String2BuiltinCode.put( "sin"    , BuiltinCode.SIN);
		String2BuiltinCode.put( "cos"    , BuiltinCode.COS);
		String2BuiltinCode.put( "tan"    , BuiltinCode.TAN);
		String2BuiltinCode.put( "sinh"    , BuiltinCode.SINH);
		String2BuiltinCode.put( "cosh"    , BuiltinCode.COSH);
		String2BuiltinCode.put( "tanh"    , BuiltinCode.TANH);
		String2BuiltinCode.put( "asin"   , BuiltinCode.ASIN);
		String2BuiltinCode.put( "acos"   , BuiltinCode.ACOS);
		String2BuiltinCode.put( "atan"   , BuiltinCode.ATAN);
		String2BuiltinCode.put( "log"    , BuiltinCode.LOG);
		String2BuiltinCode.put( "log_nz" , BuiltinCode.LOG_NZ);
		String2BuiltinCode.put( "min"    , BuiltinCode.MIN);
		String2BuiltinCode.put( "max"    , BuiltinCode.MAX);
		String2BuiltinCode.put( "maxindex", BuiltinCode.MAXINDEX);
		String2BuiltinCode.put( "minindex", BuiltinCode.MININDEX);
		String2BuiltinCode.put( "abs"    , BuiltinCode.ABS);
		String2BuiltinCode.put( "sign"   , BuiltinCode.SIGN);
		String2BuiltinCode.put( "sqrt"   , BuiltinCode.SQRT);
		String2BuiltinCode.put( "exp"    , BuiltinCode.EXP);
		String2BuiltinCode.put( "plogp"  , BuiltinCode.PLOGP);
		String2BuiltinCode.put( "print"  , BuiltinCode.PRINT);
		String2BuiltinCode.put( "printf"  , BuiltinCode.PRINTF);
		String2BuiltinCode.put( "eval"  , BuiltinCode.EVAL);
		String2BuiltinCode.put( "list"  , BuiltinCode.LIST);
		String2BuiltinCode.put( "nrow"   , BuiltinCode.NROW);
		String2BuiltinCode.put( "ncol"   , BuiltinCode.NCOL);
		String2BuiltinCode.put( "length" , BuiltinCode.LENGTH);
		String2BuiltinCode.put( "round"  , BuiltinCode.ROUND);
		String2BuiltinCode.put( "stop"   , BuiltinCode.STOP);
		String2BuiltinCode.put( "ceil"   , BuiltinCode.CEIL);
		String2BuiltinCode.put( "floor"  , BuiltinCode.FLOOR);
		String2BuiltinCode.put( "ucumk+" , BuiltinCode.CUMSUM);
		String2BuiltinCode.put( "urowcumk+" , BuiltinCode.ROWCUMSUM);
		String2BuiltinCode.put("rowCumsum", BuiltinCode.ROWCUMSUM);
		String2BuiltinCode.put( "ucum*"  , BuiltinCode.CUMPROD);
		String2BuiltinCode.put( "ucumk+*", BuiltinCode.CUMSUMPROD);
		String2BuiltinCode.put( "ucummin", BuiltinCode.CUMMIN);
		String2BuiltinCode.put( "ucummax", BuiltinCode.CUMMAX);
		String2BuiltinCode.put( "inverse", BuiltinCode.INVERSE);
		String2BuiltinCode.put( "sprop",   BuiltinCode.SPROP);
		String2BuiltinCode.put( "sigmoid", BuiltinCode.SIGMOID);
		String2BuiltinCode.put( "typeOf", BuiltinCode.TYPEOF);
		String2BuiltinCode.put( "detectSchema", BuiltinCode.DETECTSCHEMA);
		String2BuiltinCode.put( "isna", BuiltinCode.ISNA);
		String2BuiltinCode.put( "isnan", BuiltinCode.ISNAN);
		String2BuiltinCode.put( "isinf", BuiltinCode.ISINF);
		String2BuiltinCode.put( "dropInvalidType", BuiltinCode.DROP_INVALID_TYPE);
		String2BuiltinCode.put( "freplicate", BuiltinCode.FRAME_ROW_REPLICATE);
		String2BuiltinCode.put( "dropInvalidLength", BuiltinCode.DROP_INVALID_LENGTH);
		String2BuiltinCode.put( "_map", BuiltinCode.MAP);
		String2BuiltinCode.put( "valueSwap", BuiltinCode.VALUE_SWAP);
		String2BuiltinCode.put( "applySchema", BuiltinCode.APPLY_SCHEMA);
	}
	
	protected Builtin(BuiltinCode bf) {
		bFunc = bf;
	}
	
	public BuiltinCode getBuiltinCode() {
		return bFunc;
	}
	
	public static boolean isBuiltinCode(ValueFunction fn, BuiltinCode... codes) {
		for( BuiltinCode code : codes )
			if (fn instanceof Builtin && ((Builtin)fn).getBuiltinCode() == code)
				return true;
		return false;
	}

	public static boolean isBuiltinFnObject(String str) {
		return String2BuiltinCode.containsKey(str);
	}
	
	public static Builtin getBuiltinFnObject(String str) {
		BuiltinCode code = String2BuiltinCode.get(str);
		return getBuiltinFnObject( code );
	}

	public static Builtin getBuiltinFnObject(BuiltinCode code) {
		if ( code == null ) 
			return null; 
		if(code == BuiltinCode.MAX)
			return Max.getMaxFnObject();
		else if (code == BuiltinCode.MIN)
			return Min.getMinFnObject();
		return new Builtin(code);
	}

	@Override
	public double execute (double in) {
		switch(bFunc) {
			case SIN:    return FASTMATH ? FastMath.sin(in) : Math.sin(in);
			case COS:    return FASTMATH ? FastMath.cos(in) : Math.cos(in);
			case TAN:    return FASTMATH ? FastMath.tan(in) : Math.tan(in);
			case ASIN:   return FASTMATH ? FastMath.asin(in) : Math.asin(in);
			case ACOS:   return FASTMATH ? FastMath.acos(in) : Math.acos(in);
			case ATAN:   return Math.atan(in); //faster in Math
			// FastMath.*h is faster 98% of time than Math.*h in initial micro-benchmarks
			case SINH:   return FASTMATH ? FastMath.sinh(in) : Math.sinh(in);
			case COSH:   return FASTMATH ? FastMath.cosh(in) : Math.cosh(in);
			case TANH:   return FASTMATH ? FastMath.tanh(in) : Math.tanh(in);
			case CEIL:   return FASTMATH ? FastMath.ceil(in) : Math.ceil(in);
			case FLOOR:  return FASTMATH ? FastMath.floor(in) : Math.floor(in);
			case LOG:    return Math.log(in); //faster in Math
			case LOG_NZ: return (in==0) ? 0 : Math.log(in); //faster in Math
			case ABS:    return Math.abs(in); //no need for FastMath
			case SIGN:   return FASTMATH ? FastMath.signum(in) : Math.signum(in);
			case SQRT:   return Math.sqrt(in); //faster in Math
			case EXP:    return FASTMATH ? FastMath.exp(in) : Math.exp(in);
			case ROUND: return Math.round(in); //no need for FastMath
			
			case PLOGP:
				if (in == 0.0)
					return 0.0;
				else if (in < 0)
					return Double.NaN;
				else //faster in Math
					return in * Math.log(in);
			
			case SPROP:
				//sample proportion: P*(1-P)
				return in * (1 - in); 
	
			case SIGMOID:
				//sigmoid: 1/(1+exp(-x))
				return FASTMATH ? 1 / (1 + FastMath.exp(-in))  : 1 / (1 + Math.exp(-in));
			
			case ISNA: return Double.isNaN(in) ? 1 : 0;
			case ISNAN: return Double.isNaN(in) ? 1 : 0;
			case ISINF: return Double.isInfinite(in) ? 1 : 0;
			
			default:
				throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}

	@Override
	public double execute (long in) {
		return execute((double)in);
	}

	/*
	 * Builtin functions with two inputs
	 */
	@Override
	public double execute (double in1, double in2) {
		switch(bFunc) {
			/*
			 * Arithmetic relational operators (==, !=, <=, >=) must be instead of
			 * <code>Double.compare()</code> due to the inconsistencies in the way
			 * NaN and -0.0 are handled. The behavior of methods in
			 * <code>Double</code> class are designed mainly to make Java
			 * collections work properly. For more details, see the help for
			 * <code>Double.equals()</code> and <code>Double.comapreTo()</code>.
			 */
			case MAX:
			case CUMMAX:
				return Math.max(in1, in2);
			case MIN:
			case CUMMIN:
				return Math.min(in1, in2);
				
				// *** HACK ALERT *** HACK ALERT *** HACK ALERT ***
				// rowIndexMax() and its siblings require comparing four values, but
				// the aggregation API only allows two values. So the execute()
				// method receives as its argument the two cell values to be
				// compared and performs just the value part of the comparison. We
				// return an integer cast down to a double, since the aggregation
				// API doesn't have any way to return anything but a double. The
				// integer returned takes on three possible values: //
				// .     0 => keep the index associated with in1 //
				// .     1 => use the index associated with in2 //
				// .     2 => use whichever index is higher (tie in value) //
			case MAXINDEX:
				if (in1 == in2) {
					return 2;
				} else if (in1 > in2) {
					return 1;
				} else { // in1 < in2
					return 0;
				}
			case MININDEX:
				if (in1 == in2) {
					return 2;
				} else if (in1 < in2) {
					return 1;
				} else { // in1 > in2
					return 0;
				}
				// *** END HACK ***
			case LOG://faster in Math
				return (Math.log(in1)/Math.log(in2)); 
			case LOG_NZ: //faster in Math
				// if in2 == 0 then Math.log is -infinity and division by -infinity returns -0.
				// therefore to allow sparse linear algebra, we replace it with standard 0.
				return (in1 == 0.0 || in2 == 0.0) ? 0.0 : (Math.log(in1) / Math.log(in2));
			default:
				throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}
	
	@Override
	public double execute (long in1, long in2) {
		switch(bFunc) {
			case MAX:
			case CUMMAX:   return Math.max(in1, in2);
			
			case MIN:
			case CUMMIN:   return Math.min(in1, in2);
			
			case MAXINDEX: return (in1 >= in2) ? 1 : 0;
			case MININDEX: return (in1 <= in2) ? 1 : 0;
			
			case LOG:
				//faster in Math
				return Math.log(in1)/Math.log(in2);
			case LOG_NZ:
				//faster in Math
				// if in2 == 0 then Math.log is -infinity and division by -infinity returns -0.
				// therefore to allow sparse linear algebra, we replace it with standard 0.
				return (in1 == 0L || in2 == 0L) ? 0L : Math.log(in1) / Math.log(in2);
			default:
				throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}

	@Override
	public String execute (String in1) {
		switch (bFunc) {
		case PRINT:
			if (!DMLScript.suppressPrint2Stdout())
				System.out.println(in1);
			return null;
		case PRINTF:
			if (!DMLScript.suppressPrint2Stdout())
				System.out.println(in1);
			return null;
		case STOP:
			throw new DMLScriptException(in1);
		default:
			throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}

	public boolean isBinarySparseSafe(){
		switch(bFunc){
			case LOG_NZ:
				return true;
			default:
				return false;
		}
	}

	@Override
	public String toString(){
		return "Builtin:" + bFunc;
	}
}

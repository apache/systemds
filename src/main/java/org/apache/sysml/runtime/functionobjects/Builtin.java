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

package org.apache.sysml.runtime.functionobjects;

import java.util.HashMap;

import org.apache.commons.math3.util.FastMath;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLScriptException;


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
	
	public enum BuiltinCode { SIN, COS, TAN, ASIN, ACOS, ATAN, LOG, LOG_NZ, MIN, MAX, ABS, SIGN, SQRT, EXP, PLOGP, PRINT, PRINTF, NROW, NCOL, LENGTH, ROUND, MAXINDEX, MININDEX, STOP, CEIL, FLOOR, CUMSUM, CUMPROD, CUMMIN, CUMMAX, INVERSE, SPROP, SIGMOID, SELP };
	public BuiltinCode bFunc;
	
	private static final boolean FASTMATH = true;
	
	static public HashMap<String, BuiltinCode> String2BuiltinCode;
	static {
		String2BuiltinCode = new HashMap<String, BuiltinCode>();
		
		String2BuiltinCode.put( "sin"    , BuiltinCode.SIN);
		String2BuiltinCode.put( "cos"    , BuiltinCode.COS);
		String2BuiltinCode.put( "tan"    , BuiltinCode.TAN);
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
		String2BuiltinCode.put( "nrow"   , BuiltinCode.NROW);
		String2BuiltinCode.put( "ncol"   , BuiltinCode.NCOL);
		String2BuiltinCode.put( "length" , BuiltinCode.LENGTH);
		String2BuiltinCode.put( "round"  , BuiltinCode.ROUND);
		String2BuiltinCode.put( "stop"   , BuiltinCode.STOP);
		String2BuiltinCode.put( "ceil"   , BuiltinCode.CEIL);
		String2BuiltinCode.put( "floor"  , BuiltinCode.FLOOR);
		String2BuiltinCode.put( "ucumk+" , BuiltinCode.CUMSUM);
		String2BuiltinCode.put( "ucum*"  , BuiltinCode.CUMPROD);
		String2BuiltinCode.put( "ucummin", BuiltinCode.CUMMIN);
		String2BuiltinCode.put( "ucummax", BuiltinCode.CUMMAX);
		String2BuiltinCode.put( "inverse", BuiltinCode.INVERSE);
		String2BuiltinCode.put( "sprop",   BuiltinCode.SPROP);
		String2BuiltinCode.put( "sigmoid", BuiltinCode.SIGMOID);
		String2BuiltinCode.put( "sel+",    BuiltinCode.SELP);
	}
	
	// We should create one object for every builtin function that we support
	private static Builtin sinObj = null, cosObj = null, tanObj = null, asinObj = null, acosObj = null, atanObj = null;
	private static Builtin logObj = null, lognzObj = null, minObj = null, maxObj = null, maxindexObj = null, minindexObj=null;
	private static Builtin absObj = null, signObj = null, sqrtObj = null, expObj = null, plogpObj = null, printObj = null, printfObj;
	private static Builtin nrowObj = null, ncolObj = null, lengthObj = null, roundObj = null, ceilObj=null, floorObj=null; 
	private static Builtin inverseObj=null, cumsumObj=null, cumprodObj=null, cumminObj=null, cummaxObj=null;
	private static Builtin stopObj = null, spropObj = null, sigmoidObj = null, selpObj = null;
	
	private Builtin(BuiltinCode bf) {
		bFunc = bf;
	}
	
	public BuiltinCode getBuiltinCode() {
		return bFunc;
	}

	public static Builtin getBuiltinFnObject (String str) 
	{
		BuiltinCode code = String2BuiltinCode.get(str);
		return getBuiltinFnObject( code );
	}

	public static Builtin getBuiltinFnObject(BuiltinCode code) 
	{	
		if ( code == null ) 
			return null; 
			
		switch ( code ) {
		case SIN:
			if ( sinObj == null )
				sinObj = new Builtin(BuiltinCode.SIN);
			return sinObj;
		
		case COS:
			if ( cosObj == null )
				cosObj = new Builtin(BuiltinCode.COS);
			return cosObj;
		case TAN:
			if ( tanObj == null )
				tanObj = new Builtin(BuiltinCode.TAN);
			return tanObj;
		case ASIN:
			if ( asinObj == null )
				asinObj = new Builtin(BuiltinCode.ASIN);
			return asinObj;
		
		case ACOS:
			if ( acosObj == null )
				acosObj = new Builtin(BuiltinCode.ACOS);
			return acosObj;
		case ATAN:
			if ( atanObj == null )
				atanObj = new Builtin(BuiltinCode.ATAN);
			return atanObj;
		case LOG:
			if ( logObj == null )
				logObj = new Builtin(BuiltinCode.LOG);
			return logObj;
		case LOG_NZ:
			if ( lognzObj == null )
				lognzObj = new Builtin(BuiltinCode.LOG_NZ);
			return lognzObj;
		case MAX:
			if ( maxObj == null )
				maxObj = new Builtin(BuiltinCode.MAX);
			return maxObj;
		case MAXINDEX:
			if ( maxindexObj == null )
				maxindexObj = new Builtin(BuiltinCode.MAXINDEX);
			return maxindexObj;
		case MIN:
			if ( minObj == null )
				minObj = new Builtin(BuiltinCode.MIN);
			return minObj;
		case MININDEX:
			if ( minindexObj == null )
				minindexObj = new Builtin(BuiltinCode.MININDEX);
			return minindexObj;
		case ABS:
			if ( absObj == null )
				absObj = new Builtin(BuiltinCode.ABS);
			return absObj;
		case SIGN:
			if ( signObj == null )
				signObj = new Builtin(BuiltinCode.SIGN);
			return signObj;
		case SQRT:
			if ( sqrtObj == null )
				sqrtObj = new Builtin(BuiltinCode.SQRT);
			return sqrtObj;
		case EXP:
			if ( expObj == null )
				expObj = new Builtin(BuiltinCode.EXP);
			return expObj;
		case PLOGP:
			if ( plogpObj == null )
				plogpObj = new Builtin(BuiltinCode.PLOGP);
			return plogpObj;
		case PRINT:
			if ( printObj == null )
				printObj = new Builtin(BuiltinCode.PRINT);
			return printObj;
		case PRINTF:
			if (printfObj == null) {
				printfObj = new Builtin(BuiltinCode.PRINTF);
			}
			return printfObj;
		case NROW:
			if ( nrowObj == null )
				nrowObj = new Builtin(BuiltinCode.NROW);
			return nrowObj;
		case NCOL:
			if ( ncolObj == null )
				ncolObj = new Builtin(BuiltinCode.NCOL);
			return ncolObj;
		case LENGTH:
			if ( lengthObj == null )
				lengthObj = new Builtin(BuiltinCode.LENGTH);
			return lengthObj;
		case ROUND:
			if ( roundObj == null )
				roundObj = new Builtin(BuiltinCode.ROUND);
			return roundObj;
		case CEIL:
			if ( ceilObj == null )
				ceilObj = new Builtin(BuiltinCode.CEIL);
			return ceilObj;
		case FLOOR:
			if ( floorObj == null )
				floorObj = new Builtin(BuiltinCode.FLOOR);
			return floorObj;
		case CUMSUM:
			if ( cumsumObj == null )
				cumsumObj = new Builtin(BuiltinCode.CUMSUM);
			return cumsumObj;	
		case CUMPROD:
			if ( cumprodObj == null )
				cumprodObj = new Builtin(BuiltinCode.CUMPROD);
			return cumprodObj;	
		case CUMMIN:
			if ( cumminObj == null )
				cumminObj = new Builtin(BuiltinCode.CUMMIN);
			return cumminObj;	
		case CUMMAX:
			if ( cummaxObj == null )
				cummaxObj = new Builtin(BuiltinCode.CUMMAX);
			return cummaxObj;	
		case INVERSE:
			if ( inverseObj == null )
				inverseObj = new Builtin(BuiltinCode.INVERSE);
			return inverseObj;	
		case STOP:
			if ( stopObj == null )
				stopObj = new Builtin(BuiltinCode.STOP);
			return stopObj;

		case SPROP:
			if ( spropObj == null )
				spropObj = new Builtin(BuiltinCode.SPROP);
			return spropObj;
			
		case SIGMOID:
			if ( sigmoidObj == null )
				sigmoidObj = new Builtin(BuiltinCode.SIGMOID);
			return sigmoidObj;
		
		case SELP:
			if ( selpObj == null )
				selpObj = new Builtin(BuiltinCode.SELP);
			return selpObj;
			
		default:
			// Unknown code --> return null
			return null;
		}
	}

	@Override
	public double execute (double in) 
		throws DMLRuntimeException 
	{
		switch(bFunc) {
			case SIN:    return FASTMATH ? FastMath.sin(in) : Math.sin(in);
			case COS:    return FASTMATH ? FastMath.cos(in) : Math.cos(in);
			case TAN:    return FASTMATH ? FastMath.tan(in) : Math.tan(in);
			case ASIN:   return FASTMATH ? FastMath.asin(in) : Math.asin(in);
			case ACOS:   return FASTMATH ? FastMath.acos(in) : Math.acos(in);
			case ATAN:   return Math.atan(in); //faster in Math
			case CEIL:   return FASTMATH ? FastMath.ceil(in) : Math.ceil(in);
			case FLOOR:  return FASTMATH ? FastMath.floor(in) : Math.floor(in);
			case LOG:    return FASTMATH ? FastMath.log(in) : Math.log(in);		
			case LOG_NZ: return (in==0) ? 0 : FASTMATH ? FastMath.log(in) : Math.log(in);
			case ABS:    return Math.abs(in); //no need for FastMath			
			case SIGN:	 return FASTMATH ? FastMath.signum(in) : Math.signum(in);			
			case SQRT:   return Math.sqrt(in); //faster in Math		
			case EXP:    return FASTMATH ? FastMath.exp(in) : Math.exp(in);		
			case ROUND: return Math.round(in); //no need for FastMath
			
			case PLOGP:
				if (in == 0.0)
					return 0.0;
				else if (in < 0)
					return Double.NaN;
				else
					return (in * (FASTMATH ? FastMath.log(in) : Math.log(in)));
			
			case SPROP:
				//sample proportion: P*(1-P)
				return in * (1 - in); 
	
			case SIGMOID:
				//sigmoid: 1/(1+exp(-x))
				return FASTMATH ? 1 / (1 + FastMath.exp(-in))  : 1 / (1 + Math.exp(-in));
			
			case SELP:
				//select positive: x*(x>0)
				return (in > 0) ? in : 0;
				
			default:
				throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}

	@Override
	public double execute (long in) throws DMLRuntimeException {
		return execute((double)in);
	}

	/*
	 * Builtin functions with two inputs
	 */	
	@Override
	public double execute (double in1, double in2) throws DMLRuntimeException {
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
			//return (Double.compare(in1, in2) >= 0 ? in1 : in2);
			return (in1 >= in2 ? in1 : in2);
		case MIN:
		case CUMMIN:
			//return (Double.compare(in1, in2) <= 0 ? in1 : in2);
			return (in1 <= in2 ? in1 : in2);
			
			// *** HACK ALERT *** HACK ALERT *** HACK ALERT ***
			// rowIndexMax() and its siblings require comparing four values, but
			// the aggregation API only allows two values. So the execute()
			// method receives as its argument the two cell values to be
			// compared and performs just the value part of the comparison. We
			// return an integer cast down to a double, since the aggregation
			// API doesn't have any way to return anything but a double. The
			// integer returned takes on three posssible values: //
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
		case LOG:
			//if ( in1 <= 0 )
			//	throw new DMLRuntimeException("Builtin.execute(): logarithm can be computed only for non-negative numbers.");
			if( FASTMATH )
				return (FastMath.log(in1)/FastMath.log(in2)); 
			else
				return (Math.log(in1)/Math.log(in2)); 
		case LOG_NZ:
			if( FASTMATH )
				return (in1==0) ? 0 : (FastMath.log(in1)/FastMath.log(in2)); 
			else
				return (in1==0) ? 0 : (Math.log(in1)/Math.log(in2)); 
		
			
		default:
			throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}
	
	/**
	 * Simplified version without exception handling
	 * 
	 * @param in1 double 1
	 * @param in2 double 2
	 * @return result
	 */
	public double execute2(double in1, double in2) 
	{
		switch(bFunc) {		
			case MAX:
			case CUMMAX:
				//return (Double.compare(in1, in2) >= 0 ? in1 : in2); 
				return (in1 >= in2 ? in1 : in2);
			case MIN:
			case CUMMIN:
				//return (Double.compare(in1, in2) <= 0 ? in1 : in2); 
				return (in1 <= in2 ? in1 : in2);
			case MAXINDEX: 
				return (in1 >= in2) ? 1 : 0;	
			case MININDEX: 
				return (in1 <= in2) ? 1 : 0;	
				
			default:
				// For performance reasons, avoid throwing an exception 
				return -1;
		}
	}
	
	@Override
	public double execute (long in1, long in2) throws DMLRuntimeException {
		switch(bFunc) {
		
		case MAX:    
		case CUMMAX:   return (in1 >= in2 ? in1 : in2); 
		
		case MIN:    
		case CUMMIN:   return (in1 <= in2 ? in1 : in2); 
		
		case MAXINDEX: return (in1 >= in2) ? 1 : 0;
		case MININDEX: return (in1 <= in2) ? 1 : 0;
		
		case LOG:
			//if ( in1 <= 0 )
			//	throw new DMLRuntimeException("Builtin.execute(): logarithm can be computed only for non-negative numbers.");
			if( FASTMATH )
				return (FastMath.log(in1)/FastMath.log(in2));
			else
				return (Math.log(in1)/Math.log(in2));
		case LOG_NZ:
			if( FASTMATH )
				return (in1==0) ? 0 : (FastMath.log(in1)/FastMath.log(in2)); 
			else
				return (in1==0) ? 0 : (Math.log(in1)/Math.log(in2)); 
		
				
		
		default:
			throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}

	@Override
	public String execute (String in1) 
		throws DMLRuntimeException 
	{
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
}

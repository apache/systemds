/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

import java.util.HashMap;

import org.apache.commons.math.util.FastMath;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;


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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public enum BuiltinFunctionCode { INVALID, SIN, COS, TAN, ASIN, ACOS, ATAN, LOG, MIN, MAX, ABS, SQRT, EXP, PLOGP, PRINT, NROW, NCOL, LENGTH, ROUND, PRINT2, MAXINDEX  };
	public BuiltinFunctionCode bFunc;
	
	private static final boolean FASTMATH = true;
	
	static public HashMap<String, BuiltinFunctionCode> String2BuiltinFunctionCode;
	static {
		String2BuiltinFunctionCode = new HashMap<String, BuiltinFunctionCode>();
		
		String2BuiltinFunctionCode.put( "sin"    , BuiltinFunctionCode.SIN);
		String2BuiltinFunctionCode.put( "cos"    , BuiltinFunctionCode.COS);
		String2BuiltinFunctionCode.put( "tan"    , BuiltinFunctionCode.TAN);
		String2BuiltinFunctionCode.put( "asin"   , BuiltinFunctionCode.ASIN);
		String2BuiltinFunctionCode.put( "acos"   , BuiltinFunctionCode.ACOS);
		String2BuiltinFunctionCode.put( "atan"   , BuiltinFunctionCode.ATAN);
		String2BuiltinFunctionCode.put( "log"    , BuiltinFunctionCode.LOG);
		String2BuiltinFunctionCode.put( "min"    , BuiltinFunctionCode.MIN);
		String2BuiltinFunctionCode.put( "max"    , BuiltinFunctionCode.MAX);
		String2BuiltinFunctionCode.put( "maxindex"    , BuiltinFunctionCode.MAXINDEX);
		String2BuiltinFunctionCode.put( "abs"    , BuiltinFunctionCode.ABS);
		String2BuiltinFunctionCode.put( "sqrt"   , BuiltinFunctionCode.SQRT);
		String2BuiltinFunctionCode.put( "exp"    , BuiltinFunctionCode.EXP);
		String2BuiltinFunctionCode.put( "plogp"  , BuiltinFunctionCode.PLOGP);
		String2BuiltinFunctionCode.put( "print"  , BuiltinFunctionCode.PRINT);
		String2BuiltinFunctionCode.put( "print2" , BuiltinFunctionCode.PRINT2);
		String2BuiltinFunctionCode.put( "nrow"   , BuiltinFunctionCode.NROW);
		String2BuiltinFunctionCode.put( "ncol"   , BuiltinFunctionCode.NCOL);
		String2BuiltinFunctionCode.put( "length" , BuiltinFunctionCode.LENGTH);
		String2BuiltinFunctionCode.put( "round"  , BuiltinFunctionCode.ROUND);
	}
	
	// We should create one object for every builtin function that we support
	private static Builtin sinObj = null, cosObj = null, tanObj = null, asinObj = null, acosObj = null, atanObj = null;
	private static Builtin logObj = null, minObj = null, maxObj = null, maxindexObj = null;
	private static Builtin absObj = null, sqrtObj = null, expObj = null, plogpObj = null, printObj = null;
	private static Builtin nrowObj = null, ncolObj = null, lengthObj = null, roundObj = null, print2Obj = null;
	
	private Builtin(BuiltinFunctionCode bf) {
		bFunc = bf;
	}
	
	public static Builtin getBuiltinFnObject (String str) {
		
		BuiltinFunctionCode code = String2BuiltinFunctionCode.get(str);
		
		if ( code == null ) 
			return null;
		
		switch ( code ) {
		case SIN:
			if ( sinObj == null )
				sinObj = new Builtin(BuiltinFunctionCode.SIN);
			return sinObj;
		
		case COS:
			if ( cosObj == null )
				cosObj = new Builtin(BuiltinFunctionCode.COS);
			return cosObj;
		case TAN:
			if ( tanObj == null )
				tanObj = new Builtin(BuiltinFunctionCode.TAN);
			return tanObj;
		case ASIN:
			if ( asinObj == null )
				asinObj = new Builtin(BuiltinFunctionCode.ASIN);
			return asinObj;
		
		case ACOS:
			if ( acosObj == null )
				acosObj = new Builtin(BuiltinFunctionCode.ACOS);
			return acosObj;
		case ATAN:
			if ( atanObj == null )
				atanObj = new Builtin(BuiltinFunctionCode.ATAN);
			return atanObj;
		case LOG:
			if ( logObj == null )
				logObj = new Builtin(BuiltinFunctionCode.LOG);
			return logObj;
		case MAX:
			if ( maxObj == null )
				maxObj = new Builtin(BuiltinFunctionCode.MAX);
			return maxObj;
		case MAXINDEX:
			if ( maxindexObj == null )
				maxindexObj = new Builtin(BuiltinFunctionCode.MAXINDEX);
			return maxindexObj;
		case MIN:
			if ( minObj == null )
				minObj = new Builtin(BuiltinFunctionCode.MIN);
			return minObj;
		case ABS:
			if ( absObj == null )
				absObj = new Builtin(BuiltinFunctionCode.ABS);
			return absObj;
		case SQRT:
			if ( sqrtObj == null )
				sqrtObj = new Builtin(BuiltinFunctionCode.SQRT);
			return sqrtObj;
		case EXP:
			if ( expObj == null )
				expObj = new Builtin(BuiltinFunctionCode.EXP);
			return expObj;
		case PLOGP:
			if ( plogpObj == null )
				plogpObj = new Builtin(BuiltinFunctionCode.PLOGP);
			return plogpObj;
		case PRINT:
			if ( printObj == null )
				printObj = new Builtin(BuiltinFunctionCode.PRINT);
			return printObj;
		case PRINT2:
			if ( print2Obj == null )
				print2Obj = new Builtin(BuiltinFunctionCode.PRINT2);
			return print2Obj;
		case NROW:
			if ( nrowObj == null )
				nrowObj = new Builtin(BuiltinFunctionCode.NROW);
			return nrowObj;
		case NCOL:
			if ( ncolObj == null )
				ncolObj = new Builtin(BuiltinFunctionCode.NCOL);
			return ncolObj;
		case LENGTH:
			if ( lengthObj == null )
				lengthObj = new Builtin(BuiltinFunctionCode.LENGTH);
			return lengthObj;
		case ROUND:
			if ( roundObj == null )
				roundObj = new Builtin(BuiltinFunctionCode.ROUND);
			return roundObj;
		}
		
		return null;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	public boolean checkArity(int _arity) throws DMLUnsupportedOperationException {
		switch (bFunc) {
		case ABS:
		case SIN:
		case COS:
		case TAN:
		case ASIN:
		case ACOS:
		case ATAN:
		case SQRT:
		case EXP:
		case PLOGP:
		case NROW:
		case NCOL:
		case LENGTH:
		case ROUND:
		case PRINT2:
		case MAXINDEX:
			return (_arity == 1);
		
		case LOG:
		case PRINT:
			return (_arity == 1 || _arity == 2);
			
		case MAX:
		case MIN:
			return (_arity == 2);
		default:
			throw new DMLUnsupportedOperationException("checkNumberOfOperands(): Unknown opcode: " + bFunc);
		}
	}
	
	public double execute (double in) throws DMLRuntimeException {
		switch(bFunc) {
		case SIN:    return FASTMATH ? FastMath.sin(in) : Math.sin(in);
		case COS:    return FASTMATH ? FastMath.cos(in) : Math.cos(in);
		case TAN:    return FASTMATH ? FastMath.tan(in) : Math.tan(in);
		case ASIN:   return FASTMATH ? FastMath.asin(in) : Math.asin(in);
		case ACOS:   return FASTMATH ? FastMath.acos(in) : Math.acos(in);
		case ATAN:   return Math.atan(in); //faster in Math
		
		case LOG:
			//if ( in <= 0 )
			//	throw new DMLRuntimeException("Builtin.execute(): logarithm can only be computed for non-negative numbers (input = " + in + ").");
			// for negative numbers, Math.log will return NaN
			return FASTMATH ? FastMath.log(in) : Math.log(in);
		
		case ABS:
			return Math.abs(in); //no need for FastMath
			
		case SQRT:
			//if ( in < 0 )
			//	throw new DMLRuntimeException("Builtin.execute(): squareroot can only be computed for non-negative numbers (input = " + in + ").");
			return Math.sqrt(in); //faster in Math
		
		case PLOGP:
			if (in == 0.0)
				return 0.0;
			else if (in < 0)
				return Double.NaN;
			else
				return (in * (FASTMATH ? FastMath.log(in) : Math.log(in)));
			
		case EXP:
			return FASTMATH ? FastMath.exp(in) : Math.exp(in);
		
		case ROUND:
			return Math.round(in); //no need for FastMath
			
		default:
			throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}

	public double execute (int in) throws DMLRuntimeException {
		return this.execute( (double) in);
	}

	/*
	 * Builtin functions with two inputs
	 */
	
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
			//return (Double.compare(in1, in2) >= 0 ? in1 : in2);
			return (in1 >= in2 ? in1 : in2);
		case MIN:
			//return (Double.compare(in1, in2) <= 0 ? in1 : in2);
			return (in1 <= in2 ? in1 : in2);
		case MAXINDEX: 
			return (in1 >= in2) ? 1 : 0;
		case LOG:
			if ( in1 <= 0 )
				throw new DMLRuntimeException("Builtin.execute(): logarithm can be computed only for non-negative numbers.");
			if( FASTMATH )
				return (FastMath.log(in1)/FastMath.log(in2)); 
			else
				return (Math.log(in1)/Math.log(in2)); 
				
		default:
			throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}
	
	/**
	 * Simplified version without exception handling
	 * 
	 * @param in1
	 * @param in2
	 * @return
	 */
	public double execute2(double in1, double in2) 
	{
		switch(bFunc) {		
			case MAX:
				//return (Double.compare(in1, in2) >= 0 ? in1 : in2); 
				return (in1 >= in2 ? in1 : in2);
			case MIN:
				//return (Double.compare(in1, in2) <= 0 ? in1 : in2); 
				return (in1 <= in2 ? in1 : in2);
			case MAXINDEX: 
				return (in1 >= in2) ? 1 : 0;	
		}
		return -1;
	}
	
	public double execute (int in1, int in2) throws DMLRuntimeException {
		switch(bFunc) {
		
		case MAX:    return (in1 >= in2 ? in1 : in2); 
		case MIN:    return (in1 <= in2 ? in1 : in2); 
		case MAXINDEX: return (in1 >= in2) ? 1 : 0;
		case LOG:
			if ( in1 <= 0 )
				throw new DMLRuntimeException("Builtin.execute(): logarithm can be computed only for non-negative numbers.");
			if( FASTMATH )
				return (FastMath.log(in1)/FastMath.log(in2));
			else
				return (Math.log(in1)/Math.log(in2));
				
		
		default:
			throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}

	// currently, it is used only for PRINT 
	public void execute (String in1, double in2) throws DMLRuntimeException {
		switch (bFunc) {
		case PRINT:
			System.out.println(in1 + " " + in2);
		default:
			throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}

	// currently, it is used only for PRINT 
	public void execute (String in1, int in2) throws DMLRuntimeException {
		switch (bFunc) {
		case PRINT:
			System.out.println(in1 + " " + in2);
		default:
			throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}

	// currently, it is used only for PRINT 
	public void execute (String in1, boolean in2) throws DMLRuntimeException {
		switch (bFunc) {
		case PRINT:
			System.out.println(in1 + " " + in2);
		default:
			throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}

	// currently, it is used only for PRINT 
	public String execute (String in1, String in2) throws DMLRuntimeException {
		switch (bFunc) {
		case PRINT:
			System.out.println(in1 + " " + in2);
			return null;
		default:
			throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}

	// currently, it is used only for PRINT 
	public String execute (String in1) throws DMLRuntimeException {
		switch (bFunc) {
		case PRINT2:
			System.out.println(in1);
			return null;
		default:
			throw new DMLRuntimeException("Builtin.execute(): Unknown operation: " + bFunc);
		}
	}


}

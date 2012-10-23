package com.ibm.bi.dml.runtime.functionobjects;

import java.util.HashMap;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;
import org.apache.commons.math.distribution.ExponentialDistributionImpl;
import org.apache.commons.math.distribution.FDistributionImpl;
import org.apache.commons.math.distribution.NormalDistributionImpl;
import org.apache.commons.math.distribution.TDistributionImpl;

import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.DMLRuntimeException;


/**
 *  Function object for builtin function that takes a list of name=value parameters.
 *  This class can not be instantiated elsewhere.
 */


public class ParameterizedBuiltin extends ValueFunction {
	
	public enum ParameterizedBuiltinCode { INVALID, CDF, CDF_NORMAL, CDF_EXP, CDF_CHISQ, CDF_F, CDF_T, RMEMPTY };
	public enum ProbabilityDistributionCode { INVALID, NORMAL, EXP, CHISQ, F, T };
	
	public ParameterizedBuiltinCode bFunc;
	
	
	static public HashMap<String, ParameterizedBuiltinCode> String2ParameterizedBuiltinCode;
	static {
		String2ParameterizedBuiltinCode = new HashMap<String, ParameterizedBuiltinCode>();
		
		String2ParameterizedBuiltinCode.put( "cdf", ParameterizedBuiltinCode.CDF);
		String2ParameterizedBuiltinCode.put( "rmempty", ParameterizedBuiltinCode.RMEMPTY);
	}
	
	static public HashMap<String, ProbabilityDistributionCode> String2DistCode;
	static {
		String2DistCode = new HashMap<String,ProbabilityDistributionCode>();
		
		String2DistCode.put("normal"	, ProbabilityDistributionCode.NORMAL);
		String2DistCode.put("exp"		, ProbabilityDistributionCode.EXP);
		String2DistCode.put("chisq"		, ProbabilityDistributionCode.CHISQ);
		String2DistCode.put("f"			, ProbabilityDistributionCode.F);
		String2DistCode.put("t"			, ProbabilityDistributionCode.T);
	}
	
	// We should create one object for every builtin function that we support
	private static ParameterizedBuiltin normalObj = null, expObj = null, chisqObj = null, fObj = null, tObj = null;
	
	private ParameterizedBuiltin(ParameterizedBuiltinCode bf) {
		bFunc = bf;
	}

	public static ParameterizedBuiltin getParameterizedBuiltinFnObject (String str) {
		return getParameterizedBuiltinFnObject (str, null);
	}

	public static ParameterizedBuiltin getParameterizedBuiltinFnObject (String str, String str2) {
		
		ParameterizedBuiltinCode code = String2ParameterizedBuiltinCode.get(str);
		
		switch ( code ) 
		{
			case CDF:
				// str2 will point the appropriate distribution
				ProbabilityDistributionCode dcode = String2DistCode.get(str2.toLowerCase());
				switch(dcode) {
				case NORMAL:
					if ( normalObj == null )
						normalObj = new ParameterizedBuiltin(ParameterizedBuiltinCode.CDF_NORMAL);
					return normalObj;
				case EXP:
					if ( expObj == null )
						expObj = new ParameterizedBuiltin(ParameterizedBuiltinCode.CDF_EXP);
					return expObj;
				case CHISQ:
					if ( chisqObj == null )
						chisqObj = new ParameterizedBuiltin(ParameterizedBuiltinCode.CDF_CHISQ);
					return chisqObj;
				case F:
					if ( fObj == null )
						fObj = new ParameterizedBuiltin(ParameterizedBuiltinCode.CDF_F);
					return fObj;
				case T:
					if ( tObj == null )
						tObj = new ParameterizedBuiltin(ParameterizedBuiltinCode.CDF_T);
					return tObj;
				default:
					return null;
				}
				
			case RMEMPTY:
				return new ParameterizedBuiltin(ParameterizedBuiltinCode.RMEMPTY);
		}
		
		
		return null;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	public double execute(HashMap<String,String> params) throws DMLRuntimeException {
		switch(bFunc) {
		case CDF_NORMAL:
		case CDF_EXP:
		case CDF_CHISQ:
		case CDF_F:
		case CDF_T:
			try {
				return computeCDF(bFunc, params);
			} catch (MathException e) {
				throw new DMLRuntimeException(e);
			}
			
		default:
			throw new DMLRuntimeException("ParameterizedBuiltin.execute(): Unknown operation: " + bFunc);
		}
	}
	
	private double computeCDF (ParameterizedBuiltinCode bFunc, HashMap<String,String> params ) throws MathException, DMLRuntimeException {
		
		double quantile = Double.parseDouble(params.get("target"));
		
		switch(bFunc) {
		case CDF_NORMAL:
			double mean = 0.0, sd = 1.0;
			String mean_s = params.get("mean"), sd_s = params.get("sd");
			if ( (mean_s != null && sd_s == null) 
					|| (mean_s == null && sd_s != null )) {
				throw new DMLRuntimeException("" +
						"Both mean and standard deviation must be provided to compute probabilities from normal distribution " +
						"(e.g., q = cumulativeProbability(1.5, dist=\"normal\", mean=1.2, sd=2.5))");
			}
			if ( params.get("mean") != null && params.get("sd") != null ) {
				mean = Double.parseDouble(params.get("mean"));
				sd = Double.parseDouble(params.get("sd"));
			}
			NormalDistributionImpl ndist = new NormalDistributionImpl(mean, sd);
			return ndist.cumulativeProbability(quantile);
		
		case CDF_EXP:
			if ( params.get("mean") == null ) {
				throw new DMLRuntimeException("" +
						"Mean must be specified to compute probabilities from exponential distribution " +
						"(e.g., q = cumulativeProbability(1.5, dist=\"exp\", mean=1.2))");
			}
			double exp_mean = Double.parseDouble(params.get("mean"));
			ExponentialDistributionImpl expdist = new ExponentialDistributionImpl(exp_mean);
			return expdist.cumulativeProbability(quantile);
		
		case CDF_CHISQ:
			if ( params.get("df") == null ) {
				throw new DMLRuntimeException("" +
						"Degrees of freedom is needed to compute probabilities from chi-squared distribution " +
						"(e.g., q = cumulativeProbability(1.5, dist=\"chisq\", df=20))");
			}
			int df = UtilFunctions.parseToInt(params.get("df"));
			ChiSquaredDistributionImpl chdist = new ChiSquaredDistributionImpl(df);
			return chdist.cumulativeProbability(quantile);
		
		case CDF_F:
			if ( params.get("df1") == null || params.get("df2") == null ) {
				throw new DMLRuntimeException("" +
						"Degrees of freedom is needed to compute probabilities from F distribution " +
						"(e.g., q = cumulativeProbability(1.5, dist=\"f\", df1=20, df2=30))");
			}
			int df1 = UtilFunctions.parseToInt(params.get("df1"));
			int df2 = UtilFunctions.parseToInt(params.get("df2"));
			FDistributionImpl fdist = new FDistributionImpl(df1, df2);
			return fdist.cumulativeProbability(quantile);
		case CDF_T:
			if ( params.get("df") == null ) {
				throw new DMLRuntimeException("" +
						"Degrees of freedom is needed to compute probabilities from T distribution " +
						"(e.g., q = cumulativeProbability(1.5, dist=\"t\", df=10))");
			}
			int t_df = UtilFunctions.parseToInt(params.get("df"));
			TDistributionImpl tdist = new TDistributionImpl(t_df);
			return tdist.cumulativeProbability(quantile);
		}
		
		return 0.0;
	}
}

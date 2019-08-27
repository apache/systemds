/*
 * Copyright 2018 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.common;

import java.util.EnumSet;
import java.util.HashMap;

import org.tugraz.sysds.common.Types.ReturnType;

/**
 * Enum to represent all builtin functions in the default name space.
 * Each function is either native or implemented by a DML script. In
 * case of DML script, these functions are loaded during parsing. As
 * always, user-defined DML-bodied functions take precedence over all
 * builtin functions.
 * 
 * To add a new builtin script function, simply add the definition here
 * as well as a dml file in scripts/builtin with a matching name. On 
 * building SystemDS, these scripts are packaged into the jar as well.
 */
public enum Builtins {
	//builtin functions
	ABS("abs", false),
	ACOS("acos", false),
	ASIN("asin", false),
	ATAN("atan", false),
	AVG_POOL("avg_pool", false),
	AVG_POOL_BACKWARD("avg_pool_backward", false),
	BATCH_NORM2D("batch_norm2d", false, ReturnType.MULTI_RETURN),
	BATCH_NORM2D_BACKWARD("batch_norm2d_backward", false, ReturnType.MULTI_RETURN),
	BIASADD("bias_add", false),
	BIASMULT("bias_multiply", false),
	BITWAND("bitwAnd", false),
	BITWOR("bitwOr", false),
	BITWXOR("bitwXor", false),
	BITWSHIFTL("bitwShiftL", false),
	BITWSHIFTR("bitwShiftR", false),
	CAST_AS_SCALAR("as.scalar", "castAsScalar", false),
	CAST_AS_MATRIX("as.matrix", false),
	CAST_AS_FRAME("as.frame", false),
	CAST_AS_DOUBLE("as.double", false),
	CAST_AS_INT("as.integer", false),
	CAST_AS_BOOLEAN("as.logical", "as.boolean", false),
	CBIND("cbind", "append", false),
	CEIL("ceil", "ceiling", false),
	COLMAX("colMaxs", false),
	COLMEAN("colMeans", false),
	COLMIN("colMins", false),
	COLPROD("colProds", false),
	COLSD("colSds", false),
	COLSUM("colSums", false),
	COLVAR("colVars", false),
	CONV2D("conv2d", false),
	CONV2D_BACKWARD_FILTER("conv2d_backward_filter", false),
	CONV2D_BACKWARD_DATA("conv2d_backward_data", false),
	COS("cos", false),
	COV("cov", false),
	COSH("cosh", false),
	CHOLESKY("cholesky", false),
	CUMMAX("cummax", false),
	CUMMIN("cummin", false),
	CUMPROD("cumprod", false),
	CUMSUM("cumsum", false),
	CUMSUMPROD("cumsumprod", false),
	DIAG("diag", false),
	EIGEN("eigen", false, ReturnType.MULTI_RETURN),
	EXISTS("exists", false),
	EXP("exp", false),
	EVAL("eval", false),
	FLOOR("floor", false),
	IFELSE("ifelse", false),
	IMG_MIRROR("img_mirror", true),
	IMG_BRIGHTNESS("img_brightness", true),
	IMG_CROP("img_crop", true),
	INTERQUANTILE("interQuantile", false),
	INVERSE("inv", "inverse", false),
	IQM("interQuartileMean", false),
	KMEANS("kmeans", true),
	LENGTH("length", false),
	LINEAGE("lineage", false),
	LIST("list", false),  //note: builtin and parbuiltin
	LM("lm", true),
	LMCG("lmCG", true),
	LMDS("lmDS", true),
	LMPREDICT("lmpredict", true),
	LOG("log", false),
	LSTM("lstm", false, ReturnType.MULTI_RETURN),
	LSTM_BACKWARD("lstm_backward", false, ReturnType.MULTI_RETURN),
	LU("lu", false, ReturnType.MULTI_RETURN),
	MEAN("mean", "avg", false),
	MIN("min", "pmin", false),
	MAX("max", "pmax", false),
	MAX_POOL("max_pool", false),
	MAX_POOL_BACKWARD("max_pool_backward", false),
	MEDIAN("median", false),
	MOMENT("moment", "centralMoment", false),
	NCOL("ncol", false),
	NORMALIZE("normalize", true),
	NROW("nrow", false),
	OUTER("outer", false),
	OUTLIER("outlier", true, false), //TODO parameterize opposite
	PPRED("ppred", false),
	PROD("prod", false),
	QR("qr", false, ReturnType.MULTI_RETURN),
	QUANTILE("quantile", false),
	RANGE("range", false),
	RBIND("rbind", false),
	REV("rev", false),
	ROUND("round", false),
	ROWINDEXMAX("rowIndexMax", false),
	ROWINDEXMIN("rowIndexMin", false),
	ROWMIN("rowMins", false),
	ROWMAX("rowMaxs", false),
	ROWMEAN("rowMeans", false),
	ROWPROD("rowProds", false),
	ROWSD("rowSds", false),
	ROWSUM("rowSums", false),
	ROWVAR("rowVars", false),
	SAMPLE("sample", false),
	SD("sd", false),
	SEQ("seq", false),
	SIGMOD("sigmoid", true),   // 1 / (1 + exp(-X))
	SIGN("sign", false),
	SIN("sin", false),
	SINH("sinh", false),
	STEPLM("steplm",true, ReturnType.MULTI_RETURN),
	SLICEFINDER("slicefinder", true),
	SOLVE("solve", false),
	SQRT("sqrt", false),
	SUM("sum", false),
	SVD("svd", false, ReturnType.MULTI_RETURN),
	TRANS("t", false),
	TABLE("table", "ctable", false),
	TAN("tan", false),
	TANH("tanh", false),
	TRACE("trace", false),
	VAR("var", false),
	XOR("xor", false),
	WINSORIZE("winsorize", true, false), //TODO parameterize w/ prob, min/max val
	
	//parameterized builtin functions
	CDF("cdf", false, true),
	GROUPEDAGG("aggregate", "groupedAggregate", false, true),
	INVCDF("icdf", false, true),
	LISTNV("list", false, true), //note: builtin and parbuiltin
	LOWER_TRI("lower.tri", false, true),
	ORDER("order", false, true),
	PARAMSERV("paramserv", false, true),
	PCHISQ("pchisq", false, true),
	PEXP("pexp", false, true),
	PF("pf", false, true),
	PNORM("pnorm", false, true),
	PT("pt", false, true),
	QCHISQ("qchisq", false, true),
	QF("qf", false, true),
	QNORM("qnorm", false, true),
	QT("qt", false, true),
	QEXP("qexp", false, true),
	REPLACE("replace", false, true),
	RMEMPTY("removeEmpty", false, true),
	SCALE("scale", true, false),     //TODO parameterize center & scale
	TIME("time", false),
	TOSTRING("toString", false, true),
	TRANSFORMAPPLY("transformapply", false, true),
	TRANSFORMCOLMAP("transformcolmap", false, true),
	TRANSFORMDECODE("transformdecode", false, true),
	TRANSFORMENCODE("transformencode", false, true),
	TRANSFORMMETA("transformmeta", false, true),
	UPPER_TRI("upper.tri", false, true);
	
	Builtins(String name, boolean script) {
		this(name, null, script, false, ReturnType.SINGLE_RETURN);
	}
	
	Builtins(String name, boolean script, ReturnType retType) {
		this(name, null, script, false, retType);
	}
	
	Builtins(String name, boolean script, boolean parameterized) {
		this(name, null, script, parameterized, ReturnType.SINGLE_RETURN);
	}
	
	Builtins(String name, String alias, boolean script) {
		this(name, alias, script, false, ReturnType.SINGLE_RETURN);
	}
	
	Builtins(String name, String alias, boolean script, boolean parameterized) {
		this(name, alias, script, parameterized, ReturnType.SINGLE_RETURN);
	}
	
	Builtins(String name, String alias, boolean script, boolean parameterized, ReturnType retType) {
		_name = name;
		_alias = alias;
		_script = script;
		_parameterized = parameterized;
		_retType = retType;
	}
	
	private final static String BUILTIN_DIR = "scripts/builtin/";
	private final static HashMap<String, Builtins> _map = new HashMap<>();
	
	static {
		//materialize lookup map for all builtin names
		for( Builtins b : EnumSet.allOf(Builtins.class) ) {
			_map.put(b.getName(), b);
			if( b.getAlias() != null )
				_map.put(b.getAlias(), b);
		}
	}
	
	private final String _name;
	private final String _alias;
	private final boolean _script;
	private final boolean _parameterized;
	private final ReturnType _retType;
	
	public String getName() {
		return _name;
	}
	
	public String getAlias() {
		return _alias;
	}
	
	public boolean isScript() {
		return _script;
	}
	
	public boolean isParameterized() {
		return _parameterized;
	}
	
	public boolean isMultiReturn() {
		return _retType == ReturnType.MULTI_RETURN;
	}
	
	public static boolean contains(String name, boolean script, boolean parameterized) {
		Builtins tmp = get(name);
		return tmp != null && script == tmp.isScript()
			&& parameterized == tmp.isParameterized();
	}
	
	public static Builtins get(String name) {
		if( name.equals("list") )
			return LIST; //unparameterized
		return _map.get(name);
	}
	
	public static Builtins get(String name, boolean params) {
		if( name.equals("list") )
			return params ? LISTNV : LIST;
		Builtins tmp = get(name);
		return tmp != null && (params == tmp.isParameterized()) ? tmp : null;
	}
	
	public static String getFilePath(String name) {
		StringBuilder sb = new StringBuilder();
		sb.append(BUILTIN_DIR);
		sb.append(name);
		sb.append(".dml");
		return sb.toString();
	}
}

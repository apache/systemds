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

/**
 * Enum to represent all builtin functions in the default name space.
 * Each function is either native or implemented by a DML script. In
 * case of DML script, these functions are loaded during parsing. As
 * always, user-defined DML-bodied functions take precedence over all
 * builtin functions.
 * 
 * To add a new builtin script function, simply add the definition here
 * as well as a dml file in script/builtin with a matching name.
 */
public enum Builtins {
	ABS("abs", false),
	ACOS("acos", false),
	ASIN("asin", false),
	ATAN("atan", false),
	AVG_POOL("avg_pool", false),
	AVG_POOL_BACKWARD("avg_pool_backward", false),
	BATCH_NORM2D("batch_norm2d", false),
	BATCH_NORM2D_BACKWARD("batch_norm2d_backward", false),
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
	EIGEN("eigen", false),
	EXISTS("exists", false),
	EXP("exp", false),
	EVAL("eval", false),
	FLOOR("floor", false),
	IFELSE("ifelse", false),
	INTERQUANTILE("interQuantile", false),
	INVERSE("inv", "inverse", false),
	IQM("interQuartileMean", false),
	LENGTH("length", false),
	LIST("list", false),
	LOG("log", false),
	LSTM("lstm", false),
	LSTM_BACKWARD("lstm_backward", false),
	LU("lu", false),
	MEAN("mean", "avg", false),
	MIN("min", "pmin", false),
	MAX("max", "pmax", false),
	MAX_POOL("max_pool", false),
	MAX_POOL_BACKWARD("max_pool_backward", false),
	MEDIAN("median", false),
	MOMENT("moment", "centralMoment", false),
	NCOL("ncol", false),
	NROW("nrow", false),
	OUTER("outer", false),
	PPRED("ppred", false),
	PROD("prod", false),
	QR("qr", false),
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
	SOLVE("solve", false),
	SQRT("sqrt", false),
	SUM("sum", false),
	SVD("svd", false),
	TRANS("t", false),
	TABLE("table", "ctable", false),
	TAN("tan", false),
	TANH("tanh", false),
	TRACE("trace", false),
	VAR("var", false),
	XOR("xor", false),
	
	//TODO handle parameterized builtins explicitly
	//TODO remove custom handling from parsing
	CDF("cdf", false),
	INVCDF("icdf", false),
	PCHISQ("pchisq", false),
	PEXP("pexp", false),
	PF("pf", false),
	PNORM("pnorm", false),
	PT("pt", false),
	QF("qf", false),
	QNORM("qnorm", false),
	QT("qt", false),
	QEXP("qexp", false),
	QCHISQ("qchisq", false),
	
	GROUPEDAGG("aggregate", "groupedAggregate", false),
	RMEMPTY("removeEmpty", false),
	REPLACE("replace", false),
	ORDER("order", false),
	LOWER_TRI("lower.tri", false),
	UPPER_TRI("upper.tri", false),
	
	TRANSFORMAPPLY("transformapply", false),
	TRANSFORMDECODE("transformdecode", false),
	TRANSFORMENCODE("transformencode", false),
	TRANSFORMCOLMAP("transformcolmap", false),
	TRANSFORMMETA("transformmeta", false),

	TOSTRING("toString", false),
	//LIST("LIST", false), TODO both builtin and parameterized builtin 
	PARAMSERV("paramserv", false);

	
	Builtins(String name, boolean script) {
		this(name, null, script, false);
	}
	
	Builtins(String name, boolean script, boolean parameterized) {
		this(name, null, script, parameterized);
	}
	
	Builtins(String name, String alias, boolean script) {
		this(name, alias, script, false);
	}
	
	Builtins(String name, String alias, boolean script, boolean parameterized) {
		_name = name;
		_alias = alias;
		_script = script;
		_parameterized = parameterized;
	}
	
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
	
	public static boolean contains(String name, boolean script) {
		Builtins tmp = _map.get(name);
		return tmp != null && script == tmp.isScript();
	}
	
	public static Builtins get(String name) {
		return _map.get(name);
	}
	
	public static Builtins get(String name, boolean params) {
		Builtins tmp = _map.get(name);
		return tmp != null && (params == tmp.isParameterized()) ? tmp : null;
	}
}

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

package org.apache.sysds.common;

import org.apache.sysds.lops.Append;
import org.apache.sysds.lops.Compression;
import org.apache.sysds.lops.DataGen;
import org.apache.sysds.lops.DeCompression;
import org.apache.sysds.lops.LeftIndex;
import org.apache.sysds.lops.Local;
import org.apache.sysds.lops.RightIndex;

import org.apache.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.hops.FunctionOp;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

public enum Opcodes {
	MMULT("ba+*", CPType.AggregateBinary),
	TAKPM("tak+*", CPType.AggregateTernary),
	TACKPM("tack+*", CPType.AggregateTernary),

	UAKP("uak+", CPType.AggregateUnary),
	UARKP("uark+", CPType.AggregateUnary),
	UACKP("uack+", CPType.AggregateUnary),
	UASQKP("uasqk+", CPType.AggregateUnary),
	UARSQKP("uarsqk+", CPType.AggregateUnary),
	UACSQKP("uacsqk+", CPType.AggregateUnary),
	UAMEAN("uamean", CPType.AggregateUnary),
	UARMEAN("uarmean", CPType.AggregateUnary),
	UACMEAN("uacmean", CPType.AggregateUnary),
	UAVAR("uavar", CPType.AggregateUnary),
	UARVAR("uarvar", CPType.AggregateUnary),
	UACVAR("uacvar", CPType.AggregateUnary),
	UAMAX("uamax", CPType.AggregateUnary),
	UARMAX("uarmax", CPType.AggregateUnary),
	UARIMAX("uarimax", CPType.AggregateUnary),
	UACMAX("uacmax", CPType.AggregateUnary),
	UAMIN("uamin", CPType.AggregateUnary),
	UARMIN("uarmin", CPType.AggregateUnary),
	UARIMIN("uarimin", CPType.AggregateUnary),
	UACMIN("uacmin", CPType.AggregateUnary),
	UAP("ua+", CPType.AggregateUnary),
	UARP("uar+", CPType.AggregateUnary),
	UACP("uac+", CPType.AggregateUnary),
	UAM("ua*", CPType.AggregateUnary),
	UARM("uar*", CPType.AggregateUnary),
	UACM("uac*", CPType.AggregateUnary),
	UATRACE("uatrace", CPType.AggregateUnary),
	UAKTRACE("uaktrace", CPType.AggregateUnary),

	NROW("nrow", CPType.AggregateUnary),
	NCOL("ncol", CPType.AggregateUnary),
	LENGTH("length", CPType.AggregateUnary),
	EXISTS("exists", CPType.AggregateUnary),
	LINEAGE("lineage", CPType.AggregateUnary),
	UACD("uacd", CPType.AggregateUnary),
	UACDR("uacdr", CPType.AggregateUnary),
	UACDC("uacdc", CPType.AggregateUnary),
	UACDAP("uacdap", CPType.AggregateUnary),
	UACDAPR("uacdapr", CPType.AggregateUnary),
	UACDAPC("uacdapc", CPType.AggregateUnary),
	UNIQUE("unique", CPType.AggregateUnary),
	UNIQUER("uniquer", CPType.AggregateUnary),
	UNIQUEC("uniquec", CPType.AggregateUnary),

	UAGGOUTERCHAIN("uaggouterchain", CPType.UaggOuterChain),

	// Arithmetic Instruction Opcodes
	PLUS("+", CPType.Binary),
	MINUS("-", CPType.Binary),
	MULT("*", CPType.Binary),
	DIV("/", CPType.Binary),
	MODULUS("%%", CPType.Binary),
	INTDIV("%/%", CPType.Binary),
	POW("^", CPType.Binary),
	MINUS1_MULT("1-*", CPType.Binary),	  //special * case
	POW2("^2", CPType.Binary),		//special ^ case
	MULT2("*2", CPType.Binary),	   //special * case
	MINUS_NZ("-nz", CPType.Binary),	 //special - case

	// Boolean Instruction Opcodes
	AND("&&", CPType.Binary),
	OR("||", CPType.Binary),
	XOR("xor", CPType.Binary),
	BITWAND("bitwAnd", CPType.Binary),
	BITWOR("bitwOr", CPType.Binary),
	BITWXOR("bitwXor", CPType.Binary),
	BITWSHIFTL("bitwShiftL", CPType.Binary),
	BITWSHIFTR("bitwShiftR", CPType.Binary),
	NOT("!", CPType.Unary),

	// Relational Instruction Opcodes
	EQUAL("==", CPType.Binary),
	NOTEQUAL("!=", CPType.Binary),
	LESS("<", CPType.Binary),
	GREATER(">", CPType.Binary),
	LESSEQUAL("<=", CPType.Binary),
	GREATEREQUAL(">=", CPType.Binary),

	// Builtin Instruction Opcodes
	LOG("log", CPType.Builtin),
	LOGNZ("log_nz", CPType.Builtin),

	SOLVE("solve", CPType.Binary),
	MAX("max", CPType.Binary),
	MIN("min", CPType.Binary),
	DROPINVALIDTYPE("dropInvalidType", CPType.Binary),
	DROPINVALIDLENGTH("dropInvalidLength", CPType.Binary),
	FREPLICATE("freplicate", CPType.Binary),
	VALUESWAP("valueSwap", CPType.Binary),
	APPLYSCHEMA("applySchema", CPType.Binary),
	MAP("_map", CPType.Ternary),

	NMAX("nmax", CPType.BuiltinNary),
	NMIN("nmin", CPType.BuiltinNary),
	NP("n+", CPType.BuiltinNary),
	NM("n*", CPType.BuiltinNary),

	EXP("exp", CPType.Unary),
	ABS("abs", CPType.Unary),
	SIN("sin", CPType.Unary),
	COS("cos", CPType.Unary),
	TAN("tan", CPType.Unary),
	SINH("sinh", CPType.Unary),
	COSH("cosh", CPType.Unary),
	TANH("tanh", CPType.Unary),
	ASIN("asin", CPType.Unary),
	ACOS("acos", CPType.Unary),
	ATAN("atan", CPType.Unary),
	SIGN("sign", CPType.Unary),
	SQRT("sqrt", CPType.Unary),
	SQRT_MATRIX_JAVA("sqrt_matrix_java", CPType.Unary),
	PLOGP("plogp", CPType.Unary),
	PRINT("print", CPType.Unary),
	ASSERT("assert", CPType.Unary),
	ROUND("round", CPType.Unary),
	CEIL("ceil", CPType.Unary),
	FLOOR("floor", CPType.Unary),
	UCUMKP("ucumk+", CPType.Unary),
	UCUMM("ucum*", CPType.Unary),
	UCUMKPM("ucumk+*", CPType.Unary),
	UCUMMIN("ucummin", CPType.Unary),
	UCUMMAX("ucummax", CPType.Unary),
	STOP("stop", CPType.Unary),
	INVERSE("inverse", CPType.Unary),
	CHOLESKY("cholesky", CPType.Unary),
	SPROP("sprop", CPType.Unary),
	SIGMOID("sigmoid", CPType.Unary),
	TYPEOF("typeOf", CPType.Unary),
	DETECTSCHEMA("detectSchema", CPType.Unary),
	COLNAMES("colnames", CPType.Unary),
	ISNA("isna", CPType.Unary),
	ISNAN("isnan", CPType.Unary),
	ISINF("isinf", CPType.Unary),
	PRINTF("printf", CPType.BuiltinNary),
	CBIND("cbind", CPType.BuiltinNary),
	RBIND("rbind", CPType.BuiltinNary),
	EVAL("eval", CPType.BuiltinNary),
	LIST("list", CPType.BuiltinNary),

	//Parametrized builtin functions
	AUTODIFF("autoDiff", CPType.ParameterizedBuiltin),
	CONTAINS("contains", CPType.ParameterizedBuiltin),
	PARAMSERV("paramserv", CPType.ParameterizedBuiltin),
	NVLIST("nvlist", CPType.ParameterizedBuiltin),
	CDF("cdf", CPType.ParameterizedBuiltin),
	INVCDF("invcdf", CPType.ParameterizedBuiltin),
	GROUPEDAGG("groupedagg", CPType.ParameterizedBuiltin),
	RMEMPTY("rmempty", CPType.ParameterizedBuiltin),
	REPLACE("replace", CPType.ParameterizedBuiltin),
	LOWERTRI("lowertri", CPType.ParameterizedBuiltin),
	UPPERTRI("uppertri", CPType.ParameterizedBuiltin),
	REXPAND("rexpand", CPType.ParameterizedBuiltin),
	TOSTRING("toString", CPType.ParameterizedBuiltin),
	TOKENIZE("tokenize", CPType.ParameterizedBuiltin),
	TRANSFORMAPPLY("transformapply", CPType.ParameterizedBuiltin),
	TRANSFORMDECODE("transformdecode", CPType.ParameterizedBuiltin),
	TRANSFORMCOLMAP("transformcolmap", CPType.ParameterizedBuiltin),
	TRANSFORMMETA("transformmeta", CPType.ParameterizedBuiltin),
	TRANSFORMENCODE("transformencode", CPType.MultiReturnParameterizedBuiltin),

	//Ternary instruction opcodes
	PM("+*", CPType.Ternary),
	MINUSMULT("-*", CPType.Ternary),
	IFELSE("ifelse", CPType.Ternary),

	//Variable instruction opcodes
	ASSIGNVAR("assignvar", CPType.Variable),
	CPVAR("cpvar", CPType.Variable),
	MVVAR("mvvar", CPType.Variable),
	RMVAR("rmvar", CPType.Variable),
	RMFILEVAR("rmfilevar", CPType.Variable),
	CAST_AS_SCALAR(OpOp1.CAST_AS_SCALAR.toString(), CPType.Variable),
	CAST_AS_MATRIX(OpOp1.CAST_AS_MATRIX.toString(), CPType.Variable),
	CAST_AS_FRAME_VAR("cast_as_frame", CPType.Variable),
	CAST_AS_FRAME(OpOp1.CAST_AS_FRAME.toString(), CPType.Variable),
	CAST_AS_LIST(OpOp1.CAST_AS_LIST.toString(), CPType.Variable),
	CAST_AS_DOUBLE(OpOp1.CAST_AS_DOUBLE.toString(), CPType.Variable),
	CAST_AS_INT(OpOp1.CAST_AS_INT.toString(), CPType.Variable),
	CAST_AS_BOOLEAN(OpOp1.CAST_AS_BOOLEAN.toString(), CPType.Variable),
	ATTACHFILETOVAR("attachfiletovar", CPType.Variable),
	READ("read", CPType.Variable),
	WRITE("write", CPType.Variable),
	CREATEVAR("createvar", CPType.Variable),

	//Reorg instruction opcodes
	TRANSPOSE("r'", CPType.Reorg),
	REV("rev", CPType.Reorg),
	ROLL("roll", CPType.Reorg),
	DIAG("rdiag", CPType.Reorg),
	RESHAPE("rshape", CPType.Reshape),
	SORT("rsort", CPType.Reorg),

	// Opcodes related to convolutions
	RELU_BACKWARD("relu_backward", CPType.Dnn),
	RELU_MAXPOOLING("relu_maxpooling", CPType.Dnn),
	RELU_MAXPOOLING_BACKWARD("relu_maxpooling_backward", CPType.Dnn),
	MAXPOOLING("maxpooling", CPType.Dnn),
	MAXPOOLING_BACKWARD("maxpooling_backward", CPType.Dnn),
	AVGPOOLING("avgpooling", CPType.Dnn),
	AVGPOOLING_BACKWARD("avgpooling_backward", CPType.Dnn),
	CONV2D("conv2d", CPType.Dnn),
	CONV2D_BIAS_ADD("conv2d_bias_add", CPType.Dnn),
	CONV2D_BACKWARD_FILTER("conv2d_backward_filter", CPType.Dnn),
	CONV2D_BACKWARD_DATA("conv2d_backward_data", CPType.Dnn),
	BIAS_ADD("bias_add", CPType.Dnn),
	BIAS_MULTIPLY("bias_multiply", CPType.Dnn),
	BATCH_NORM2D("batch_norm2d", CPType.Dnn),
	BATCH_NORM2D_BACKWARD("batch_norm2d_backward", CPType.Dnn),
	LSTM("lstm", CPType.Dnn),
	LSTM_BACKWARD("lstm_backward", CPType.Dnn),

	//Quaternary instruction opcodes
	WSLOSS("wsloss", CPType.Quaternary),
	WSIGMOID("wsigmoid", CPType.Quaternary),
	WDIVMM("wdivmm", CPType.Quaternary),
	WCEMM("wcemm", CPType.Quaternary),
	WUMM("wumm", CPType.Quaternary),

	//User-defined function Opcodes
	FCALL(FunctionOp.OPCODE, CPType.FCall),

	APPEND(Append.OPCODE, CPType.Append),
	REMOVE("remove", CPType.Append),

	//data generation opcodes
	RANDOM(DataGen.RAND_OPCODE, CPType.Rand),
	SEQUENCE(DataGen.SEQ_OPCODE, CPType.Rand),
	STRINGINIT(DataGen.SINIT_OPCODE, CPType.StringInit),
	SAMPLE(DataGen.SAMPLE_OPCODE, CPType.Rand),
	TIME(DataGen.TIME_OPCODE, CPType.Rand),
	FRAME(DataGen.FRAME_OPCODE, CPType.Rand),

	CTABLE("ctable", CPType.Ctable),
	CTABLEEXPAND("ctableexpand", CPType.Ctable),

	//central moment, covariance, quantiles (sort/pick)
	CM("cm", CPType.CentralMoment),
	COV("cov", CPType.Covariance),
	QSORT("qsort", CPType.QSort),
	QPICK("qpick", CPType.QPick),

	RIGHT_INDEX(RightIndex.OPCODE, CPType.MatrixIndexing),
	LEFT_INDEX(LeftIndex.OPCODE, CPType.MatrixIndexing),

	TSMM("tsmm", CPType.MMTSJ),
	PMM("pmm", CPType.PMMJ),
	MMCHAIN("mmchain", CPType.MMChain),

	QR("qr", CPType.MultiReturnBuiltin),
	LU("lu", CPType.MultiReturnBuiltin),
	EIGEN("eigen", CPType.MultiReturnBuiltin),
	FFT("fft", CPType.MultiReturnBuiltin),
	IFFT("ifft", CPType.MultiReturnComplexMatrixBuiltin),
	FFT_LINEARIZED("fft_linearized", CPType.MultiReturnBuiltin),
	IFFT_LINEARIZED("ifft_linearized", CPType.MultiReturnComplexMatrixBuiltin),
	STFT("stft", CPType.MultiReturnComplexMatrixBuiltin),
	SVD("svd", CPType.MultiReturnBuiltin),
	RCM("rcm", CPType.MultiReturnComplexMatrixBuiltin),

	PARTITION("partition", CPType.Partition),
	COMPRESS(Compression.OPCODE, CPType.Compression),
	DECOMPRESS(DeCompression.OPCODE, CPType.DeCompression),
	SPOOF("spoof", CPType.SpoofFused),
	PREFETCH("prefetch", CPType.Prefetch),
	EVICT("_evict", CPType.EvictLineageCache),
	BROADCAST("broadcast", CPType.Broadcast),
	TRIGREMOTE("trigremote", CPType.TrigRemote),
	LOCAL(Local.OPCODE, CPType.Local),

	SQL("sql", CPType.Sql);

	// Constructor
	Opcodes(String name, CPType type) {
		this._name = name;
		this._type = type;
	}

	// Fields
	private final String _name;
	private final CPType _type;

	private static final Map<String, Opcodes> _lookupMap = new HashMap<>();

	// Initialize lookup map
	static {
		for (Opcodes op : EnumSet.allOf(Opcodes.class)) {
			_lookupMap.put(op.toString(), op);
		}
	}

	// Getters
	@Override
	public String toString() {
		return _name;
	}

	public CPType getType() {
		return _type;
	}

	public static CPType getCPTypeByOpcode(String opcode) {
		for (Opcodes op : Opcodes.values()) {
			if (op.toString().equalsIgnoreCase(opcode.trim())) {
				return op.getType();
			}
		}
		return null;
	}
}

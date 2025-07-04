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

import org.apache.sysds.lops.*;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.hops.FunctionOp;

import java.util.HashMap;
import java.util.Map;

public enum Opcodes {
	MMULT("ba+*", InstructionType.AggregateBinary),
	TAKPM("tak+*", InstructionType.AggregateTernary),
	TACKPM("tack+*", InstructionType.AggregateTernary),

	UAKP("uak+", InstructionType.AggregateUnary),
	UARKP("uark+", InstructionType.AggregateUnary),
	UACKP("uack+", InstructionType.AggregateUnary),
	UASQKP("uasqk+", InstructionType.AggregateUnary),
	UARSQKP("uarsqk+", InstructionType.AggregateUnary),
	UACSQKP("uacsqk+", InstructionType.AggregateUnary),
	UAMEAN("uamean", InstructionType.AggregateUnary),
	UARMEAN("uarmean", InstructionType.AggregateUnary),
	UACMEAN("uacmean", InstructionType.AggregateUnary),
	UAVAR("uavar", InstructionType.AggregateUnary),
	UARVAR("uarvar", InstructionType.AggregateUnary),
	UACVAR("uacvar", InstructionType.AggregateUnary),
	UAMAX("uamax", InstructionType.AggregateUnary),
	UARMAX("uarmax", InstructionType.AggregateUnary),
	UARIMAX("uarimax", InstructionType.AggregateUnary),
	UACMAX("uacmax", InstructionType.AggregateUnary),
	UAMIN("uamin", InstructionType.AggregateUnary),
	UARMIN("uarmin", InstructionType.AggregateUnary),
	UARIMIN("uarimin", InstructionType.AggregateUnary),
	UACMIN("uacmin", InstructionType.AggregateUnary),
	UAP("ua+", InstructionType.AggregateUnary),
	UARP("uar+", InstructionType.AggregateUnary),
	UACP("uac+", InstructionType.AggregateUnary),
	UAM("ua*", InstructionType.AggregateUnary),
	UARM("uar*", InstructionType.AggregateUnary),
	UACM("uac*", InstructionType.AggregateUnary),
	UATRACE("uatrace", InstructionType.AggregateUnary),
	UAKTRACE("uaktrace", InstructionType.AggregateUnary),

	NROW("nrow", InstructionType.AggregateUnary),
	NCOL("ncol", InstructionType.AggregateUnary),
	LENGTH("length", InstructionType.AggregateUnary),
	EXISTS("exists", InstructionType.AggregateUnary),
	LINEAGE("lineage", InstructionType.AggregateUnary),
	UACD("uacd", InstructionType.AggregateUnary, InstructionType.AggregateUnarySketch),
	UACDR("uacdr", InstructionType.AggregateUnary, InstructionType.AggregateUnarySketch),
	UACDC("uacdc", InstructionType.AggregateUnary, InstructionType.AggregateUnarySketch),
	UACDAP("uacdap", InstructionType.AggregateUnary, InstructionType.AggregateUnarySketch),
	UACDAPR("uacdapr", InstructionType.AggregateUnary, InstructionType.AggregateUnarySketch),
	UACDAPC("uacdapc", InstructionType.AggregateUnary, InstructionType.AggregateUnarySketch),
	UNIQUE("unique", InstructionType.AggregateUnary),
	UNIQUER("uniquer", InstructionType.AggregateUnary),
	UNIQUEC("uniquec", InstructionType.AggregateUnary),

	UAGGOUTERCHAIN("uaggouterchain", InstructionType.UaggOuterChain),

	// Arithmetic Instruction Opcodes
	PLUS("+", InstructionType.Binary),
	MINUS("-", InstructionType.Binary),
	MULT("*", InstructionType.Binary),
	DIV("/", InstructionType.Binary),
	MODULUS("%%", InstructionType.Binary),
	INTDIV("%/%", InstructionType.Binary),
	POW("^", InstructionType.Binary),
	MINUS1_MULT("1-*", InstructionType.Binary),	  //special * case
	POW2("^2", InstructionType.Binary),		//special ^ case
	MULT2("*2", InstructionType.Binary),	   //special * case
	MINUS_NZ("-nz", InstructionType.Binary),	 //special - case

	// Boolean Instruction Opcodes
	AND("&&", InstructionType.Binary),
	OR("||", InstructionType.Binary),
	XOR("xor", InstructionType.Binary),
	BITWAND("bitwAnd", InstructionType.Binary),
	BITWOR("bitwOr", InstructionType.Binary),
	BITWXOR("bitwXor", InstructionType.Binary),
	BITWSHIFTL("bitwShiftL", InstructionType.Binary),
	BITWSHIFTR("bitwShiftR", InstructionType.Binary),
	NOT("!", InstructionType.Unary),

	// Relational Instruction Opcodes
	EQUAL("==", InstructionType.Binary),
	NOTEQUAL("!=", InstructionType.Binary),
	LESS("<", InstructionType.Binary),
	GREATER(">", InstructionType.Binary),
	LESSEQUAL("<=", InstructionType.Binary),
	GREATEREQUAL(">=", InstructionType.Binary),

	// Builtin Instruction Opcodes
	LOG("log", InstructionType.Builtin),
	LOGNZ("log_nz", InstructionType.Builtin),

	SOLVE("solve", InstructionType.Binary),
	MAX("max", InstructionType.Binary),
	MIN("min", InstructionType.Binary),
	DROPINVALIDTYPE("dropInvalidType", InstructionType.Binary),
	DROPINVALIDLENGTH("dropInvalidLength", InstructionType.Binary),
	FREPLICATE("freplicate", InstructionType.Binary),
	VALUESWAP("valueSwap", InstructionType.Binary),
	APPLYSCHEMA("applySchema", InstructionType.Binary),
	MAP("_map", InstructionType.Ternary),

	NMAX("nmax", InstructionType.BuiltinNary),
	NMIN("nmin", InstructionType.BuiltinNary),
	NP("n+", InstructionType.BuiltinNary),
	NM("n*", InstructionType.BuiltinNary),

	EXP("exp", InstructionType.Unary),
	ABS("abs", InstructionType.Unary),
	SIN("sin", InstructionType.Unary),
	COS("cos", InstructionType.Unary),
	TAN("tan", InstructionType.Unary),
	SINH("sinh", InstructionType.Unary),
	COSH("cosh", InstructionType.Unary),
	TANH("tanh", InstructionType.Unary),
	ASIN("asin", InstructionType.Unary),
	ACOS("acos", InstructionType.Unary),
	ATAN("atan", InstructionType.Unary),
	SIGN("sign", InstructionType.Unary),
	SQRT("sqrt", InstructionType.Unary),
	SQRT_MATRIX_JAVA("sqrt_matrix_java", InstructionType.Unary),
	PLOGP("plogp", InstructionType.Unary),
	PRINT("print", InstructionType.Unary),
	ASSERT("assert", InstructionType.Unary),
	ROUND("round", InstructionType.Unary),
	CEIL("ceil", InstructionType.Unary),
	FLOOR("floor", InstructionType.Unary),
	UCUMKP("ucumk+", InstructionType.Unary),
	UCUMM("ucum*", InstructionType.Unary),
	UCUMKPM("ucumk+*", InstructionType.Unary),
	UCUMMIN("ucummin", InstructionType.Unary),
	UCUMMAX("ucummax", InstructionType.Unary),
	STOP("stop", InstructionType.Unary),
	INVERSE("inverse", InstructionType.Unary),
	CHOLESKY("cholesky", InstructionType.Unary),
	DET("det", InstructionType.Unary),
	SPROP("sprop", InstructionType.Unary),
	SIGMOID("sigmoid", InstructionType.Unary),
	TYPEOF("typeOf", InstructionType.Unary),
	DETECTSCHEMA("detectSchema", InstructionType.Unary),
	COLNAMES("colnames", InstructionType.Unary),
	ISNA("isna", InstructionType.Unary),
	ISNAN("isnan", InstructionType.Unary),
	ISINF("isinf", InstructionType.Unary),
	PRINTF("printf", InstructionType.BuiltinNary),
	CBIND("cbind", InstructionType.BuiltinNary),
	RBIND("rbind", InstructionType.BuiltinNary),
	EVAL("eval", InstructionType.BuiltinNary),
	LIST("list", InstructionType.BuiltinNary),
	EINSUM("einsum", InstructionType.BuiltinNary),
	//Parametrized builtin functions
	AUTODIFF("autoDiff", InstructionType.ParameterizedBuiltin),
	CONTAINS("contains", InstructionType.ParameterizedBuiltin),
	PARAMSERV("paramserv", InstructionType.ParameterizedBuiltin),
	NVLIST("nvlist", InstructionType.ParameterizedBuiltin),
	CDF("cdf", InstructionType.ParameterizedBuiltin),
	INVCDF("invcdf", InstructionType.ParameterizedBuiltin),
	GROUPEDAGG("groupedagg", InstructionType.ParameterizedBuiltin),
	RMEMPTY("rmempty", InstructionType.ParameterizedBuiltin),
	REPLACE("replace", InstructionType.ParameterizedBuiltin),
	LOWERTRI("lowertri", InstructionType.ParameterizedBuiltin),
	UPPERTRI("uppertri", InstructionType.ParameterizedBuiltin),
	REXPAND("rexpand", InstructionType.ParameterizedBuiltin),
	TOSTRING("toString", InstructionType.ParameterizedBuiltin),
	TOKENIZE("tokenize", InstructionType.ParameterizedBuiltin),
	TRANSFORMAPPLY("transformapply", InstructionType.ParameterizedBuiltin),
	TRANSFORMDECODE("transformdecode", InstructionType.ParameterizedBuiltin),
	TRANSFORMCOLMAP("transformcolmap", InstructionType.ParameterizedBuiltin),
	TRANSFORMMETA("transformmeta", InstructionType.ParameterizedBuiltin),
	TRANSFORMENCODE("transformencode", InstructionType.MultiReturnParameterizedBuiltin,  InstructionType.MultiReturnBuiltin),


	//Ternary instruction opcodes
	PM("+*", InstructionType.Ternary),
	MINUSMULT("-*", InstructionType.Ternary),
	IFELSE("ifelse", InstructionType.Ternary),

	//Variable instruction opcodes
	ASSIGNVAR("assignvar", InstructionType.Variable),
	CPVAR("cpvar", InstructionType.Variable),
	MVVAR("mvvar", InstructionType.Variable),
	RMVAR("rmvar", InstructionType.Variable),
	RMFILEVAR("rmfilevar", InstructionType.Variable),
	CAST_AS_SCALAR(OpOp1.CAST_AS_SCALAR.toString(), InstructionType.Variable),
	CAST_AS_MATRIX(OpOp1.CAST_AS_MATRIX.toString(), InstructionType.Variable, InstructionType.Cast),
	CAST_AS_FRAME_VAR("cast_as_frame", InstructionType.Variable),
	CAST_AS_FRAME(OpOp1.CAST_AS_FRAME.toString(), InstructionType.Variable, InstructionType.Cast),
	CAST_AS_LIST(OpOp1.CAST_AS_LIST.toString(), InstructionType.Variable),
	CAST_AS_DOUBLE(OpOp1.CAST_AS_DOUBLE.toString(), InstructionType.Variable),
	CAST_AS_INT(OpOp1.CAST_AS_INT.toString(), InstructionType.Variable),
	CAST_AS_BOOLEAN(OpOp1.CAST_AS_BOOLEAN.toString(), InstructionType.Variable),
	ATTACHFILETOVAR("attachfiletovar", InstructionType.Variable),
	READ("read", InstructionType.Variable),
	WRITE("write", InstructionType.Variable, InstructionType.Write),
	CREATEVAR("createvar", InstructionType.Variable),

	//Reorg instruction opcodes
	TRANSPOSE("r'", InstructionType.Reorg),
	REV("rev", InstructionType.Reorg),
	ROLL("roll", InstructionType.Reorg),
	DIAG("rdiag", InstructionType.Reorg),
	RESHAPE("rshape", InstructionType.Reshape, InstructionType.MatrixReshape),
	SORT("rsort", InstructionType.Reorg),

	// Opcodes related to convolutions
	RELU_BACKWARD("relu_backward", InstructionType.Dnn),
	RELU_MAXPOOLING("relu_maxpooling", InstructionType.Dnn),
	RELU_MAXPOOLING_BACKWARD("relu_maxpooling_backward", InstructionType.Dnn),
	MAXPOOLING("maxpooling", InstructionType.Dnn),
	MAXPOOLING_BACKWARD("maxpooling_backward", InstructionType.Dnn),
	AVGPOOLING("avgpooling", InstructionType.Dnn),
	AVGPOOLING_BACKWARD("avgpooling_backward", InstructionType.Dnn),
	CONV2D("conv2d", InstructionType.Dnn),
	CONV2D_BIAS_ADD("conv2d_bias_add", InstructionType.Dnn),
	CONV2D_BACKWARD_FILTER("conv2d_backward_filter", InstructionType.Dnn),
	CONV2D_BACKWARD_DATA("conv2d_backward_data", InstructionType.Dnn),
	BIAS_ADD("bias_add", InstructionType.Dnn),
	BIAS_MULTIPLY("bias_multiply", InstructionType.Dnn),
	BATCH_NORM2D("batch_norm2d", InstructionType.Dnn),
	BATCH_NORM2D_BACKWARD("batch_norm2d_backward", InstructionType.Dnn),
	LSTM("lstm", InstructionType.Dnn),
	LSTM_BACKWARD("lstm_backward", InstructionType.Dnn),

	//Quaternary instruction opcodes
	WSLOSS("wsloss", InstructionType.Quaternary),
	WSIGMOID("wsigmoid", InstructionType.Quaternary),
	WDIVMM("wdivmm", InstructionType.Quaternary),
	WCEMM("wcemm", InstructionType.Quaternary),
	WUMM("wumm", InstructionType.Quaternary),

	//User-defined function Opcodes
	FCALL(FunctionOp.OPCODE, InstructionType.FCall),

	APPEND(Append.OPCODE, InstructionType.Append),
	REMOVE("remove", InstructionType.Append),

	//data generation opcodes
	RANDOM(DataGen.RAND_OPCODE, InstructionType.Rand),
	SEQUENCE(DataGen.SEQ_OPCODE, InstructionType.Rand),
	STRINGINIT(DataGen.SINIT_OPCODE, InstructionType.StringInit),
	SAMPLE(DataGen.SAMPLE_OPCODE, InstructionType.Rand),
	TIME(DataGen.TIME_OPCODE, InstructionType.Rand),
	FRAME(DataGen.FRAME_OPCODE, InstructionType.Rand),

	CTABLE("ctable", InstructionType.Ctable),
	CTABLEEXPAND("ctableexpand", InstructionType.Ctable),

	//central moment, covariance, quantiles (sort/pick)
	CM("cm", InstructionType.CentralMoment),
	COV("cov", InstructionType.Covariance),
	QSORT("qsort", InstructionType.QSort),
	QPICK("qpick", InstructionType.QPick),

	RIGHT_INDEX(RightIndex.OPCODE, InstructionType.MatrixIndexing),
	LEFT_INDEX(LeftIndex.OPCODE, InstructionType.MatrixIndexing),

	TSMM("tsmm", InstructionType.MMTSJ, InstructionType.TSMM),
	PMM("pmm", InstructionType.PMMJ, InstructionType.PMM),
	MMCHAIN("mmchain", InstructionType.MMChain),

	QR("qr", InstructionType.MultiReturnBuiltin),
	LU("lu", InstructionType.MultiReturnBuiltin),
	EIGEN("eigen", InstructionType.MultiReturnBuiltin),
	FFT("fft", InstructionType.MultiReturnBuiltin),
	IFFT("ifft", InstructionType.MultiReturnComplexMatrixBuiltin),
	FFT_LINEARIZED("fft_linearized", InstructionType.MultiReturnBuiltin),
	IFFT_LINEARIZED("ifft_linearized", InstructionType.MultiReturnComplexMatrixBuiltin),
	STFT("stft", InstructionType.MultiReturnComplexMatrixBuiltin),
	SVD("svd", InstructionType.MultiReturnBuiltin),
	RCM("rcm", InstructionType.MultiReturnComplexMatrixBuiltin),

	PARTITION("partition", InstructionType.Partition),
	COMPRESS(Compression.OPCODE, InstructionType.Compression, InstructionType.Compression),
	DECOMPRESS(DeCompression.OPCODE, InstructionType.DeCompression, InstructionType.DeCompression),
	QUANTIZE_COMPRESS("quantize_compress", InstructionType.QuantizeCompression),
	SPOOF("spoof", InstructionType.SpoofFused),
	PREFETCH("prefetch", InstructionType.Prefetch),
	EVICT("_evict", InstructionType.EvictLineageCache),
	BROADCAST("broadcast", InstructionType.Broadcast),
	TRIGREMOTE("trigremote", InstructionType.TrigRemote),
	LOCAL(Local.OPCODE, InstructionType.Local),

	SQL("sql", InstructionType.Sql),

	//SP Opcodes
	MAPMM("mapmm", InstructionType.MAPMM),
	MAPMMCHAIN("mapmmchain", InstructionType.MAPMMCHAIN),
	TSMM2("tsmm2", InstructionType.TSMM2),
	CPMM("cpmm", InstructionType.CPMM),
	RMM("rmm", InstructionType.RMM),
	ZIPMM("zipmm", InstructionType.ZIPMM),
	PMAPMM("pmapmm", InstructionType.PMAPMM),

	MAPLEFTINDEX("mapLeftIndex", InstructionType.MatrixIndexing),

	MAPPLUS("map+", InstructionType.Binary),
	MAPMINUS("map-", InstructionType.Binary),
	MAPMULT("map*", InstructionType.Binary),
	MAPDIV("map/", InstructionType.Binary),
	MAPMOD("map%%", InstructionType.Binary),
	MAPINTDIV("map%/%", InstructionType.Binary),
	MAPMINUS1_MULT("map1-*", InstructionType.Binary),
	MAPPOW("map^", InstructionType.Binary),
	MAPPM("map+*", InstructionType.Binary),
	MAPMINUSMULT("map-*", InstructionType.Binary),
	MAPDROPINVALIDLENGTH("mapdropInvalidLength", InstructionType.Binary),

	MAPGT("map>", InstructionType.Binary),
	MAPGE("map>=", InstructionType.Binary),
	MAPLT("map<", InstructionType.Binary),
	MAPLE("map<=", InstructionType.Binary),
	MAPEQ("map==", InstructionType.Binary),
	MAPNEQ("map!=", InstructionType.Binary),

	MAPAND("map&&", InstructionType.Binary),
	MAPOR("map||", InstructionType.Binary),
	MAPXOR("mapxor", InstructionType.Binary),
	MAPBITWAND("mapbitwAnd", InstructionType.Binary),
	MAPBITWOR("mapbitwOr", InstructionType.Binary),
	MAPBITWXOR("mapbitwXor", InstructionType.Binary),
	MAPBITWSHIFTL("mapbitwShiftL", InstructionType.Binary),
	MAPBITWSHIFTR("mapbitwShiftR", InstructionType.Binary),

	MAPMAX("mapmax", InstructionType.Binary),
	MAPMIN("mapmin", InstructionType.Binary),

	//REBLOCK Instruction Opcodes
	RBLK("rblk", null, InstructionType.Reblock),
	CSVRBLK("csvrblk", InstructionType.CSVReblock),
	LIBSVMRBLK("libsvmrblk", InstructionType.LIBSVMReblock),

	//Spark-specific instructions
	DEFAULTCPOPCODE(Checkpoint.DEFAULT_CP_OPCODE, InstructionType.Checkpoint),
	ASYNCCPOPCODE(Checkpoint.ASYNC_CP_OPCODE, InstructionType.Checkpoint),

	MAPGROUPEDAGG("mapgroupedagg", InstructionType.ParameterizedBuiltin),

	MAPPEND("mappend", InstructionType.MAppend),
	RAPPEND("rappend", InstructionType.RAppend),
	GAPPEND("gappend", InstructionType.GAppend),
	GALIGNEDAPPEND("galignedappend", InstructionType.GAlignedAppend),

	//quaternary instruction opcodes
	WEIGHTEDSQUAREDLOSS(WeightedSquaredLoss.OPCODE, InstructionType.Quaternary),
	WEIGHTEDSQUAREDLOSSR(WeightedSquaredLossR.OPCODE, InstructionType.Quaternary),
	WEIGHTEDSIGMOID(WeightedSigmoid.OPCODE, InstructionType.Quaternary),
	WEIGHTEDSIGMOIDR(WeightedSigmoidR.OPCODE, InstructionType.Quaternary),
	WEIGHTEDDIVMM(WeightedDivMM.OPCODE, InstructionType.Quaternary),
	WEIGHTEDDIVMMR(WeightedDivMMR.OPCODE, InstructionType.Quaternary),
	WEIGHTEDCROSSENTROPY(WeightedCrossEntropy.OPCODE, InstructionType.Quaternary),
	WEIGHTEDCROSSENTROPYR(WeightedCrossEntropyR.OPCODE, InstructionType.Quaternary),
	WEIGHTEDUNARYMM(WeightedUnaryMM.OPCODE, InstructionType.Quaternary),
	WEIGHTEDUNARYMMR(WeightedUnaryMMR.OPCODE, InstructionType.Quaternary),

	//cumsum/cumprod/cummin/cummax
	UCUMACKP("ucumack+", InstructionType.CumsumAggregate),
	UCUMACM("ucumac*", InstructionType.CumsumAggregate),
	UCUMACPM("ucumac+*", InstructionType.CumsumAggregate),
	UCUMACMIN("ucumacmin", InstructionType.CumsumAggregate),
	UCUMACMAX("ucumacmax", InstructionType.CumsumAggregate),
	BCUMOFFKP("bcumoffk+", InstructionType.CumsumOffset),
	BCUMOFFM("bcumoff*", InstructionType.CumsumOffset),
	BCUMOFFPM("bcumoff+*", InstructionType.CumsumOffset),
	BCUMOFFMIN("bcumoffmin", InstructionType.CumsumOffset),
	BCUMOFFMAX("bcumoffmax", InstructionType.CumsumOffset),

	BINUAGGCHAIN("binuaggchain", InstructionType.BinUaggChain),

	CASTDTM("castdtm", InstructionType.Variable, InstructionType.Cast),
	CASTDTF("castdtf", InstructionType.Variable, InstructionType.Cast),

	//FED Opcodes
	FEDINIT("fedinit", InstructionType.Init);

	// Constructors
	Opcodes(String name, InstructionType type) {
		this._name = name;
		this._type = type;
		this._spType=null;
		this._fedType=null;
	}

	Opcodes(String name, InstructionType type, InstructionType spType){
		this._name=name;
		this._type=type;
		this._spType=spType;
		this._fedType=null;
	}

	Opcodes(String name, InstructionType type, InstructionType spType, InstructionType fedType){
		this._name=name;
		this._type=type;
		this._spType=spType;
		this._fedType=fedType;
	}

	// Fields
	private final String _name;
	private final InstructionType _type;
	private final InstructionType _spType;
	private final InstructionType _fedType;

	private static final Map<String, Opcodes> _lookupMap = new HashMap<>();

	static {
		for (Opcodes op : Opcodes.values()) {
			if (op._name != null) {
				_lookupMap.put(op._name.toLowerCase(), op);
			}
		}
	}

	// Getters
	@Override
	public String toString() {
		return _name;
	}

	public InstructionType getType() {
		return _type;
	}

	public InstructionType getSpType() {
		return _spType != null ? _spType : _type;
	}

	public InstructionType getFedType(){
		return _fedType != null ? _fedType : _type;
	}

	public static InstructionType getTypeByOpcode(String opcode, Types.ExecType type) {
		if (opcode == null || opcode.trim().isEmpty()) {
			return null;
		}
		Opcodes op = _lookupMap.get(opcode.trim().toLowerCase());
		if( op != null ) {
			switch (type) {
				case SPARK:
					return (op.getSpType() != null) ? op.getSpType() : op.getType();
				case FED:
					return (op.getFedType() != null) ? op.getFedType() : op.getType();
				default:
					return op.getType();
			}
		}
		return null;
	}
}

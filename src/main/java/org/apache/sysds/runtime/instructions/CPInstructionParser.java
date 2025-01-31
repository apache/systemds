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

package org.apache.sysds.runtime.instructions;

import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.lops.Append;
import org.apache.sysds.lops.Compression;
import org.apache.sysds.lops.DataGen;
import org.apache.sysds.lops.DeCompression;
import org.apache.sysds.lops.LeftIndex;
import org.apache.sysds.lops.Local;
import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AggregateTernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AppendCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BroadcastCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BuiltinNaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.apache.sysds.runtime.instructions.cp.CentralMomentCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CompressionCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CovarianceCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CtableCPInstruction;
import org.apache.sysds.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysds.runtime.instructions.cp.DeCompressionCPInstruction;
import org.apache.sysds.runtime.instructions.cp.DnnCPInstruction;
import org.apache.sysds.runtime.instructions.cp.EvictCPInstruction;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.IndexingCPInstruction;
import org.apache.sysds.runtime.instructions.cp.LocalCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMChainCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnComplexMatrixBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.PMMJCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.PrefetchCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuantilePickCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuantileSortCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuaternaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ReorgCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ReshapeCPInstruction;
import org.apache.sysds.runtime.instructions.cp.SpoofCPInstruction;
import org.apache.sysds.runtime.instructions.cp.SqlCPInstruction;
import org.apache.sysds.runtime.instructions.cp.StringInitCPInstruction;
import org.apache.sysds.runtime.instructions.cp.TernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UaggOuterChainCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.instructions.cpfile.MatrixIndexingCPFileInstruction;

public class CPInstructionParser extends InstructionParser {
	protected static final Log LOG = LogFactory.getLog(CPInstructionParser.class.getName());

	public static final HashMap<String, CPType> String2CPInstructionType;
	static {
		String2CPInstructionType = new HashMap<>();
		String2CPInstructionType.put( "ba+*"    , CPType.AggregateBinary);
		String2CPInstructionType.put( "tak+*"   , CPType.AggregateTernary);
		String2CPInstructionType.put( "tack+*"  , CPType.AggregateTernary);
		
		String2CPInstructionType.put( "uak+"    , CPType.AggregateUnary);
		String2CPInstructionType.put( "uark+"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uack+"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uasqk+"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarsqk+" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacsqk+" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uamean"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarmean" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacmean" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uavar"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarvar"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacvar"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uamax"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarmax"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarimax" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacmax"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uamin"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarmin"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uarimin" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacmin"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "ua+"     , CPType.AggregateUnary);
		String2CPInstructionType.put( "uar+"    , CPType.AggregateUnary);
		String2CPInstructionType.put( "uac+"    , CPType.AggregateUnary);
		String2CPInstructionType.put( "ua*"     , CPType.AggregateUnary);
		String2CPInstructionType.put( "uar*"    , CPType.AggregateUnary);
		String2CPInstructionType.put( "uac*"    , CPType.AggregateUnary);
		String2CPInstructionType.put( "uatrace" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uaktrace", CPType.AggregateUnary);
		String2CPInstructionType.put( "nrow"    , CPType.AggregateUnary);
		String2CPInstructionType.put( "ncol"    , CPType.AggregateUnary);
		String2CPInstructionType.put( "length"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "exists"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "lineage" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacd"    , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacdr"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacdc"   , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacdap"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacdapr" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uacdapc" , CPType.AggregateUnary);
		String2CPInstructionType.put( "unique"  , CPType.AggregateUnary);
		String2CPInstructionType.put( "uniquer" , CPType.AggregateUnary);
		String2CPInstructionType.put( "uniquec" , CPType.AggregateUnary);

		String2CPInstructionType.put( "uaggouterchain", CPType.UaggOuterChain);
		
		// Arithmetic Instruction Opcodes 
		String2CPInstructionType.put( "+"    , CPType.Binary);
		String2CPInstructionType.put( "-"    , CPType.Binary);
		String2CPInstructionType.put( "*"    , CPType.Binary);
		String2CPInstructionType.put( "/"    , CPType.Binary);
		String2CPInstructionType.put( "%%"   , CPType.Binary);
		String2CPInstructionType.put( "%/%"  , CPType.Binary);
		String2CPInstructionType.put( "^"    , CPType.Binary);
		String2CPInstructionType.put( "1-*"  , CPType.Binary); //special * case
		String2CPInstructionType.put( "^2"   , CPType.Binary); //special ^ case
		String2CPInstructionType.put( "*2"   , CPType.Binary); //special * case
		String2CPInstructionType.put( "-nz"  , CPType.Binary); //special - case
		
		// Boolean Instruction Opcodes 
		String2CPInstructionType.put( "&&"   , CPType.Binary);
		String2CPInstructionType.put( "||"   , CPType.Binary);
		String2CPInstructionType.put( "xor"  , CPType.Binary);
		String2CPInstructionType.put( "bitwAnd", CPType.Binary);
		String2CPInstructionType.put( "bitwOr", CPType.Binary);
		String2CPInstructionType.put( "bitwXor", CPType.Binary);
		String2CPInstructionType.put( "bitwShiftL", CPType.Binary);
		String2CPInstructionType.put( "bitwShiftR", CPType.Binary);
		String2CPInstructionType.put( "!"    , CPType.Unary);

		// Relational Instruction Opcodes 
		String2CPInstructionType.put( "=="   , CPType.Binary);
		String2CPInstructionType.put( "!="   , CPType.Binary);
		String2CPInstructionType.put( "<"    , CPType.Binary);
		String2CPInstructionType.put( ">"    , CPType.Binary);
		String2CPInstructionType.put( "<="   , CPType.Binary);
		String2CPInstructionType.put( ">="   , CPType.Binary);
		
		// Builtin Instruction Opcodes 
		String2CPInstructionType.put( "log"  , CPType.Builtin);
		String2CPInstructionType.put( "log_nz"  , CPType.Builtin);

		String2CPInstructionType.put( "solve"  , CPType.Binary);
		String2CPInstructionType.put( "max"  , CPType.Binary);
		String2CPInstructionType.put( "min"  , CPType.Binary);
		String2CPInstructionType.put( "dropInvalidType"  , CPType.Binary);
		String2CPInstructionType.put( "dropInvalidLength"  , CPType.Binary);
		String2CPInstructionType.put( "freplicate"  , CPType.Binary);
		String2CPInstructionType.put( "valueSwap"  , CPType.Binary);
		String2CPInstructionType.put( "applySchema"  , CPType.Binary);
		String2CPInstructionType.put( "_map"  , CPType.Ternary); // _map represents the operation map

		String2CPInstructionType.put( "nmax", CPType.BuiltinNary);
		String2CPInstructionType.put( "nmin", CPType.BuiltinNary);
		String2CPInstructionType.put( "n+"  , CPType.BuiltinNary);
		String2CPInstructionType.put( "n*"  , CPType.BuiltinNary);

		String2CPInstructionType.put( "exp"   , CPType.Unary);
		String2CPInstructionType.put( "abs"   , CPType.Unary);
		String2CPInstructionType.put( "sin"   , CPType.Unary);
		String2CPInstructionType.put( "cos"   , CPType.Unary);
		String2CPInstructionType.put( "tan"   , CPType.Unary);
		String2CPInstructionType.put( "sinh"   , CPType.Unary);
		String2CPInstructionType.put( "cosh"   , CPType.Unary);
		String2CPInstructionType.put( "tanh"   , CPType.Unary);
		String2CPInstructionType.put( "asin"  , CPType.Unary);
		String2CPInstructionType.put( "acos"  , CPType.Unary);
		String2CPInstructionType.put( "atan"  , CPType.Unary);
		String2CPInstructionType.put( "sign"  , CPType.Unary);
		String2CPInstructionType.put( "sqrt"  , CPType.Unary);
		String2CPInstructionType.put( "plogp" , CPType.Unary);
		String2CPInstructionType.put( "print" , CPType.Unary);
		String2CPInstructionType.put( "assert" , CPType.Unary);
		String2CPInstructionType.put( "round" , CPType.Unary);
		String2CPInstructionType.put( "ceil"  , CPType.Unary);
		String2CPInstructionType.put( "floor" , CPType.Unary);
		String2CPInstructionType.put( "ucumk+", CPType.Unary);
		String2CPInstructionType.put( "ucum*" , CPType.Unary);
		String2CPInstructionType.put( "ucumk+*" , CPType.Unary);
		String2CPInstructionType.put( "ucummin", CPType.Unary);
		String2CPInstructionType.put( "ucummax", CPType.Unary);
		String2CPInstructionType.put( "stop"  , CPType.Unary);
		String2CPInstructionType.put( "inverse", CPType.Unary);
		String2CPInstructionType.put( "sqrt_matrix_java", CPType.Unary);
		String2CPInstructionType.put( "cholesky",CPType.Unary);
		String2CPInstructionType.put( "sprop", CPType.Unary);
		String2CPInstructionType.put( "sigmoid", CPType.Unary);
		String2CPInstructionType.put( "typeOf", CPType.Unary);
		String2CPInstructionType.put( "detectSchema", CPType.Unary);
		String2CPInstructionType.put( "colnames", CPType.Unary);
		String2CPInstructionType.put( "isna", CPType.Unary);
		String2CPInstructionType.put( "isnan", CPType.Unary);
		String2CPInstructionType.put( "isinf", CPType.Unary);
		String2CPInstructionType.put( "printf", CPType.BuiltinNary);
		String2CPInstructionType.put( "cbind",  CPType.BuiltinNary);
		String2CPInstructionType.put( "rbind",  CPType.BuiltinNary);
		String2CPInstructionType.put( "eval",   CPType.BuiltinNary);
		String2CPInstructionType.put( "list",   CPType.BuiltinNary);
		
		// Parameterized Builtin Functions
		String2CPInstructionType.put( "autoDiff" ,      CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "contains",       CPType.ParameterizedBuiltin);
		String2CPInstructionType.put("paramserv",       CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "nvlist",         CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "cdf",            CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "invcdf",         CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "groupedagg",     CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "rmempty" ,       CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "replace",        CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "lowertri",       CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "uppertri",       CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "rexpand",        CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "toString",       CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "tokenize",       CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "transformapply", CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "transformdecode",CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "transformcolmap",CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "transformmeta",  CPType.ParameterizedBuiltin);
		String2CPInstructionType.put( "transformencode",CPType.MultiReturnParameterizedBuiltin);
		
		// Ternary Instruction Opcodes
		String2CPInstructionType.put( "+*",      CPType.Ternary);
		String2CPInstructionType.put( "-*",      CPType.Ternary);
		String2CPInstructionType.put( "ifelse",  CPType.Ternary);
		
		// Variable Instruction Opcodes 
		String2CPInstructionType.put( "assignvar"   , CPType.Variable);
		String2CPInstructionType.put( "cpvar"       , CPType.Variable);
		String2CPInstructionType.put( "mvvar"       , CPType.Variable);
		String2CPInstructionType.put( "rmvar"       , CPType.Variable);
		String2CPInstructionType.put( "rmfilevar"   , CPType.Variable);
		String2CPInstructionType.put( OpOp1.CAST_AS_SCALAR.toString(),  CPType.Variable);
		String2CPInstructionType.put( OpOp1.CAST_AS_MATRIX.toString(),  CPType.Variable);
		String2CPInstructionType.put( "cast_as_frame", CPType.Variable);
		String2CPInstructionType.put( OpOp1.CAST_AS_FRAME.toString(),   CPType.Variable);
		String2CPInstructionType.put( OpOp1.CAST_AS_LIST.toString(),    CPType.Variable);
		String2CPInstructionType.put( OpOp1.CAST_AS_DOUBLE.toString(),  CPType.Variable);
		String2CPInstructionType.put( OpOp1.CAST_AS_INT.toString(),     CPType.Variable);
		String2CPInstructionType.put( OpOp1.CAST_AS_BOOLEAN.toString(), CPType.Variable);
		String2CPInstructionType.put( "attachfiletovar"  , CPType.Variable);
		String2CPInstructionType.put( "read"        , CPType.Variable);
		String2CPInstructionType.put( "write"       , CPType.Variable);
		String2CPInstructionType.put( "createvar"   , CPType.Variable);

		// Reorg Instruction Opcodes (repositioning of existing values)
		String2CPInstructionType.put( "r'"          , CPType.Reorg);
		String2CPInstructionType.put( "rev"         , CPType.Reorg);
		String2CPInstructionType.put( "roll"         , CPType.Reorg);
		String2CPInstructionType.put( "rdiag"       , CPType.Reorg);
		String2CPInstructionType.put( "rshape"      , CPType.Reshape);
		String2CPInstructionType.put( "rsort"       , CPType.Reorg);

		// Opcodes related to convolutions
		String2CPInstructionType.put( "relu_backward"      , CPType.Dnn);
		String2CPInstructionType.put( "relu_maxpooling"      , CPType.Dnn);
		String2CPInstructionType.put( "relu_maxpooling_backward"      , CPType.Dnn);
		String2CPInstructionType.put( "maxpooling"      , CPType.Dnn);
		String2CPInstructionType.put( "maxpooling_backward"      , CPType.Dnn);
		String2CPInstructionType.put( "avgpooling"      , CPType.Dnn);
		String2CPInstructionType.put( "avgpooling_backward"      , CPType.Dnn);
		String2CPInstructionType.put( "conv2d"      , CPType.Dnn);
		String2CPInstructionType.put( "conv2d_bias_add"      , CPType.Dnn);
		String2CPInstructionType.put( "conv2d_backward_filter"      , CPType.Dnn);
		String2CPInstructionType.put( "conv2d_backward_data"      , CPType.Dnn);
		String2CPInstructionType.put( "bias_add"      , CPType.Dnn);
		String2CPInstructionType.put( "bias_multiply"      , CPType.Dnn);
		String2CPInstructionType.put( "batch_norm2d",           CPType.Dnn);
		String2CPInstructionType.put( "batch_norm2d_backward",  CPType.Dnn);
		String2CPInstructionType.put( "lstm"      , CPType.Dnn);
		String2CPInstructionType.put( "lstm_backward"      , CPType.Dnn);

		// Quaternary instruction opcodes
		String2CPInstructionType.put( "wsloss"  , CPType.Quaternary);
		String2CPInstructionType.put( "wsigmoid", CPType.Quaternary);
		String2CPInstructionType.put( "wdivmm",   CPType.Quaternary);
		String2CPInstructionType.put( "wcemm",    CPType.Quaternary);
		String2CPInstructionType.put( "wumm",     CPType.Quaternary);
		
		// User-defined function Opcodes
		String2CPInstructionType.put(FunctionOp.OPCODE, CPType.FCall);

		String2CPInstructionType.put(Append.OPCODE, CPType.Append);
		String2CPInstructionType.put( "remove",      CPType.Append);
		
		// data generation opcodes
		String2CPInstructionType.put( DataGen.RAND_OPCODE   , CPType.Rand);
		String2CPInstructionType.put( DataGen.SEQ_OPCODE    , CPType.Rand);
		String2CPInstructionType.put( DataGen.SINIT_OPCODE  , CPType.StringInit);
		String2CPInstructionType.put( DataGen.SAMPLE_OPCODE , CPType.Rand);
		String2CPInstructionType.put( DataGen.TIME_OPCODE   , CPType.Rand);
		String2CPInstructionType.put( DataGen.FRAME_OPCODE   , CPType.Rand);

		String2CPInstructionType.put( "ctable",       CPType.Ctable);
		String2CPInstructionType.put( "ctableexpand", CPType.Ctable);
		
		//central moment, covariance, quantiles (sort/pick)
		String2CPInstructionType.put( "cm",    CPType.CentralMoment);
		String2CPInstructionType.put( "cov",   CPType.Covariance);
		String2CPInstructionType.put( "qsort", CPType.QSort);
		String2CPInstructionType.put( "qpick", CPType.QPick);
		
		
		String2CPInstructionType.put( RightIndex.OPCODE, CPType.MatrixIndexing);
		String2CPInstructionType.put( LeftIndex.OPCODE, CPType.MatrixIndexing);
	
		String2CPInstructionType.put( "tsmm",    CPType.MMTSJ);
		String2CPInstructionType.put( "pmm",     CPType.PMMJ);
		String2CPInstructionType.put( "mmchain", CPType.MMChain);
		
		String2CPInstructionType.put( "qr",    CPType.MultiReturnBuiltin);
		String2CPInstructionType.put( "lu",    CPType.MultiReturnBuiltin);
		String2CPInstructionType.put( "eigen", CPType.MultiReturnBuiltin);
		String2CPInstructionType.put( "fft",   CPType.MultiReturnBuiltin);
		String2CPInstructionType.put( "ifft",  CPType.MultiReturnComplexMatrixBuiltin);
		String2CPInstructionType.put( "fft_linearized", CPType.MultiReturnBuiltin);
		String2CPInstructionType.put( "ifft_linearized", CPType.MultiReturnComplexMatrixBuiltin);
		String2CPInstructionType.put( "img_transform_matrix", CPType.MultiReturnComplexMatrixBuiltin);
		String2CPInstructionType.put( "stft", CPType.MultiReturnComplexMatrixBuiltin);
		String2CPInstructionType.put( "svd",   CPType.MultiReturnBuiltin);
		String2CPInstructionType.put( "rcm",   CPType.MultiReturnComplexMatrixBuiltin);
		
		String2CPInstructionType.put( "partition", CPType.Partition);
		String2CPInstructionType.put( Compression.OPCODE,  CPType.Compression);
		String2CPInstructionType.put( DeCompression.OPCODE, CPType.DeCompression);
		String2CPInstructionType.put( "spoof",     CPType.SpoofFused);
		String2CPInstructionType.put( "prefetch",  CPType.Prefetch);
		String2CPInstructionType.put( "_evict",  CPType.EvictLineageCache);
		String2CPInstructionType.put( "broadcast",  CPType.Broadcast);
		String2CPInstructionType.put( "trigremote",  CPType.TrigRemote);
		String2CPInstructionType.put( Local.OPCODE, CPType.Local);
		
		String2CPInstructionType.put( "sql", CPType.Sql);
	}

	public static CPInstruction parseSingleInstruction (String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		CPType cptype = InstructionUtils.getCPType(str); 
		if ( cptype == null ) 
			throw new DMLRuntimeException("Unable derive cptype for instruction: " + str);
		CPInstruction cpinst = parseSingleInstruction(cptype, str);
		if ( cpinst == null )
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return cpinst;
	}
	
	public static CPInstruction parseSingleInstruction ( CPType cptype, String str ) {
		ExecType execType;
		if ( str == null || str.isEmpty() ) 
			return null;
		switch(cptype) {
			case AggregateUnary:
				return AggregateUnaryCPInstruction.parseInstruction(str);
			
			case AggregateBinary:
				return AggregateBinaryCPInstruction.parseInstruction(str);
	
			case AggregateTernary:
				return AggregateTernaryCPInstruction.parseInstruction(str);
			
			case Unary:
				return UnaryCPInstruction.parseInstruction(str);

			case Binary:
				return BinaryCPInstruction.parseInstruction(str);
			
			case Ternary:
				return TernaryCPInstruction.parseInstruction(str);
			
			case Quaternary:
				return QuaternaryCPInstruction.parseInstruction(str);
			
			case BuiltinNary:
				return BuiltinNaryCPInstruction.parseInstruction(str);
			
			case Ctable:
				return CtableCPInstruction.parseInstruction(str);
			
			case Reorg:
				return ReorgCPInstruction.parseInstruction(str);
				
			case Dnn:
				 return DnnCPInstruction.parseInstruction(str);
				
			case UaggOuterChain:
				return UaggOuterChainCPInstruction.parseInstruction(str);
				
			case Reshape:
				return ReshapeCPInstruction.parseInstruction(str);
	
			case Append:
				return AppendCPInstruction.parseInstruction(str);
			
			case Variable:
				return VariableCPInstruction.parseInstruction(str);
				
			case Rand:
				return DataGenCPInstruction.parseInstruction(str);

			case StringInit:
				return StringInitCPInstruction.parseInstruction(str);
				
			case FCall:
				return FunctionCallCPInstruction.parseInstruction(str);

			case ParameterizedBuiltin:
				return ParameterizedBuiltinCPInstruction.parseInstruction(str);
			
			case MultiReturnParameterizedBuiltin:
				return MultiReturnParameterizedBuiltinCPInstruction.parseInstruction(str);
		
			case MultiReturnComplexMatrixBuiltin:
				return MultiReturnComplexMatrixBuiltinCPInstruction.parseInstruction(str);
				
			case MultiReturnBuiltin:
				return MultiReturnBuiltinCPInstruction.parseInstruction(str);
				
			case QSort:
				return QuantileSortCPInstruction.parseInstruction(str);
			
			case QPick:
				return QuantilePickCPInstruction.parseInstruction(str);
			
			case MatrixIndexing:
				execType = ExecType.valueOf( str.split(Instruction.OPERAND_DELIM)[0] ); 
				if( execType == ExecType.CP )
					return IndexingCPInstruction.parseInstruction(str);
				else //exectype CP_FILE
					return MatrixIndexingCPFileInstruction.parseInstruction(str);
			
			case Builtin: 
				String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
				if(parts[0].equals("log") || parts[0].equals("log_nz")) {
					if(InstructionUtils.isInteger(parts[3])) // B=log(A), y=log(x)
						// We exploit the fact the number of threads is specified as an integer at parts 3.
						return UnaryCPInstruction.parseInstruction(str);
					else // B=log(A,10), y=log(x,10)
						return BinaryCPInstruction.parseInstruction(str);
				}
				throw new DMLRuntimeException("Invalid Builtin Instruction: " + str );
			
			case MMTSJ:
				return MMTSJCPInstruction.parseInstruction(str);
			
			case PMMJ:
				return PMMJCPInstruction.parseInstruction(str);
			
			case MMChain:
				return MMChainCPInstruction.parseInstruction(str);
			
			case CentralMoment:
				return CentralMomentCPInstruction.parseInstruction(str);
	
			case Covariance:
				return CovarianceCPInstruction.parseInstruction(str);

			case Compression:
				return CompressionCPInstruction.parseInstruction(str);
			
			case DeCompression:
				return DeCompressionCPInstruction.parseInstruction(str);
				
			case Local:
				return LocalCPInstruction.parseInstruction(str);

			case SpoofFused:
				return SpoofCPInstruction.parseInstruction(str);
				
			case Sql:
				return SqlCPInstruction.parseInstruction(str);
				
			case Prefetch:
				return PrefetchCPInstruction.parseInstruction(str);
				
			case Broadcast:
				return BroadcastCPInstruction.parseInstruction(str);

			case EvictLineageCache:
				return EvictCPInstruction.parseInstruction(str);
			
			default:
				throw new DMLRuntimeException("Invalid CP Instruction Type: " + cptype );
		}
	}
}

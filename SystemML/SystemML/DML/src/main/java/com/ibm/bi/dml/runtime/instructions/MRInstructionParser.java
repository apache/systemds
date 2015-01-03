/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions;

import java.util.HashMap;

import com.ibm.bi.dml.lops.BinaryM;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.MapMult;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateUnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AppendInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.BinUaggChainInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.BinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.BinaryMInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CM_N_COVInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CSVReblockInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CSVWriteInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CombineBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CombineTertiaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CombineUnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CumsumAggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CumsumOffsetInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CumsumSplitInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.DataGenMRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.DataPartitionMRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MMTSJMRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MapMultChainInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MatrixReshapeMRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ParameterizedBuiltinMRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.PickByCountInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RandInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RemoveEmptyMRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReorgInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReplicateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ScalarInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.SeqInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.TertiaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.UnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ZeroOutInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction.MRINSTRUCTION_TYPE;


public class MRInstructionParser extends InstructionParser 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	static public HashMap<String, MRINSTRUCTION_TYPE> String2MRInstructionType;
	static {
		String2MRInstructionType = new HashMap<String, MRINSTRUCTION_TYPE>();
		
		// AGG Instruction Opcodes 
		String2MRInstructionType.put( "a+"    , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "ak+"   , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "a*"    , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "amax"  , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "amin"  , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "amean"  , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "arimax"  , MRINSTRUCTION_TYPE.Aggregate);
		String2MRInstructionType.put( "arimin"  , MRINSTRUCTION_TYPE.Aggregate);

		// AGG_BINARY Instruction Opcodes 
		String2MRInstructionType.put( "cpmm" 	, MRINSTRUCTION_TYPE.AggregateBinary);
		String2MRInstructionType.put( "rmm"  	, MRINSTRUCTION_TYPE.AggregateBinary);
		String2MRInstructionType.put( MapMult.OPCODE, MRINSTRUCTION_TYPE.AggregateBinary);
		
		// AGG_UNARY Instruction Opcodes 
		String2MRInstructionType.put( "ua+"   , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uar+"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uac+"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uak+"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uark+" , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uack+" , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uamean", MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarmean",MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uacmean",MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "ua*"   , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uamax" , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uamin" , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uatrace" , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uaktrace", MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarmax"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarimax"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uacmax"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarmin"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uarimin"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "uacmin"  , MRINSTRUCTION_TYPE.AggregateUnary);

		// BUILTIN Instruction Opcodes 
		String2MRInstructionType.put( "abs"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "sin"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "cos"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "tan"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "asin" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "acos" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "atan" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "sqrt" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "exp"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "log"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "slog" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "pow"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "round", MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "ceil" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "floor", MRINSTRUCTION_TYPE.Unary);

		// Specific UNARY Instruction Opcodes
		String2MRInstructionType.put( "tsmm" , MRINSTRUCTION_TYPE.MMTSJ);
		String2MRInstructionType.put( "mapmultchain", MRINSTRUCTION_TYPE.MapMultChain);
		String2MRInstructionType.put( "binuaggchain", MRINSTRUCTION_TYPE.BinUaggChain);
		
		// BINARY and SCALAR Instruction Opcodes 
		String2MRInstructionType.put( "+"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "-"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "s-r"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "*"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "/"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "%%"   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "%/%"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "so"   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "^"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "max"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "min"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( ">"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( ">="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "<"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "<="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "=="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "!="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "^"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "^2"   , MRINSTRUCTION_TYPE.ArithmeticBinary); //special ^ case
		String2MRInstructionType.put( "^2c-" , MRINSTRUCTION_TYPE.ArithmeticBinary); //special ^ case
		String2MRInstructionType.put( "*2"   , MRINSTRUCTION_TYPE.ArithmeticBinary); //special * case
		String2MRInstructionType.put( "map+"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map-"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map*"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map/"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map%%"   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map%/%"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map^"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "mapmax"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "mapmin"  , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map>"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map>="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map<"    , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map<="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map=="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		String2MRInstructionType.put( "map!="   , MRINSTRUCTION_TYPE.ArithmeticBinary);
		
		// REORG Instruction Opcodes 
		String2MRInstructionType.put( "r'"     , MRINSTRUCTION_TYPE.Reorg);
		String2MRInstructionType.put( "rdiag"  , MRINSTRUCTION_TYPE.Reorg);
		
		// REPLICATE Instruction Opcodes
		String2MRInstructionType.put( "rep"     , MRINSTRUCTION_TYPE.Replicate);
		
		// DataGen Instruction Opcodes 
		String2MRInstructionType.put( DataGen.RAND_OPCODE   , MRINSTRUCTION_TYPE.Rand);
		String2MRInstructionType.put( DataGen.SEQ_OPCODE   , MRINSTRUCTION_TYPE.Seq);
		
		// REBLOCK Instruction Opcodes 
		String2MRInstructionType.put( "rblk"   , MRINSTRUCTION_TYPE.Reblock);
		String2MRInstructionType.put( "csvrblk", MRINSTRUCTION_TYPE.CSVReblock);
		
		// Tertiary Reorg Instruction Opcodes 
		String2MRInstructionType.put( "ctabletransform", MRINSTRUCTION_TYPE.Tertiary);
		String2MRInstructionType.put( "ctabletransformscalarweight", MRINSTRUCTION_TYPE.Tertiary);
		String2MRInstructionType.put( "ctableexpandscalarweight", MRINSTRUCTION_TYPE.Tertiary);
		String2MRInstructionType.put( "ctabletransformhistogram", MRINSTRUCTION_TYPE.Tertiary);
		String2MRInstructionType.put( "ctabletransformweightedhistogram", MRINSTRUCTION_TYPE.Tertiary);
		
		// Combine Instruction Opcodes
		String2MRInstructionType.put( "combinebinary" , MRINSTRUCTION_TYPE.CombineBinary);
		String2MRInstructionType.put( "combineunary"  , MRINSTRUCTION_TYPE.CombineUnary);
		String2MRInstructionType.put( "combinetertiary" , MRINSTRUCTION_TYPE.CombineTertiary);
		
		// PickByCount Instruction Opcodes
		String2MRInstructionType.put( "valuepick"  , MRINSTRUCTION_TYPE.PickByCount);  // for quantile()
		String2MRInstructionType.put( "rangepick"  , MRINSTRUCTION_TYPE.PickByCount);  // for interQuantile()
		
		// CM Instruction Opcodes
		String2MRInstructionType.put( "cm"  , MRINSTRUCTION_TYPE.CM_N_COV); 
		String2MRInstructionType.put( "cov"  , MRINSTRUCTION_TYPE.CM_N_COV); 
		String2MRInstructionType.put( "mean"  , MRINSTRUCTION_TYPE.CM_N_COV); 
		
		//groupedAgg Instruction Opcodes
		String2MRInstructionType.put( "groupedagg"  , MRINSTRUCTION_TYPE.GroupedAggregate); 
		//String2MRInstructionType.put( "grpcm"  , MRINSTRUCTION_TYPE.GroupedAggregate); 
		
		//rangereindexing
		String2MRInstructionType.put( "rangeReIndex"  , MRINSTRUCTION_TYPE.RangeReIndex);
		String2MRInstructionType.put( "rangeReIndexForLeft"  , MRINSTRUCTION_TYPE.RangeReIndex);
		String2MRInstructionType.put( "zeroOut"  , MRINSTRUCTION_TYPE.ZeroOut);

		//append
		String2MRInstructionType.put( "mappend"  , MRINSTRUCTION_TYPE.Append);
		String2MRInstructionType.put( "rappend"  , MRINSTRUCTION_TYPE.Append);
		String2MRInstructionType.put( "gappend"  , MRINSTRUCTION_TYPE.Append);
		
		//misc
		String2MRInstructionType.put( "rshape", MRINSTRUCTION_TYPE.MatrixReshape);
		
		//partitioning
		String2MRInstructionType.put( "partition", MRINSTRUCTION_TYPE.Partition);
		
		//cumsum
		String2MRInstructionType.put( "ucumack+", MRINSTRUCTION_TYPE.CumsumAggregate);
		String2MRInstructionType.put( "ucumsplit", MRINSTRUCTION_TYPE.CumsumSplit);
		String2MRInstructionType.put( "bcumoffk+", MRINSTRUCTION_TYPE.CumsumOffset);
		
		//dummy (pseudo instructions)
		String2MRInstructionType.put( "sort", MRINSTRUCTION_TYPE.Sort);
		
		String2MRInstructionType.put( "csvwrite", MRINSTRUCTION_TYPE.CSVWrite);
		
		//parameterized builtins
		String2MRInstructionType.put( "replace", MRINSTRUCTION_TYPE.ParameterizedBuiltin);
		
		//remove empty (special type since binary not unary)
		String2MRInstructionType.put( "rmempty", MRINSTRUCTION_TYPE.RemoveEmpty);
	}
	
	
	public static MRInstruction parseSingleInstruction (String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;
		
		MRINSTRUCTION_TYPE mrtype = InstructionUtils.getMRType(str); 
		return MRInstructionParser.parseSingleInstruction(mrtype, str);
	}
	
	public static MRInstruction parseSingleInstruction (MRINSTRUCTION_TYPE mrtype, String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;
		
		switch(mrtype) {
		case Aggregate:
			return (MRInstruction) AggregateInstruction.parseInstruction(str);
			
		case ArithmeticBinary:
			String opcode = InstructionUtils.getOpCode(str);
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
			// extract datatypes of first and second input operands
			String dt1 = parts[1].split(Instruction.DATATYPE_PREFIX)[1].split(Instruction.VALUETYPE_PREFIX)[0];
			String dt2 = parts[2].split(Instruction.DATATYPE_PREFIX)[1].split(Instruction.VALUETYPE_PREFIX)[0];
			if ( dt1.equalsIgnoreCase("SCALAR") || dt2.equalsIgnoreCase("SCALAR") ) {
				return (MRInstruction) ScalarInstruction.parseInstruction(str);
			}
			else {
				if( BinaryM.isOpcode( opcode ) )
					return (MRInstruction) BinaryMInstruction.parseInstruction(str);
				else
					return (MRInstruction) BinaryInstruction.parseInstruction(str);
			}
		case AggregateBinary:
			return (MRInstruction) AggregateBinaryInstruction.parseInstruction(str);
			
		case AggregateUnary:
			return (MRInstruction) AggregateUnaryInstruction.parseInstruction(str);
			
		case Tertiary: 
			return (MRInstruction) TertiaryInstruction.parseInstruction(str);
		
		case Rand:
			return (MRInstruction) RandInstruction.parseInstruction(str);
			
		case Seq:
			return (MRInstruction) SeqInstruction.parseInstruction(str);
			
		case Reblock:
			return (MRInstruction) ReblockInstruction.parseInstruction(str);
		
		case Append:
			return (MRInstruction) AppendInstruction.parseInstruction(str);
			
		case Reorg:
			return (MRInstruction) ReorgInstruction.parseInstruction(str);
			
		case Replicate:
			return (MRInstruction) ReplicateInstruction.parseInstruction(str);
		
		case Unary:
			return (MRInstruction) UnaryInstruction.parseInstruction(str);
			
		case MMTSJ:
			return (MRInstruction) MMTSJMRInstruction.parseInstruction(str);
		
		case MapMultChain:
			return (MRInstruction) MapMultChainInstruction.parseInstruction(str);
		
		case BinUaggChain:
			return (MRInstruction) BinUaggChainInstruction.parseInstruction(str);
			
		case CombineTertiary:
			return (MRInstruction) CombineTertiaryInstruction.parseInstruction(str);
			
		case CombineBinary:
			return (MRInstruction) CombineBinaryInstruction.parseInstruction(str);
			
		case CombineUnary:
			return (MRInstruction) CombineUnaryInstruction.parseInstruction(str);
			
		case PickByCount:
			return (MRInstruction) PickByCountInstruction.parseInstruction(str);
			
		case CM_N_COV:
			return (MRInstruction) CM_N_COVInstruction.parseInstruction(str);
	
		case GroupedAggregate:
			return (MRInstruction) GroupedAggregateInstruction.parseInstruction(str);
		
		case RangeReIndex:
			return (MRInstruction) RangeBasedReIndexInstruction.parseInstruction(str);
		
		case ZeroOut:
			return (MRInstruction) ZeroOutInstruction.parseInstruction(str);
		
		case MatrixReshape:
			return (MRInstruction) MatrixReshapeMRInstruction.parseInstruction(str);	
		
		case Sort: //workaround for dummy MR sort instruction
			return (MRInstruction)UnaryInstruction.parseInstruction(str);
		
		case CSVReblock:
			return (MRInstruction)CSVReblockInstruction.parseInstruction(str);
			
		case CSVWrite:
			return (MRInstruction)CSVWriteInstruction.parseInstruction(str);
			
		case ParameterizedBuiltin:
			return (MRInstruction)ParameterizedBuiltinMRInstruction.parseInstruction(str);
		
		case RemoveEmpty:
			return (MRInstruction)RemoveEmptyMRInstruction.parseInstruction(str);
			
		case Partition:
			return (MRInstruction)DataPartitionMRInstruction.parseInstruction(str);
			
		case CumsumAggregate:
			return (MRInstruction)CumsumAggregateInstruction.parseInstruction(str);
			
		case CumsumSplit:
			return (MRInstruction)CumsumSplitInstruction.parseInstruction(str);
		
		case CumsumOffset:
			return (MRInstruction)CumsumOffsetInstruction.parseInstruction(str);
		
		case INVALID:
		
		default: 
			throw new DMLRuntimeException("Invalid MR Instruction Type: " + mrtype );
		}
	}
	
	public static MRInstruction[] parseMixedInstructions ( String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;
		
		Instruction[] inst = InstructionParser.parseMixedInstructions(str);
		MRInstruction[] mrinst = new MRInstruction[inst.length];
		for ( int i=0; i < inst.length; i++ ) {
			mrinst[i] = (MRInstruction) inst[i];
		}
		
		return mrinst;
	}
	
	// TODO: figure out if we need all the functions below 
	
	//unary operation contains scalar, transform, reorg, aggregate unary
	public static UnaryInstruction[] parseUnaryInstructions(String str) throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		UnaryInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new UnaryInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (UnaryInstruction) UnaryInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
	public static AggregateInstruction[] parseAggregateInstructions(String str) throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		AggregateInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new AggregateInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (AggregateInstruction) AggregateInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
	public static ReblockInstruction[] parseReblockInstructions(String str) throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		ReblockInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new ReblockInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (ReblockInstruction) ReblockInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
	public static CSVReblockInstruction[] parseCSVReblockInstructions(String str) throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		CSVReblockInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new CSVReblockInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (CSVReblockInstruction) CSVReblockInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
	public static CSVWriteInstruction[] parseCSVWriteInstructions(String str) throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		CSVWriteInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new CSVWriteInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (CSVWriteInstruction) CSVWriteInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
	public static AggregateBinaryInstruction[] parseAggregateBinaryInstructions(String str) throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		AggregateBinaryInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new AggregateBinaryInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (AggregateBinaryInstruction) AggregateBinaryInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
	public static DataGenMRInstruction[] parseDataGenInstructions(String str) throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		DataGenMRInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new DataGenMRInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (DataGenMRInstruction) InstructionParser.parseSingleInstruction(strlist[i]);
			}
		}
		return inst;
	}
	
	public static MRInstruction[] parseCombineInstructions(String str) throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		MRInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new MRInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				MRINSTRUCTION_TYPE type = InstructionUtils.getMRType(strlist[i]);
				if(type==MRINSTRUCTION_TYPE.CombineBinary)
					inst[i] = (CombineBinaryInstruction) CombineBinaryInstruction.parseInstruction( strlist[i] );
				else if(type==MRINSTRUCTION_TYPE.CombineTertiary)
					inst[i] = (CombineTertiaryInstruction)CombineTertiaryInstruction.parseInstruction(strlist[i]);
				else
					throw new DMLRuntimeException("unknown combine instruction: "+strlist[i]);
			}
		}
		return inst;
	}
	
	public static CM_N_COVInstruction[] parseCM_N_COVInstructions(String str) throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		CM_N_COVInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new CM_N_COVInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (CM_N_COVInstruction) CM_N_COVInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}

	public static GroupedAggregateInstruction[] parseGroupedAggInstructions(String str) 
	throws DMLUnsupportedOperationException, DMLRuntimeException{
		GroupedAggregateInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new GroupedAggregateInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (GroupedAggregateInstruction) GroupedAggregateInstruction.parseInstruction( strlist[i] );
			}
		}
		return inst;
	}
	
}

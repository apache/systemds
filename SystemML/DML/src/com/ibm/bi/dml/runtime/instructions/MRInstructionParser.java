package com.ibm.bi.dml.runtime.instructions;

import java.util.HashMap;

import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateUnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AppendInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.BinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CM_N_COVInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CombineBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CombineTertiaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CombineUnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.PickByCountInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RandInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReorgInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ScalarInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.TertiaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.UnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction.MRINSTRUCTION_TYPE;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class MRInstructionParser extends InstructionParser {

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

		// AGG_BINARY Instruction Opcodes 
		String2MRInstructionType.put( "cpmm" , MRINSTRUCTION_TYPE.AggregateBinary);
		String2MRInstructionType.put( "rmm"  , MRINSTRUCTION_TYPE.AggregateBinary);

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
		String2MRInstructionType.put( "uacmin"  , MRINSTRUCTION_TYPE.AggregateUnary);
		String2MRInstructionType.put( "rdiagM2V", MRINSTRUCTION_TYPE.AggregateUnary);

		// BINARY Instruction Opcodes 
		String2MRInstructionType.put( "+"    , MRINSTRUCTION_TYPE.Binary);
		String2MRInstructionType.put( "-"    , MRINSTRUCTION_TYPE.Binary);
		String2MRInstructionType.put( "*"    , MRINSTRUCTION_TYPE.Binary);
		String2MRInstructionType.put( "/"    , MRINSTRUCTION_TYPE.Binary);
		String2MRInstructionType.put( "max"  , MRINSTRUCTION_TYPE.Binary);
		String2MRInstructionType.put( "min"  , MRINSTRUCTION_TYPE.Binary);
		String2MRInstructionType.put( ">"    , MRINSTRUCTION_TYPE.Binary);
		String2MRInstructionType.put( ">="   , MRINSTRUCTION_TYPE.Binary);
		String2MRInstructionType.put( "<"    , MRINSTRUCTION_TYPE.Binary);
		String2MRInstructionType.put( "<="   , MRINSTRUCTION_TYPE.Binary);
		String2MRInstructionType.put( "=="   , MRINSTRUCTION_TYPE.Binary);
		String2MRInstructionType.put( "!="   , MRINSTRUCTION_TYPE.Binary);

		// BUILTIN Instruction Opcodes 
		String2MRInstructionType.put( "abs"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "sin"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "cos"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "tan"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "sqrt" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "exp"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "log"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "slog" , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "pow"  , MRINSTRUCTION_TYPE.Unary);
		String2MRInstructionType.put( "round", MRINSTRUCTION_TYPE.Unary);

		// SCALAR Instruction Opcodes 
		String2MRInstructionType.put( "+"    , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "-"    , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "s-r"   , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "*"    , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "/"    , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "so"    , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "^"    , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "max"  , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "min"  , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( ">"    , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( ">="   , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "<"    , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "<="   , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "=="   , MRINSTRUCTION_TYPE.Scalar);
		String2MRInstructionType.put( "!="   , MRINSTRUCTION_TYPE.Scalar);
		// String2InstructionType.put( "sl"    , MRINSTRUCTION_TYPE.Scalar);

		// REORG Instruction Opcodes 
		String2MRInstructionType.put( "r'"      , MRINSTRUCTION_TYPE.Reorg);
		String2MRInstructionType.put( "rdiagV2M", MRINSTRUCTION_TYPE.Reorg);
		
		// RAND Instruction Opcodes 
		String2MRInstructionType.put( "Rand"   , MRINSTRUCTION_TYPE.Rand);
		
		// REBLOCK Instruction Opcodes 
		String2MRInstructionType.put( "rblk"   , MRINSTRUCTION_TYPE.Reblock);
		
		// Tertiary Reorg Instruction Opcodes 
		String2MRInstructionType.put( "ctabletransform", MRINSTRUCTION_TYPE.Tertiary);
		String2MRInstructionType.put( "ctabletransformscalarweight", MRINSTRUCTION_TYPE.Tertiary);
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
			
		case AggregateBinary:
			return (MRInstruction) AggregateBinaryInstruction.parseInstruction(str);
			
		case AggregateUnary:
			return (MRInstruction) AggregateUnaryInstruction.parseInstruction(str);
			
		case Binary: 
			return (MRInstruction) BinaryInstruction.parseInstruction(str);
		
		case Tertiary: 
			return (MRInstruction) TertiaryInstruction.parseInstruction(str);
		
		case Rand:
			return (MRInstruction) RandInstruction.parseInstruction(str);
			
		case Reblock:
			return (MRInstruction) ReblockInstruction.parseInstruction(str);
			
		case Append:
			return (MRInstruction) AppendInstruction.parseInstruction(str);
			
		case Reorg:
			return (MRInstruction) ReorgInstruction.parseInstruction(str);
			
		//case Replicate:
		//	return (MRInstruction) ReplicateInstruction.parseInstruction(str);
		
		case Scalar:
			return (MRInstruction) ScalarInstruction.parseInstruction(str);
			
		case Unary:
			return (MRInstruction) UnaryInstruction.parseInstruction(str);
			
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
	
	public static RandInstruction[] parseRandInstructions(String str) throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		RandInstruction[] inst=null;
		if(str!=null && !str.isEmpty())
		{
			String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
			inst = new RandInstruction[strlist.length];
			
			for(int i=0; i < strlist.length; i++)
			{
				inst[i] = (RandInstruction) RandInstruction.parseInstruction( strlist[i] );
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

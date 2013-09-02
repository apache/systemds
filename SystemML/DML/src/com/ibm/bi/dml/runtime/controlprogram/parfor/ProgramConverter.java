package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.StringTokenizer;
import java.util.Vector;
import java.util.Map.Entry;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.packagesupport.ExternalFunctionInvocationInstruction;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.ExternalFunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ExternalFunctionProgramBlockCP;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.instructions.CPInstructionParser;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructionParser;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FunctionCallCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

/**
 * Static functionalities for 
 *   * creating deep copies of program blocks, instructions, function program blocks
 *   * serializing and parsing of programs, program blocks, functions program blocks
 * 
 * TODO: CV, EL, ELUse program blocks not considered so far (not for BI 2.0 release)
 * TODO: rewrite class to instance-based invocation (grown gradually and now inappropriate design)
 *
 */
public class ProgramConverter 
{
	//use escaped unicodes for separators in order to prevent string conflict
	public static final String NEWLINE           = "\n"; //System.lineSeparator();
	public static final String COMPONENTS_DELIM  = "\u003b"; //";";
	public static final String ELEMENT_DELIM     = "\u002c"; //",";
	public static final String DATA_FIELD_DELIM  = "\u007c"; //"|";
	public static final String KEY_VALUE_DELIM   = "\u003d"; //"=";
	public static final String LEVELIN           = "\u007b"; //"{";
	public static final String LEVELOUT          = "\u007d"; //"}";	
	public static final String EMPTY             = "null";
	public static final String EXT_FUNCTION      = "extfunct";
	
	//public static final String CP_ROOT_THREAD_SEPARATOR = "/";//File.separator;
	public static final String CP_ROOT_THREAD_ID = "_t0";       
	public static final String CP_CHILD_THREAD   = "_t";
		
	public static final String PARFOR_CDATA_BEGIN = "<![CDATA[";
	public static final String PARFOR_CDATA_END = " ]]>";
	
	public static final String PARFOR_PROG_BEGIN = " PROG" + LEVELIN;
	public static final String PARFOR_PROG_END   = LEVELOUT;	
	public static final String PARFORBODY_BEGIN  = PARFOR_CDATA_BEGIN+"PARFORBODY" + LEVELIN;
	public static final String PARFORBODY_END    = LEVELOUT+PARFOR_CDATA_END;
	public static final String PARFOR_VARS_BEGIN = "VARS: ";
	public static final String PARFOR_VARS_END   = "";
	public static final String PARFOR_PBS_BEGIN  = " PBS" + LEVELIN;
	public static final String PARFOR_PBS_END    = LEVELOUT;
	public static final String PARFOR_INST_BEGIN = " INST: ";
	public static final String PARFOR_INST_END   = "";
	public static final String PARFOR_EC_BEGIN   = " EC: ";
	public static final String PARFOR_EC_END     = "";	
	public static final String PARFOR_PB_BEGIN   = " PB" + LEVELIN;
	public static final String PARFOR_PB_END     = LEVELOUT;
	public static final String PARFOR_PB_WHILE   = " WHILE" + LEVELIN;
	public static final String PARFOR_PB_FOR     = " FOR" + LEVELIN;
	public static final String PARFOR_PB_PARFOR  = " PARFOR" + LEVELIN;
	public static final String PARFOR_PB_IF      = " IF" + LEVELIN;
	public static final String PARFOR_PB_FC      = " FC" + LEVELIN;
	public static final String PARFOR_PB_EFC     = " EFC" + LEVELIN;
	
	//exception msgs
	public static final String NOT_SUPPORTED_EXTERNALFUNCTION_PB = "Not supported: ExternalFunctionProgramBlock contains MR instructions. " +
			                                                       "(ExternalFunctionPRogramBlockCP can be used)";
	public static final String NOT_SUPPORTED_MR_INSTRUCTION      = "Not supported: Instructions of type other than CP instructions";
	public static final String NOT_SUPPORTED_MR_PARFOR           = "Not supported: Nested ParFOR REMOTE_MR due to possible deadlocks." +
			                                                       "(LOCAL can be used for innner ParFOR)";
	public static final String NOT_SUPPORTED_PB                  = "Not supported: type of program block";
	public static final String NOT_SUPPORTED_EXECUTION_CONTEXT   = "Parsing of external system execution context not supported yet.";
	
	
	////////////////////////////////
	// CREATION of DEEP COPIES
	////////////////////////////////
	
	/**
	 * Creates a deep copy of the given execution context.
	 * For rt_platform=Hadoop, execution context has a symbol table.
	 * @throws CloneNotSupportedException 
	 */
	public static ExecutionContext createDeepCopyExecutionContext(ExecutionContext ec) 
		throws CloneNotSupportedException 
	{
		ExecutionContext cpec = new ExecutionContext(false);
		cpec.setVariables((LocalVariableMap) ec.getVariables().clone());
	
		return cpec;
	}
	
	/**
	 * This recursively creates a deep copy of program blocks and transparently replaces filenames according to the
	 * specified parallel worker in order to avoid conflicts between parworkers. This happens recursively in order
	 * to support arbitrary control-flow constructs within a parfor. 
	 * 
	 * @param childBlocks
	 * @param pid
	 * @param plain full deep copy without id replacement 
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ArrayList<ProgramBlock> rcreateDeepCopyProgramBlocks(ArrayList<ProgramBlock> childBlocks, long pid, int IDPrefix, HashSet<String> fnStack, boolean plain) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		ArrayList<ProgramBlock> tmp = new ArrayList<ProgramBlock>();
		
		for( ProgramBlock pb : childBlocks )
		{
			Program prog = pb.getProgram();
			ProgramBlock tmpPB = null;
			
			if( pb instanceof WhileProgramBlock || pb instanceof ForProgramBlock || pb instanceof ParForProgramBlock || pb instanceof IfProgramBlock )
			{
				if( pb instanceof WhileProgramBlock ) 
				{
					tmpPB = createDeepCopyWhileProgramBlock((WhileProgramBlock) pb, pid, IDPrefix, prog, fnStack, plain);
				}
				else if ( pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock) )
				{
					tmpPB = createDeepCopyForProgramBlock((ForProgramBlock) pb, pid, IDPrefix, prog, fnStack, plain );
				}
				else if ( pb instanceof ParForProgramBlock )
				{
					ParForProgramBlock pfpb = (ParForProgramBlock) pb;
					if( ParForProgramBlock.ALLOW_NESTED_PARALLELISM )
						tmpPB = createDeepCopyParForProgramBlock(pfpb, pid, IDPrefix, prog, fnStack, plain);
					else 
						tmpPB = createDeepCopyForProgramBlock((ForProgramBlock) pb, pid, IDPrefix, prog, fnStack, plain);
				}				
				else if ( pb instanceof IfProgramBlock )
				{
					tmpPB = createDeepCopyIfProgramBlock((IfProgramBlock) pb, pid, IDPrefix, prog, fnStack, plain);
				}
			}	
			else
			{
				tmpPB = new ProgramBlock(prog); // general case use for most PBs
				
				//for recompile in the master node JVM
				tmpPB.setStatementBlock(createStatementBlockCopy(pb.getStatementBlock(), pid, plain)); 
				//tmpPB.setStatementBlock(pb.getStatementBlock()); 
				tmpPB.setThreadID(pid);
			}

			//copy instructions
			tmpPB.setInstructions( createDeepCopyInstructionSet(pb.getInstructions(), pid, IDPrefix, prog, fnStack, plain, true) );
			
			//copy symbol table
			//tmpPB.setVariables( pb.getVariables() ); //implicit cloning			
			
			tmp.add(tmpPB);
		}
		
		return tmp;
	}
	
	/**
	 * 
	 * @param wpb
	 * @param pid
	 * @param IDPrefix
	 * @param prog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static WhileProgramBlock createDeepCopyWhileProgramBlock(WhileProgramBlock wpb, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, boolean plain) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ArrayList<Instruction> predinst = createDeepCopyInstructionSet(wpb.getPredicate(), pid, IDPrefix, prog, fnStack, plain, true);
		WhileProgramBlock tmpPB = new WhileProgramBlock(prog, predinst);
		tmpPB.setStatementBlock( wpb.getStatementBlock() );
		tmpPB.setThreadID(pid);
		
		tmpPB.setExitInstructions2( createDeepCopyInstructionSet(wpb.getExitInstructions(), pid, IDPrefix, prog, fnStack, plain, true));
		tmpPB.setChildBlocks(rcreateDeepCopyProgramBlocks(wpb.getChildBlocks(), pid, IDPrefix, fnStack, plain));
		
		return tmpPB;
	}
	
	/**
	 * 
	 * @param ipb
	 * @param pid
	 * @param IDPrefix
	 * @param prog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static IfProgramBlock createDeepCopyIfProgramBlock(IfProgramBlock ipb, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, boolean plain) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		ArrayList<Instruction> predinst = createDeepCopyInstructionSet(ipb.getPredicate(), pid, IDPrefix, prog, fnStack, plain, true);
		IfProgramBlock tmpPB = new IfProgramBlock(prog, predinst);
		tmpPB.setStatementBlock( ipb.getStatementBlock() );
		tmpPB.setThreadID(pid);
		
		tmpPB.setExitInstructions2( createDeepCopyInstructionSet(ipb.getExitInstructions(), pid, IDPrefix, prog, fnStack, plain, true));
		tmpPB.setChildBlocksIfBody(rcreateDeepCopyProgramBlocks(ipb.getChildBlocksIfBody(), pid, IDPrefix, fnStack, plain));
		tmpPB.setChildBlocksElseBody(rcreateDeepCopyProgramBlocks(ipb.getChildBlocksElseBody(), pid, IDPrefix, fnStack, plain));
		
		return tmpPB;
	}
	
	/**
	 * 
	 * @param fpb
	 * @param pid
	 * @param IDPrefix
	 * @param prog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ForProgramBlock createDeepCopyForProgramBlock(ForProgramBlock fpb, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, boolean plain) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ForProgramBlock tmpPB = new ForProgramBlock(prog,fpb.getIterablePredicateVars());
		tmpPB.setStatementBlock(fpb.getStatementBlock());
		tmpPB.setThreadID(pid);
		
		tmpPB.setFromInstructions( createDeepCopyInstructionSet(fpb.getFromInstructions(), pid, IDPrefix, prog, fnStack, plain, true) );
		tmpPB.setToInstructions( createDeepCopyInstructionSet(fpb.getToInstructions(), pid, IDPrefix, prog, fnStack, plain, true) );
		tmpPB.setIncrementInstructions( createDeepCopyInstructionSet(fpb.getIncrementInstructions(), pid, IDPrefix, prog, fnStack, plain, true) );
		tmpPB.setExitInstructions( createDeepCopyInstructionSet(fpb.getExitInstructions(), pid, IDPrefix, prog, fnStack, plain, true) );
		tmpPB.setChildBlocks( rcreateDeepCopyProgramBlocks(fpb.getChildBlocks(), pid, IDPrefix, fnStack, plain) );
		
		return tmpPB;
	}
	
	/**
	 * 
	 * @param fpb
	 * @param prog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public static ForProgramBlock createShallowCopyForProgramBlock(ForProgramBlock fpb, Program prog ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ForProgramBlock tmpPB = new ForProgramBlock(prog,fpb.getIterablePredicateVars());
		
		tmpPB.setFromInstructions( fpb.getFromInstructions() );
		tmpPB.setToInstructions( fpb.getToInstructions() );
		tmpPB.setIncrementInstructions( fpb.getIncrementInstructions() );
		tmpPB.setExitInstructions( fpb.getExitInstructions() );
		tmpPB.setChildBlocks( fpb.getChildBlocks() );
		
		return tmpPB;
	}
	
	/**
	 * 
	 * @param pfpb
	 * @param prog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public static ForProgramBlock createShallowCopyParForProgramBlock(ParForProgramBlock pfpb, Program prog ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ParForProgramBlock tmpPB = new ParForProgramBlock(prog,pfpb.getIterablePredicateVars(),pfpb.getParForParams());
		
		tmpPB.setStatementBlock( pfpb.getStatementBlock() );
		tmpPB.setResultVariables( pfpb.getResultVariables() );
		
		tmpPB.setFromInstructions( pfpb.getFromInstructions() );
		tmpPB.setToInstructions( pfpb.getToInstructions() );
		tmpPB.setIncrementInstructions( pfpb.getIncrementInstructions() );
		tmpPB.setExitInstructions( pfpb.getExitInstructions() );
		tmpPB.setChildBlocks( pfpb.getChildBlocks() );
		
		return tmpPB;
	}
	
	/**
	 * 
	 * @param pfpb
	 * @param pid
	 * @param IDPrefix
	 * @param prog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ParForProgramBlock createDeepCopyParForProgramBlock(ParForProgramBlock pfpb, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, boolean plain) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ParForProgramBlock tmpPB = null;
		
		if( IDPrefix == -1 ) //still on master node
			tmpPB = new ParForProgramBlock(prog,pfpb.getIterablePredicateVars(),pfpb.getParForParams()); 
		else //child of remote ParWorker at any level
			tmpPB = new ParForProgramBlock(IDPrefix, prog, pfpb.getIterablePredicateVars(),pfpb.getParForParams());
		
		tmpPB.setStatementBlock( pfpb.getStatementBlock() );
		tmpPB.setThreadID(pid);
		
		tmpPB.disableOptimization(); //already done in top-level parfor
		tmpPB.setResultVariables( pfpb.getResultVariables() );
		
		tmpPB.setFromInstructions( createDeepCopyInstructionSet(pfpb.getFromInstructions(), pid, IDPrefix, prog, fnStack, plain, true) );
		tmpPB.setToInstructions( createDeepCopyInstructionSet(pfpb.getToInstructions(), pid, IDPrefix, prog, fnStack, plain, true) );
		tmpPB.setIncrementInstructions( createDeepCopyInstructionSet(pfpb.getIncrementInstructions(), pid, IDPrefix, prog, fnStack, plain, true) );
		tmpPB.setExitInstructions( createDeepCopyInstructionSet(pfpb.getExitInstructions(), pid, IDPrefix, prog, fnStack, plain, true) );

		//NOTE: Normally, no recursive copy because (1) copied on each execution in this PB anyway 
		//and (2) leave placeholders as they are. However, if plain, an explicit deep copy is requested.
		if( plain )
			tmpPB.setChildBlocks( rcreateDeepCopyProgramBlocks(pfpb.getChildBlocks(), pid, IDPrefix, fnStack, plain) ); 
		else
			tmpPB.setChildBlocks( pfpb.getChildBlocks() );
		
		return tmpPB;
	}
	
	/**
	 * This creates a deep copy of a function program block. The central reference to singletons of function program blocks
	 * poses the need for explicit copies in order to prevent conflicting writes of temporary variables (see ExternalFunctionProgramBlock.
	 * 
	 * @param oldName
	 * @param pid
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	public static void createDeepCopyFunctionProgramBlock(String namespace, String oldName, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, boolean plain) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		FunctionProgramBlock fpb = prog.getFunctionProgramBlock(namespace, oldName);
		String fnameNew = (plain)? oldName :(oldName+CP_CHILD_THREAD+pid); 
		String fnameNewKey = namespace+Program.KEY_DELIM+fnameNew;

		if( prog.getFunctionProgramBlocks().containsKey(fnameNewKey) )
			return; //prevent redundant deep copy if already existent
		
		if( fpb == null )
		{
			throw new DMLRuntimeException("Unable to create a deep copy of the FunctionProgramBlock "+oldName+" because it does not exist.");
		}
		
		//create deep copy
		FunctionProgramBlock copy = null;
		Vector<DataIdentifier> tmp1 = new Vector<DataIdentifier>(); 
		Vector<DataIdentifier> tmp2 = new Vector<DataIdentifier>(); 
		if( fpb.getInputParams()!= null )
			tmp1.addAll(fpb.getInputParams());
		if( fpb.getOutputParams()!= null )
			tmp2.addAll(fpb.getOutputParams());
		
		if( fpb instanceof ExternalFunctionProgramBlockCP )
		{
			ExternalFunctionProgramBlockCP efpb = (ExternalFunctionProgramBlockCP) fpb;
			HashMap<String,String> tmp3 = efpb.getOtherParams();		
			if( IDPrefix!=-1 )
				copy = new ExternalFunctionProgramBlockCP(prog,tmp1,tmp2,tmp3,efpb.getBaseDir().replaceAll(CP_CHILD_THREAD+IDPrefix, CP_CHILD_THREAD+pid));
			else	
				copy = new ExternalFunctionProgramBlockCP(prog,tmp1,tmp2,tmp3,efpb.getBaseDir().replaceAll(CP_ROOT_THREAD_ID, CP_CHILD_THREAD+pid));
		}
		else if( fpb instanceof ExternalFunctionProgramBlock )
		{
			ExternalFunctionProgramBlock efpb = (ExternalFunctionProgramBlock) fpb;
			HashMap<String,String> tmp3 = efpb.getOtherParams();
			if( IDPrefix!=-1 )
				copy = new ExternalFunctionProgramBlock(prog,tmp1,tmp2,tmp3,efpb.getBaseDir().replaceAll(CP_CHILD_THREAD+IDPrefix, CP_CHILD_THREAD+pid));
			else	
				copy = new ExternalFunctionProgramBlock(prog,tmp1,tmp2,tmp3,efpb.getBaseDir().replaceAll(CP_ROOT_THREAD_ID, CP_CHILD_THREAD+pid));
		}
		else
		{
			if( !fnStack.contains(fnameNewKey) ) {
				fnStack.add(fnameNewKey);
				copy = new FunctionProgramBlock(prog, tmp1, tmp2);
				copy.setChildBlocks( rcreateDeepCopyProgramBlocks(fpb.getChildBlocks(), pid, IDPrefix, fnStack, plain) );
				fnStack.remove(fnameNewKey);
			}
			else //stop deep copy for recursive function calls
				copy = fpb;
		}
		
		//copy.setVariables( (LocalVariableMap) fpb.getVariables() ); //implicit cloning
		//note: instructions not used by function program block
		
		//put 
		prog.addFunctionProgramBlock(namespace, fnameNew, copy);
	}
	
	/**
	 * 
	 * @param fpb
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public static FunctionProgramBlock createDeepCopyFunctionProgramBlock(FunctionProgramBlock fpb, HashSet<String> fnStack) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		if( fpb == null )
			throw new DMLRuntimeException("Unable to create a deep copy of a non-existing FunctionProgramBlock.");
	
		//create deep copy
		FunctionProgramBlock copy = null;
		Vector<DataIdentifier> tmp1 = new Vector<DataIdentifier>(); 
		Vector<DataIdentifier> tmp2 = new Vector<DataIdentifier>(); 
		if( fpb.getInputParams()!= null )
			tmp1.addAll(fpb.getInputParams());
		if( fpb.getOutputParams()!= null )
			tmp2.addAll(fpb.getOutputParams());
		
		copy = new FunctionProgramBlock(fpb.getProgram(), tmp1, tmp2);
		copy.setChildBlocks( rcreateDeepCopyProgramBlocks(fpb.getChildBlocks(), 0, -1, fnStack, true) );
		//copy.setVariables( (LocalVariableMap) fpb.getVariables() ); //implicit cloning
		//note: instructions not used by function program block
	
		return copy;
	}

	
	/**
	 * Creates a deep copy of an array of instructions and replaces the placeholders of parworker
	 * IDs with the concrete IDs of this parfor instance. This is a helper method uses for generating
	 * deep copies of program blocks.
	 * 
	 * @param instSet
	 * @param pid
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	@SuppressWarnings("unchecked")
	public static ArrayList<Instruction> createDeepCopyInstructionSet(ArrayList<Instruction> instSet, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, boolean plain, boolean cpFunctions) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ArrayList<Instruction> tmp = (ArrayList<Instruction>) instSet.clone();
		for( int i=0; i<instSet.size(); i++ )
		{
			Instruction inst1 = instSet.get(i);
			if( inst1 instanceof FunctionCallCPInstruction && cpFunctions )
			{
				FunctionCallCPInstruction finst1 = (FunctionCallCPInstruction) inst1;
				createDeepCopyFunctionProgramBlock( finst1.getNamespace(),
						                            finst1.getFunctionName(), 
						                            pid, IDPrefix, prog, fnStack, plain );
			}
			
			Instruction inst2 = cloneInstruction( inst1, pid, plain, cpFunctions ); 
			tmp.set(i, inst2);
		}
		
		return tmp;
	}
	
	
	/**
	 * 
	 * @param pid
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	@SuppressWarnings("unchecked")
	public static Instruction cloneInstruction( Instruction oInst, long pid, boolean plain, boolean cpFunctions ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		Instruction inst = null;

		String tmpString = oInst.toString().replaceAll(ProgramConverter.CP_ROOT_THREAD_ID, 
				                                       ProgramConverter.CP_CHILD_THREAD+pid);
		
		Class<Instruction> cla = null;
		Method parse = null;
		try
		{
			cla = (Class) oInst.getClass();
			parse = cla.getMethod("parseInstruction", String.class);
		}
		catch(Exception ex){}
		
		try
		{
			if( parse != null )
			{
				if( oInst instanceof FunctionCallCPInstruction && cpFunctions )
				{
					FunctionCallCPInstruction tmp = (FunctionCallCPInstruction) oInst;
					if( !plain )
						tmpString = tmp.toString().replaceAll(tmp.getFunctionName(), tmp.getFunctionName() + CP_CHILD_THREAD+pid);  
					//otherwise: preserve functionname
				}
				
				inst = (Instruction) parse.invoke(null, tmpString);
			}
			else if( oInst instanceof CPInstruction )
			{
				CPInstruction tmp = (CPInstruction) oInst;
				inst = CPInstructionParser.parseSingleInstruction(tmp.getCPInstructionType(), tmpString);
			}
			else if( oInst instanceof MRInstruction )
			{
				MRInstruction tmp = (MRInstruction) oInst;
				inst = MRInstructionParser.parseSingleInstruction(tmp.getMRInstructionType(), tmpString);
			}
			else if( oInst instanceof MRJobInstruction )
			{
				MRJobInstruction tmp = (MRJobInstruction)oInst;
				inst = new MRJobInstruction(tmp, ProgramConverter.CP_ROOT_THREAD_ID, ProgramConverter.CP_CHILD_THREAD+pid);
				
				/*MRJobInstruction tmpNew = new MRJobInstruction(tmp.getJobType());
				Field[] fields = cla.getDeclaredFields();
				for( Field f : fields )
				{
					f.setAccessible(true);
					if(!Modifier.isStatic(f.getModifiers()))
						f.set(tmpNew, f.get(tmp));
				}
				
				String[] in = tmp.getIv_inputs().clone();
				String[] out = tmp.getIv_outputs().clone();
				String rand = tmp.getIv_randInstructions();
				if(in!=null)
					for( int j=0;j<in.length; j++)
						if( in[j]!=null )
							in[j]=in[j].replaceAll(ProgramConverter.CP_ROOT_THREAD_ID, ProgramConverter.CP_CHILD_THREAD+pid);
				if(out!=null)
					for( int j=0;j<out.length; j++)
						if( out[j]!=null )
							out[j]=out[j].replaceAll(ProgramConverter.CP_ROOT_THREAD_ID, ProgramConverter.CP_CHILD_THREAD+pid);
				rand = rand.replaceAll(ProgramConverter.CP_ROOT_THREAD_ID, ProgramConverter.CP_CHILD_THREAD+pid);
				tmpNew.setIv_inputs(in);
				tmpNew.setIv_outputs(out);	
				tmpNew.setRandInstructions(rand);
				
				inst = tmpNew;*/
			}
			else
				throw new DMLUnsupportedOperationException("Unable to clone instruction of type "+cla.toString());
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return inst;
	}
	
	/**
	 * 
	 * @param sb
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static StatementBlock createStatementBlockCopy( StatementBlock sb, long pid, boolean plain ) 
		throws DMLRuntimeException
	{
		StatementBlock ret = null;
		
		try
		{
			if( OptimizerUtils.ALLOW_PARALLEL_DYN_RECOMPILATION 
				&& DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID	
				&& sb != null 
				&& Recompiler.requiresRecompilation( sb.get_hops() )  )
			{
				ret = new StatementBlock();
				ArrayList<Hops> hops = Recompiler.deepCopyHopsDag( sb.get_hops() );
				if( !plain )
					Recompiler.updateFunctionNames( hops, pid );
				ret.set_hops( hops );
			}
			else
			{
				ret = sb;
			}
		}
		catch( Exception ex )
		{
			throw new DMLRuntimeException( ex );
		}
		
		return ret;
	}
	
	
	////////////////////////////////
	// SERIALIZATION 
	////////////////////////////////	
	
	/**
	 * @param body
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	public static String serializeParForBody( ParForBody body ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ArrayList<ProgramBlock> pbs = body.getChildBlocks();
		ArrayList<String> rVnames   = body.getResultVarNames();
		ExecutionContext ec         = body.getEc();
		
		if( pbs.size()==0 )
			return PARFORBODY_BEGIN + PARFORBODY_END;
		
		Program prog = pbs.get( 0 ).getProgram();
		
		StringBuilder sb = new StringBuilder();
		sb.append( PARFORBODY_BEGIN );
		sb.append( NEWLINE );
		
		//handle DMLScript UUID (propagate original uuid for writing to scratch space)
		sb.append( DMLScript.getUUID() );
		sb.append( COMPONENTS_DELIM );
		sb.append( NEWLINE );		
		
		//handle DML config
		sb.append( ConfigurationManager.getConfig().serializeDMLConfig() );
		sb.append( COMPONENTS_DELIM );
		sb.append( NEWLINE );
		
		//handle program
		sb.append( PARFOR_PROG_BEGIN );
		sb.append( NEWLINE );
		sb.append( serializeProgram(prog, pbs) );
		sb.append( PARFOR_PROG_END );
		sb.append( NEWLINE );
		sb.append( COMPONENTS_DELIM );
		sb.append( NEWLINE );
		
		//handle result variable names
		sb.append( serializeStringArrayList( rVnames ) );
		sb.append( COMPONENTS_DELIM );
		
		//handle execution context
		//note: this includes also the symbol table (serialize only the top-level variable map,
		//      (symbol tables for nested/child blocks are created at parse time, on the remote side)
		sb.append( PARFOR_EC_BEGIN );
		sb.append( serializeExecutionContext(ec) );
		sb.append( PARFOR_EC_END );
		sb.append( NEWLINE );
		sb.append( COMPONENTS_DELIM );
		sb.append( NEWLINE );
		
		//handle program blocks -- ONLY instructions, not variables.
		sb.append( PARFOR_PBS_BEGIN );
		sb.append( NEWLINE );
		sb.append( rSerializeProgramBlocks(pbs) );
		sb.append( PARFOR_PBS_END );
		sb.append( NEWLINE );
		
		sb.append( PARFORBODY_END );
		
		return sb.toString();		
	}
	
	/**
	 * 
	 * @param prog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static String serializeProgram( Program prog, ArrayList<ProgramBlock> pbs ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//note program contains variables, programblocks and function program blocks 
		//but in order to avoid redundancy, we only serialize function program blocks
		
		HashMap<String, FunctionProgramBlock> fpb = prog.getFunctionProgramBlocks();		
		HashSet<String> cand = new HashSet<String>();
		rFindSerializationCandidates(pbs, cand);
		
		return rSerializeFunctionProgramBlocks( fpb, cand );
	}
	
	/**
	 * 
	 * @param pbs
	 * @param cand
	 * @throws DMLRuntimeException
	 */
	public static void rFindSerializationCandidates( ArrayList<ProgramBlock> pbs, HashSet<String> cand ) 
		throws DMLRuntimeException
	{
		for( ProgramBlock pb : pbs )
		{
			if( pb instanceof WhileProgramBlock )
			{
				WhileProgramBlock wpb = (WhileProgramBlock) pb;
				rFindSerializationCandidates(wpb.getChildBlocks(), cand );			
			}
			else if ( pb instanceof ForProgramBlock || pb instanceof ParForProgramBlock )
			{
				ForProgramBlock fpb = (ForProgramBlock) pb; 
				rFindSerializationCandidates(fpb.getChildBlocks(), cand);
			}				
			else if ( pb instanceof IfProgramBlock )
			{
				IfProgramBlock ipb = (IfProgramBlock) pb;
				rFindSerializationCandidates(ipb.getChildBlocksIfBody(), cand);
				if( ipb.getChildBlocksElseBody() != null )
					rFindSerializationCandidates(ipb.getChildBlocksElseBody(), cand);
			}
			else //all generic program blocks
			{
				for( Instruction inst : pb.getInstructions() )
					if( inst instanceof FunctionCallCPInstruction )
					{
						FunctionCallCPInstruction fci = (FunctionCallCPInstruction) inst;
						String fkey = fci.getNamespace() + Program.KEY_DELIM + fci.getFunctionName();
						if( !cand.contains(fkey) ) //memoization for multiple calls, recursion
						{
							cand.add( fkey ); //add to candidates
							
							//investigate chains of function calls
							FunctionProgramBlock fpb = pb.getProgram().getFunctionProgramBlock(fci.getNamespace(), fci.getFunctionName());
							rFindSerializationCandidates(fpb.getChildBlocks(), cand);
						}
					}
			}
		}
	}
	
	
	/**
	 * 
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static String serializeVariables (LocalVariableMap vars) 
		throws DMLRuntimeException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( PARFOR_VARS_BEGIN );
		sb.append( vars.serialize() );
		sb.append( PARFOR_VARS_END );
		
		return sb.toString();
	}
	
	/**
	 * 
	 * @param key
	 * @param dat
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static String serializeDataObject(String key, Data dat) 
		throws DMLRuntimeException
	{
		// SCHEMA: <name>|<datatype>|<valuetype>|value
		// (scalars are serialize by value, matrices by filename)
		
		StringBuilder sb = new StringBuilder();
	
		//prepare data for serialization
		String name = key;
		DataType datatype = dat.getDataType();
		ValueType valuetype = dat.getValueType();
		String value = null;
		String[] matrixMetaData = null; 
		
		switch( datatype )
		{
			case SCALAR:
				ScalarObject so = (ScalarObject) dat;
				//name = so.getName();
				value = so.getStringValue();
				break;
			case MATRIX:
				MatrixObject mo = (MatrixObject) dat;
				MatrixFormatMetaData md = (MatrixFormatMetaData) dat.getMetaData();
				MatrixCharacteristics mc = md.getMatrixCharacteristics();
				value = mo.getFileName();
				PDataPartitionFormat partFormat = (mo.getPartitionFormat()!=null) ? mo.getPartitionFormat() : PDataPartitionFormat.NONE;
				matrixMetaData = new String[8];
				matrixMetaData[0] = String.valueOf( mc.get_rows() );
				matrixMetaData[1] = String.valueOf( mc.get_cols() );
				matrixMetaData[2] = String.valueOf( mc.get_rows_per_block() );
				matrixMetaData[3] = String.valueOf( mc.get_cols_per_block() );
				matrixMetaData[4] = String.valueOf( mc.getNonZeros() );
				matrixMetaData[5] = InputInfo.inputInfoToString( md.getInputInfo() );
				matrixMetaData[6] = OutputInfo.outputInfoToString( md.getOutputInfo() );
				matrixMetaData[7] = String.valueOf( partFormat );
				break;
			default:
				throw new DMLRuntimeException("Unable to serialize datatype "+datatype);
		}
		
		//serialize data
		sb.append(name);
		sb.append(DATA_FIELD_DELIM);
		sb.append(datatype);
		sb.append(DATA_FIELD_DELIM);
		sb.append(valuetype);
		sb.append(DATA_FIELD_DELIM);
		sb.append(value);		
		if( matrixMetaData != null )
			for( int i=0; i<matrixMetaData.length; i++ )
			{
				sb.append(DATA_FIELD_DELIM);
				sb.append(matrixMetaData[i]);
			}
		
		return sb.toString();
	}
	
	/**
	 * 
	 * @param ec
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static String serializeExecutionContext( ExecutionContext ec ) 
		throws DMLRuntimeException
	{
		String ret = null;
		
		if( ec != null )
		{
			LocalVariableMap vars = ec.getVariables();
			ret = serializeVariables( vars );	
		}
		else
		{
			ret = EMPTY;
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param inst
	 * @return
	 * @throws DMLRuntimeException
	 */
	@SuppressWarnings("all")
	public static String serializeInstructions( ArrayList<Instruction> inst ) 
		throws DMLRuntimeException
	{	
		StringBuilder sb = new StringBuilder();
		int count = 0;
		for( Instruction linst : inst )
		{
			//check that only cp instruction are transmitted 
			if( !(   linst instanceof CPInstruction 
				  || linst instanceof ExternalFunctionInvocationInstruction ) )
			{
				throw new DMLRuntimeException( NOT_SUPPORTED_MR_INSTRUCTION + " " +linst.getClass().getName() );
			}
			
			if( count > 0 )
				sb.append( ELEMENT_DELIM );
			
			sb.append( checkAndReplaceLiterals( linst.toString() ) );
			count++;
		}
		
		return sb.toString();	
	}
	
	/**
	 * Replacement of internal delimiters occurring in literals of instructions
	 * in order to ensure robustness of serialization and parsing.
	 * (e.g. print( "a,b" ) would break the parsing of instruction that internally
	 * are separated with a "," )
	 * 
	 * @param instStr
	 * @return
	 */
	public static String checkAndReplaceLiterals( String instStr )
	{
		String tmp = instStr;

		//1) check own delimiters
		if( tmp.contains(COMPONENTS_DELIM) )
			tmp = tmp.replaceAll(COMPONENTS_DELIM, ".");
		
		if( tmp.contains(ELEMENT_DELIM) )
			tmp = tmp.replaceAll(ELEMENT_DELIM, ".");

		if( tmp.contains(LEVELIN) )
			tmp = tmp.replaceAll(LEVELIN, ".");

		if( tmp.contains(LEVELOUT) )
			tmp = tmp.replaceAll(LEVELOUT, ".");
		
		//NOTE: DATA_FIELD_DELIM and KEY_VALUE_DELIM not required
		//because those literals cannot occur in critical places.
		
		//2) check end tag of CDATA
		if( tmp.contains(PARFOR_CDATA_END) )
			tmp = tmp.replaceAll(PARFOR_CDATA_END, ".");		
		
		return tmp;
	}
	
	
	/**
	 * 
	 * @param vars
	 * @return
	 */
	public static String serializeStringHashMap( HashMap<String,String> vars)
	{
		StringBuilder sb = new StringBuilder();
		int count=0;
		for( Entry<String,String> e : vars.entrySet() )
		{
			if(count>0)
				sb.append( ELEMENT_DELIM );
			sb.append( e.getKey() );
			sb.append( KEY_VALUE_DELIM );
			sb.append( e.getValue() );
			count++;
		}
		return sb.toString();
	}
	
	/**
	 * 
	 * @param vars
	 * @return
	 */
	public static String serializeStringHashSet( HashSet<String> set)
	{
		StringBuilder sb = new StringBuilder();
		int count=0;
		for( String s : set )
		{
			if(count>0)
				sb.append( ELEMENT_DELIM );
			sb.append( s );
			count++;
		}
		return sb.toString();
	}
	
	
	/**
	 * 
	 * @param vars
	 * @return
	 */
	public static String serializeStringArrayList( ArrayList<String> vars)
	{
		StringBuilder sb = new StringBuilder();
		int count=0;
		for( String s : vars )
		{
			if(count>0)
				sb.append( ELEMENT_DELIM );
			sb.append( s );
			count++;
		}
		return sb.toString();
	}
	
	/**
	 * 
	 * @param vars
	 * @return
	 */
	public static String serializeStringArray( String[] vars)
	{
		StringBuilder sb = new StringBuilder();
		int count=0;
		for( String s : vars )
		{
			if(count>0)
				sb.append( ELEMENT_DELIM );
			if( s != null )
				sb.append( s );
			else
				sb.append( "null" );
			
			count++;
		}
		return sb.toString();
	}

	/**
	 * 
	 * @param var
	 * @return
	 */
	public static String serializeDataIdentifiers( ArrayList<DataIdentifier> var)
	{
		StringBuilder sb = new StringBuilder();
		int count=0;
		for( DataIdentifier dat : var )
		{
			if(count>0)
				sb.append( ELEMENT_DELIM );
			sb.append( serializeDataIdentifier(dat) );
			count++;
		}
		return sb.toString();
	}
	
	/**
	 * 
	 * @param dat
	 * @return
	 */
	public static String serializeDataIdentifier( DataIdentifier dat )
	{
		// SCHEMA: <name>|<datatype>|<valuetype>
		
		StringBuilder sb = new StringBuilder();
		sb.append(dat.getName());
		sb.append(DATA_FIELD_DELIM);
		sb.append(dat.getDataType());
		sb.append(DATA_FIELD_DELIM);
		sb.append(dat.getValueType());
		
		return sb.toString();
	}
	
	/**
	 * 
	 * @param pbs
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static String rSerializeFunctionProgramBlocks(HashMap<String,FunctionProgramBlock> pbs, HashSet<String> cand) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		StringBuilder sb = new StringBuilder();
		
		int count = 0;
		for( Entry<String,FunctionProgramBlock> pb : pbs.entrySet() )
		{
			if( !cand.contains(pb.getKey()) ) //skip function not included in the parfor body
				continue;
				
			if( count>0 )
			{
			   sb.append( ELEMENT_DELIM );
			   sb.append( NEWLINE );
			}
			sb.append( pb.getKey() );
			sb.append( KEY_VALUE_DELIM );
			sb.append( rSerializeProgramBlock( pb.getValue() ) );
			
			count++;
		}
		sb.append(NEWLINE);
		return sb.toString();
	}
	
	/**
	 * 
	 * @param pbs
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static String rSerializeProgramBlocks(ArrayList<ProgramBlock> pbs) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		StringBuilder sb = new StringBuilder();
		int count = 0;
		for( ProgramBlock pb : pbs )
		{
			if( count>0 )
			{
			   sb.append( ELEMENT_DELIM );
			   sb.append(NEWLINE);
			}
			sb.append( rSerializeProgramBlock(pb) );
			count++;
		}

		return sb.toString();
	}
	
	/**
	 * 
	 * @param pb
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static String rSerializeProgramBlock( ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		StringBuilder sb = new StringBuilder();
		
		//handle header
		if( pb instanceof WhileProgramBlock ) 
			sb.append( PARFOR_PB_WHILE );
		else if ( pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock) )
			sb.append( PARFOR_PB_FOR );
		else if ( pb instanceof ParForProgramBlock )
			sb.append( PARFOR_PB_PARFOR );
		else if ( pb instanceof IfProgramBlock )
			sb.append( PARFOR_PB_IF );
		else if ( pb instanceof FunctionProgramBlock && !(pb instanceof ExternalFunctionProgramBlock) )
			sb.append( PARFOR_PB_FC );
		else if ( pb instanceof ExternalFunctionProgramBlock )
			sb.append( PARFOR_PB_EFC );
		else //all generic program blocks
			sb.append( PARFOR_PB_BEGIN );
		
		//handle variables (not required only on top level)
		/*sb.append( PARFOR_VARS_BEGIN );
		sb.append( serializeVariables( ) ); 
		sb.append( PARFOR_VARS_END );
		sb.append( COMPONENTS_DELIM );
		*/
		
		//handle body
		if( pb instanceof WhileProgramBlock )
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( wpb.getPredicate() ) );
			sb.append( PARFOR_INST_END );
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( wpb.getExitInstructions() ) );
			sb.append( PARFOR_INST_END );
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_PBS_BEGIN );
			sb.append( rSerializeProgramBlocks( wpb.getChildBlocks()) );
			sb.append( PARFOR_PBS_END );
		}
		else if ( pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock ) )
		{
			ForProgramBlock fpb = (ForProgramBlock) pb; 
			sb.append( serializeStringArray(fpb.getIterablePredicateVars()) );
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( fpb.getFromInstructions() ) );
			sb.append( PARFOR_INST_END );	
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( fpb.getToInstructions() ) );
			sb.append( PARFOR_INST_END );	
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( fpb.getIncrementInstructions() ) );
			sb.append( PARFOR_INST_END );
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( fpb.getExitInstructions() ) );
			sb.append( PARFOR_INST_END );
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_PBS_BEGIN );
			sb.append( rSerializeProgramBlocks( fpb.getChildBlocks()) );
			sb.append( PARFOR_PBS_END );
		}
		else if ( pb instanceof ParForProgramBlock )
		{	
			ParForProgramBlock pfpb = (ParForProgramBlock) pb; 
			
			//check for nested remote ParFOR
			if( PExecMode.valueOf( pfpb.getParForParams().get( ParForStatementBlock.EXEC_MODE )) == PExecMode.REMOTE_MR )
				throw new DMLUnsupportedOperationException( NOT_SUPPORTED_MR_PARFOR );
			
			sb.append( serializeStringArray(pfpb.getIterablePredicateVars()) );
			sb.append( COMPONENTS_DELIM );
			sb.append( serializeStringArrayList( pfpb.getResultVariables()) );
			sb.append( COMPONENTS_DELIM );
			sb.append( serializeStringHashMap( pfpb.getParForParams()) ); //parameters of nested parfor
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( pfpb.getFromInstructions() ) );
			sb.append( PARFOR_INST_END );	
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( pfpb.getToInstructions() ) );
			sb.append( PARFOR_INST_END );	
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( pfpb.getIncrementInstructions() ) );
			sb.append( PARFOR_INST_END );	
			sb.append( COMPONENTS_DELIM );	
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( pfpb.getExitInstructions() ) );
			sb.append( PARFOR_INST_END );
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_PBS_BEGIN );
			sb.append( rSerializeProgramBlocks( pfpb.getChildBlocks() ) );
			sb.append( PARFOR_PBS_END );
		}				
		else if ( pb instanceof IfProgramBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( ipb.getPredicate() ) );
			sb.append( PARFOR_INST_END );	
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( ipb.getExitInstructions() ) );
			sb.append( PARFOR_INST_END );
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_PBS_BEGIN );
			sb.append( rSerializeProgramBlocks( ipb.getChildBlocksIfBody() ) );
			sb.append( PARFOR_PBS_END );
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_PBS_BEGIN );
			sb.append( rSerializeProgramBlocks( ipb.getChildBlocksElseBody() ) );
			sb.append( PARFOR_PBS_END );
		}
		else if( pb instanceof FunctionProgramBlock && !(pb instanceof ExternalFunctionProgramBlock) )
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock) pb;
			
			sb.append( serializeDataIdentifiers( fpb.getInputParams() ) );
			sb.append( COMPONENTS_DELIM );
			sb.append( serializeDataIdentifiers( fpb.getOutputParams() ) );
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( fpb.getInstructions() ) );
			sb.append( PARFOR_INST_END );	
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_PBS_BEGIN );
			sb.append( rSerializeProgramBlocks( fpb.getChildBlocks() ) );
			sb.append( PARFOR_PBS_END );
			sb.append( COMPONENTS_DELIM );
		}
		else if( pb instanceof ExternalFunctionProgramBlock )
		{
			if( !(pb instanceof ExternalFunctionProgramBlockCP) ) 
			{
				throw new DMLRuntimeException( NOT_SUPPORTED_EXTERNALFUNCTION_PB );
			}
			
			ExternalFunctionProgramBlockCP fpb = (ExternalFunctionProgramBlockCP) pb;
			
			sb.append( serializeDataIdentifiers( fpb.getInputParams() ) );
			sb.append( COMPONENTS_DELIM );
			sb.append( serializeDataIdentifiers( fpb.getOutputParams() ) );
			sb.append( COMPONENTS_DELIM );
			sb.append( serializeStringHashMap( fpb.getOtherParams() ) );
			sb.append( COMPONENTS_DELIM );
			sb.append( fpb.getBaseDir() );
			sb.append( COMPONENTS_DELIM );
			
			sb.append( PARFOR_INST_BEGIN );
			//create on construction anyway 
			//sb.append( serializeInstructions( fpb.getInstructions() ) ); 
			sb.append( PARFOR_INST_END );	
			sb.append( COMPONENTS_DELIM );
			sb.append( PARFOR_PBS_BEGIN );
			sb.append( rSerializeProgramBlocks( fpb.getChildBlocks() ) );
			sb.append( PARFOR_PBS_END );
		}
		else //all generic program blocks
		{
			sb.append( PARFOR_INST_BEGIN );
			sb.append( serializeInstructions( pb.getInstructions() ) );
			sb.append( PARFOR_INST_END );
		}
		
		
		//handle end
		sb.append( PARFOR_PB_END );
		
		return sb.toString();
	}

	
	////////////////////////////////
	// PARSING 
	////////////////////////////////
	
	
	/**
	 * 
	 * @param in
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ParForBody parseParForBody( String in, int id ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ParForBody body = new ParForBody();
		
		//header elimination
		String tmpin = in.replaceAll(NEWLINE, ""); //normalization
		tmpin = tmpin.substring(PARFORBODY_BEGIN.length(),tmpin.length()-PARFORBODY_END.length()); //remove start/end
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(tmpin, COMPONENTS_DELIM);
		
		//handle DMLScript UUID (NOTE: set directly in DMLScript)
		//(master UUID is used for all nodes (in order to simply cleanup))
		DMLScript.setUUID( st.nextToken() );
		
		//handle DML config (NOTE: set directly in ConfigurationManager)
		String confStr = st.nextToken();
		if( !InfrastructureAnalyzer.isLocalMode() ) {
			DMLConfig config = DMLConfig.parseDMLConfig(confStr);
			ConfigurationManager.setConfig(config);
		}
		
		//handle program
		String progStr = st.nextToken();
	    progStr = progStr.replaceAll(CP_ROOT_THREAD_ID, CP_CHILD_THREAD+id); //replace for all instruction  
		Program prog = parseProgram( progStr, id ); 
		
		//handle result variable names
		String rvarStr = st.nextToken();
		ArrayList<String> rvars = parseStringArrayList(rvarStr);
		body.setResultVarNames(rvars);
		
		//handle execution context
		String ecStr = st.nextToken();
		ExecutionContext ec = parseExecutionContext( ecStr );
			
		//handle program blocks
		String spbs = st.nextToken();
		spbs = spbs.replaceAll(CP_ROOT_THREAD_ID, CP_CHILD_THREAD+id); //replace for all instruction 
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(spbs, prog, id);
		
		body.setChildBlocks( pbs );
		body.setEc( ec );
		
		return body;		
	}
	
	/**
	 * 
	 * @param in
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static Program parseProgram( String in, int id ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String lin = in.substring( PARFOR_PROG_BEGIN.length(),in.length()-PARFOR_PROG_END.length()).trim(); 
		
		Program prog = new Program();
		HashMap<String,FunctionProgramBlock> fc = parseFunctionProgramBlocks(lin, prog, id);
	
		for( Entry<String,FunctionProgramBlock> e : fc.entrySet() )
		{
			String[] keypart = e.getKey().split( Program.KEY_DELIM );
			String namespace = keypart[0];
			String name      = keypart[1];
			
			prog.addFunctionProgramBlock(namespace, name, e.getValue());
		}
		
		return prog;
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static LocalVariableMap parseVariables(String in) 
		throws DMLRuntimeException
	{
		String varStr = in.substring( PARFOR_VARS_BEGIN.length(),in.length()-PARFOR_VARS_END.length()).trim(); 
		return LocalVariableMap.deserialize(varStr);
	}
	
	/**
	 * 
	 * @param in
	 * @param prog
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static HashMap<String,FunctionProgramBlock> parseFunctionProgramBlocks( String in, Program prog, int id ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		HashMap<String,FunctionProgramBlock> ret = new HashMap<String, FunctionProgramBlock>();
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer( in, ELEMENT_DELIM );
		
		while( st.hasMoreTokens() )
		{
			String lvar  = st.nextToken(); //with ID = CP_CHILD_THREAD+id for current use
			
			//put first copy into prog (for direct use)
			int index = lvar.indexOf( KEY_VALUE_DELIM );
			String tmp1 = lvar.substring(0, index); // + CP_CHILD_THREAD+id;
			String tmp2 = lvar.substring(index + 1);
			ret.put(tmp1, (FunctionProgramBlock)rParseProgramBlock(tmp2, prog, id));
			
			//put first copy into prog (for future deep copies)
			//index = lvar2.indexOf("=");
			//tmp1 = lvar2.substring(0, index);
			//tmp2 = lvar2.substring(index + 1);
			//ret.put(tmp1, (FunctionProgramBlock)rParseProgramBlock(tmp2, prog, id));
		}

		return ret;
	}
	
	/**
	 * 
	 * @param in
	 * @param prog
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ArrayList<ProgramBlock> rParseProgramBlocks(String in, Program prog, int id) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ArrayList<ProgramBlock> pbs = new ArrayList<ProgramBlock>();
		String tmpdata = in.substring(PARFOR_PBS_BEGIN.length(),in.length()-PARFOR_PBS_END.length()); ;
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(tmpdata, ELEMENT_DELIM);
		
		while( st.hasMoreTokens() )
		{
			String tmp = st.nextToken();
			pbs.add( rParseProgramBlock( tmp, prog, id ) );
		}
		
		return pbs;
	}
	
	/**
	 * 
	 * @param in
	 * @param prog
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ProgramBlock rParseProgramBlock( String in, Program prog, int id )
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ProgramBlock pb = null;
		
		if( in.startsWith( PARFOR_PB_WHILE ) )
			pb = rParseWhileProgramBlock( in, prog, id );
		else if ( in.startsWith(PARFOR_PB_FOR ) )
			pb = rParseForProgramBlock( in, prog, id );
		else if ( in.startsWith(PARFOR_PB_PARFOR ) )
			pb = rParseParForProgramBlock( in, prog, id );
		else if ( in.startsWith(PARFOR_PB_IF ) )
			pb = rParseIfProgramBlock( in, prog, id );
		else if ( in.startsWith(PARFOR_PB_FC ) )
			pb = rParseFunctionProgramBlock( in, prog, id );
		else if ( in.startsWith(PARFOR_PB_EFC ) )
			pb = rParseExternalFunctionProgramBlock( in, prog, id );
 		else if ( in.startsWith(PARFOR_PB_BEGIN ) )
			pb = rParseGenericProgramBlock( in, prog, id );
		else 
			throw new DMLUnsupportedOperationException( NOT_SUPPORTED_PB+" "+in );
		
		return pb;
	}

	/**
	 * 
	 * @param in
	 * @param prog
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static WhileProgramBlock rParseWhileProgramBlock( String in, Program prog, int id ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String lin = in.substring( PARFOR_PB_WHILE.length(),in.length()-PARFOR_PB_END.length()); 
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(lin, COMPONENTS_DELIM);
		
		//LocalVariableMap vars = parseVariables(st.nextToken());
		
		//predicate instructions
		ArrayList<Instruction> inst = parseInstructions(st.nextToken());
		
		//exit instructions
		ArrayList<Instruction> exit = parseInstructions(st.nextToken());
		
		//program blocks
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(st.nextToken(), prog, id);
		
		WhileProgramBlock wpb = new WhileProgramBlock(prog,inst);
		wpb.setExitInstructions2(exit);
		wpb.setChildBlocks(pbs);
		//wpb.setVariables(vars);
		
		return wpb;
	}
	
	/**
	 * 
	 * @param in
	 * @param prog
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ForProgramBlock rParseForProgramBlock( String in, Program prog, int id ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String lin = in.substring( PARFOR_PB_FOR.length(),in.length()-PARFOR_PB_END.length()); 
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(lin, COMPONENTS_DELIM);
		
		//LocalVariableMap vars = parseVariables(st.nextToken());
		
		//inputs
		String[] iterPredVars = parseStringArray(st.nextToken());
		
		//instructions
		ArrayList<Instruction> from = parseInstructions(st.nextToken());
		ArrayList<Instruction> to = parseInstructions(st.nextToken());
		ArrayList<Instruction> incr = parseInstructions(st.nextToken());
		
		//exit instructions
		ArrayList<Instruction> exit = parseInstructions(st.nextToken());

		//program blocks
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(st.nextToken(), prog, id);

		ForProgramBlock fpb = new ForProgramBlock(prog, iterPredVars);
		fpb.setFromInstructions(from);
		fpb.setToInstructions(to);
		fpb.setIncrementInstructions(incr);
		fpb.setExitInstructions(exit);
		fpb.setChildBlocks(pbs);
		//fpb.setVariables(vars);
		
		return fpb;
	}
	
	/**
	 * 
	 * @param in
	 * @param prog
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ParForProgramBlock rParseParForProgramBlock( String in, Program prog, int id ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String lin = in.substring( PARFOR_PB_PARFOR.length(),in.length()-PARFOR_PB_END.length()); 
		lin = lin.replaceAll(CP_CHILD_THREAD+id, CP_ROOT_THREAD_ID); // reset placeholder to preinit state (replaced by deep copies of nested parfor pbs)
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(lin, COMPONENTS_DELIM);
		
		//LocalVariableMap vars = parseVariables(st.nextToken());
		
		//inputs
		String[] iterPredVars = parseStringArray(st.nextToken());
		ArrayList<String> resultVars = parseStringArrayList(st.nextToken());
		HashMap<String,String> params = parseStringHashMap(st.nextToken());
		
		//instructions
		ArrayList<Instruction> from = parseInstructions(st.nextToken());
		ArrayList<Instruction> to = parseInstructions(st.nextToken());
		ArrayList<Instruction> incr = parseInstructions(st.nextToken());
		
		//exit instructions
		ArrayList<Instruction> exit = parseInstructions(st.nextToken());

		//program blocks
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(st.nextToken(), prog, id);

		ParForProgramBlock pfpb = new ParForProgramBlock(id, prog, iterPredVars, params);
		pfpb.disableOptimization(); //already done in top-level parfor
		pfpb.setResultVariables(resultVars);		
		pfpb.setFromInstructions(from);
		pfpb.setToInstructions(to);
		pfpb.setIncrementInstructions(incr);
		pfpb.setExitInstructions(exit);
		pfpb.setChildBlocks(pbs);
		//pfpb.setVariables(vars);
		
		return pfpb;
	}
	
	/**
	 * 
	 * @param in
	 * @param prog
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static IfProgramBlock rParseIfProgramBlock( String in, Program prog, int id ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String lin = in.substring( PARFOR_PB_IF.length(),in.length()-PARFOR_PB_END.length()); 
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(lin, COMPONENTS_DELIM);
		
		//LocalVariableMap vars = parseVariables(st.nextToken());
		
		//predicate instructions
		ArrayList<Instruction> inst = parseInstructions(st.nextToken());
		
		//exit instructions
		ArrayList<Instruction> exit = parseInstructions(st.nextToken());
		
		//program blocks: if and else
		ArrayList<ProgramBlock> pbs1 = rParseProgramBlocks(st.nextToken(), prog, id);
		ArrayList<ProgramBlock> pbs2 = rParseProgramBlocks(st.nextToken(), prog, id);
		
		IfProgramBlock ipb = new IfProgramBlock(prog,inst);
		ipb.setExitInstructions2(exit);
		ipb.setChildBlocksIfBody(pbs1);
		ipb.setChildBlocksElseBody(pbs2);
		//ipb.setVariables(vars);
		
		return ipb;
	}
	
	/**
	 * 
	 * @param in
	 * @param prog
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static FunctionProgramBlock rParseFunctionProgramBlock( String in, Program prog, int id ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String lin = in.substring( PARFOR_PB_FC.length(),in.length()-PARFOR_PB_END.length()); 
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(lin, COMPONENTS_DELIM);
		
		//LocalVariableMap vars = parseVariables(st.nextToken());
		
		//inputs and outputs
		ArrayList<DataIdentifier> dat1 = parseDataIdentifiers(st.nextToken());
		ArrayList<DataIdentifier> dat2 = parseDataIdentifiers(st.nextToken());
		
		//instructions
		ArrayList<Instruction> inst = parseInstructions(st.nextToken());

		//program blocks
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(st.nextToken(), prog, id);

		Vector<DataIdentifier> tmp1 = new Vector<DataIdentifier>(dat1);
		Vector<DataIdentifier> tmp2 = new Vector<DataIdentifier>(dat2);
		FunctionProgramBlock fpb = new FunctionProgramBlock(prog, tmp1, tmp2);
		fpb.setInstructions(inst);
		fpb.setChildBlocks(pbs);
		//fpb.setVariables(vars);
		
		return fpb;
	}
	
	/**
	 * 
	 * @param in
	 * @param prog
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ExternalFunctionProgramBlock rParseExternalFunctionProgramBlock( String in, Program prog, int id ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String lin = in.substring( PARFOR_PB_EFC.length(),in.length()-PARFOR_PB_END.length()); 
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(lin, COMPONENTS_DELIM);
		
		//LocalVariableMap vars = parseVariables(st.nextToken());
		
		//inputs, outputs and params
		ArrayList<DataIdentifier> dat1 = parseDataIdentifiers(st.nextToken());
		ArrayList<DataIdentifier> dat2 = parseDataIdentifiers(st.nextToken());
		HashMap<String,String> dat3 = parseStringHashMap(st.nextToken());

		//basedir
		String basedir = st.nextToken();
		
		//instructions
		@SuppressWarnings("unused")
		ArrayList<Instruction> inst = parseInstructions(st.nextToken()); //required for removing INST BEGIN, END

		//program blocks
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(st.nextToken(), prog, id);

		Vector<DataIdentifier> tmp1 = new Vector<DataIdentifier>(dat1);
		Vector<DataIdentifier> tmp2 = new Vector<DataIdentifier>(dat2);
		
		//only CP external functions, because no nested MR jobs for reblocks
		ExternalFunctionProgramBlockCP efpb = new ExternalFunctionProgramBlockCP(prog, tmp1, tmp2, dat3, basedir);
		//efpb.setInstructions(inst);
		efpb.setChildBlocks(pbs);
		//efpb.setVariables(vars);
		
		return efpb;
	}
	
	/**
	 * 
	 * @param in
	 * @param prog
	 * @param id
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ProgramBlock rParseGenericProgramBlock( String in, Program prog, int id ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String lin = in.substring( PARFOR_PB_BEGIN.length(),in.length()-PARFOR_PB_END.length()); 
		StringTokenizer st = new StringTokenizer(lin,COMPONENTS_DELIM);
		//LocalVariableMap vars = parseVariables(st.nextToken());
		
		ArrayList<Instruction> inst = parseInstructions(st.nextToken());
		
		ProgramBlock pb = new ProgramBlock(prog);
		pb.setInstructions(inst);
		//pb.setVariables(vars);
		
		return pb;
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ArrayList<Instruction> parseInstructions( String in ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ArrayList<Instruction> insts = new ArrayList<Instruction>();  
		

		String lin = in.substring( PARFOR_INST_BEGIN.length(),in.length()-PARFOR_INST_END.length()); 
		StringTokenizer st = new StringTokenizer(lin, ELEMENT_DELIM);
		while(st.hasMoreTokens())
		{
			//Note that at this point only CP instructions and External function instruction can occur
			String instStr = st.nextToken(); 
			Instruction tmpinst = CPInstructionParser.parseSingleInstruction(instStr);
			insts.add( tmpinst );
		}
		
		return insts;
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static HashMap<String,String> parseStringHashMap( String in )
	{
		HashMap<String,String> vars = new HashMap<String, String>();
		StringTokenizer st = new StringTokenizer(in,ELEMENT_DELIM);
		while( st.hasMoreTokens() )
		{
			String lin = st.nextToken();
			int index = lin.indexOf( KEY_VALUE_DELIM );
			String tmp1 = lin.substring(0, index);
			String tmp2 = lin.substring(index + 1);			
			vars.put(tmp1, tmp2);
		}
		
		return vars;
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static ArrayList<String> parseStringArrayList( String in )
	{
		ArrayList<String> vars = new ArrayList<String>();
		StringTokenizer st = new StringTokenizer(in,ELEMENT_DELIM);
		while( st.hasMoreTokens() )
		{
			String tmp = st.nextToken();			
			vars.add(tmp);
		}
		
		return vars;
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static String[] parseStringArray( String in )
	{
		StringTokenizer st = new StringTokenizer(in, ELEMENT_DELIM);
		int len = st.countTokens();
		String[] a = new String[len];
		for( int i=0; i<len; i++ )
		{
			String tmp = st.nextToken();
			if( tmp.equals("null") )
				a[i] = null;
			else
				a[i] = tmp;
		}
		return a;
	}

	/**
	 * 
	 * @param in
	 * @return
	 */
	public static ArrayList<DataIdentifier> parseDataIdentifiers( String in )
	{
		ArrayList<DataIdentifier> vars = new ArrayList<DataIdentifier>();
		StringTokenizer st = new StringTokenizer(in, ELEMENT_DELIM);
		while( st.hasMoreTokens() )
		{
			String tmp = st.nextToken();
			DataIdentifier dat = parseDataIdentifier( tmp );
			vars.add(dat);
		}

		return vars;
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static DataIdentifier parseDataIdentifier( String in )
	{
		StringTokenizer st = new StringTokenizer(in, DATA_FIELD_DELIM);
		String name = st.nextToken();
		DataType dt = DataType.valueOf(st.nextToken());
		ValueType vt = ValueType.valueOf(st.nextToken());
		
		DataIdentifier dat = new DataIdentifier(name);
		dat.setDataType(dt);
		dat.setValueType(vt);
		
		return dat;
	}
	
	/**
	 * NOTE: MRJobConfiguration cannot be used for the general case because program blocks and
	 * related symbol tables can be hierarchically structured.
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Object[] parseDataObject(String in) 
		throws DMLRuntimeException
	{
		Object[] ret = new Object[2];
	
		StringTokenizer st = new StringTokenizer(in, DATA_FIELD_DELIM );
		String name = st.nextToken();
		DataType datatype = DataType.valueOf( st.nextToken() );
		ValueType valuetype = ValueType.valueOf( st.nextToken() );
		String valString = st.nextToken();
		Data dat = null;
		switch( datatype )
		{
			case SCALAR:
			{
				switch ( valuetype )
				{
					case INT:
						int value1 = Integer.parseInt(valString);
						dat = new IntObject(name,value1);
						break;
					case DOUBLE:
						double value2 = Double.parseDouble(valString);
						dat = new DoubleObject(name,value2);
						break;
					case BOOLEAN:
						boolean value3 = Boolean.parseBoolean(valString);
						dat = new BooleanObject(name,value3);
						break;
					case STRING:
						dat = new StringObject(name,valString);
						break;
					default:
						throw new DMLRuntimeException("Unable to parse valuetype "+valuetype);		
				}
				break;
		    }
			case MATRIX:
			{
				MatrixObject mo = new MatrixObject(valuetype,valString);
				long rows = Long.parseLong( st.nextToken() );
				long cols = Long.parseLong( st.nextToken() );
				int brows = Integer.parseInt( st.nextToken() );
				int bcols = Integer.parseInt( st.nextToken() );
				long nnz = Long.parseLong( st.nextToken() );
				InputInfo iin = InputInfo.stringToInputInfo( st.nextToken() );
				OutputInfo oin = OutputInfo.stringToOutputInfo( st.nextToken() );		
				PDataPartitionFormat partFormat = PDataPartitionFormat.valueOf( st.nextToken() );
				MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, brows, bcols, nnz); 
				MatrixFormatMetaData md = new MatrixFormatMetaData( mc, oin, iin );
				mo.setMetaData( md );
				mo.setVarName( name );
				if( partFormat!=PDataPartitionFormat.NONE )
					mo.setPartitioned( partFormat, -1 ); //TODO once we support BLOCKWISE_N we should support it here as well
				dat = mo;
				break;
			}
			default:
				throw new DMLRuntimeException("Unable to parse datatype "+datatype);
		}
		
		ret[0] = name;
		ret[1] = dat;
		return ret;
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static ExecutionContext parseExecutionContext(String in) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ExecutionContext ec = null;
		
		String lin = in.substring(PARFOR_EC_BEGIN.length(),in.length()-PARFOR_EC_END.length()).trim(); 
		
		if( !lin.equals( EMPTY ) )
		{
			LocalVariableMap vars = parseVariables(lin);
			ec = new ExecutionContext( false );
			ec.setVariables(vars);
		}
		
		return ec;
	}

	
	/**
	 * Custom StringTokenizer for splitting strings of hierarchies. The basic idea is to
	 * search for delim-Strings on the same hierarchy level, while delims of lower hierarchy
	 * levels are skipped.  
	 * 
	 */
	private static class HierarchyAwareStringTokenizer //extends StringTokenizer
	{
		private String _str = null;
		private String _del = null;
		private int    _off = -1;
		
		public HierarchyAwareStringTokenizer( String in, String delim )
		{
			//super(in);
			_str = in;
			_del = delim;
			_off = delim.length();
		}

		public boolean hasMoreTokens() 
		{
			return (_str.length() > 0);
		}

		public String nextToken() 
		{
			int nextDelim = determineNextSameLevelIndexOf(_str, _del);		
			String token = null;
			if(nextDelim < 0) 
			{
				nextDelim = _str.length();
				_off = 0;
			}
			token = _str.substring(0,nextDelim);
			_str = _str.substring( nextDelim + _off );
			return token;
		}
				
		private int determineNextSameLevelIndexOf( String data, String pattern  )
		{
			String tmpdata = data;
			int index      = 0;
			int count      = 0;
			int off=0,i1,i2,i3,min;
			
			while(true)
			{
				i1 = tmpdata.indexOf(pattern);
				i2 = tmpdata.indexOf(LEVELIN);
				i3 = tmpdata.indexOf(LEVELOUT);
				
				if( i1 < 0 ) return i1; //no pattern found at all			
				
				min = i1; //min >= 0 by definition
				if( i2 >= 0 ) min = Math.min(min, i2);
				if( i3 >= 0 ) min = Math.min(min, i3);
				
				//stack maintenance
				if( i1 == min && count == 0 )
					return index+i1;
				else if( i2 == min )
				{
					count++;
					off = LEVELIN.length();
				}
				else if( i3 == min )
				{
					count--;
					off = LEVELOUT.length();
				}
			
				//prune investigated string
				index += min+off;
				tmpdata = tmpdata.substring(min+off);
			}
		}
	}
}

/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.util;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.JobConf;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.conf.CompilerConfig;
import org.tugraz.sysds.conf.CompilerConfig.ConfigType;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.conf.DMLConfig;
import org.tugraz.sysds.hops.Hop;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.hops.recompile.Recompiler;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.parser.DMLProgram;
import org.tugraz.sysds.parser.DataIdentifier;
import org.tugraz.sysds.parser.ForStatementBlock;
import org.tugraz.sysds.parser.IfStatementBlock;
import org.tugraz.sysds.parser.ParForStatementBlock;
import org.tugraz.sysds.parser.ParForStatementBlock.ResultVar;
import org.tugraz.sysds.parser.StatementBlock;
import org.tugraz.sysds.parser.WhileStatementBlock;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.codegen.CodegenUtils;
import org.tugraz.sysds.runtime.controlprogram.BasicProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.ForProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.IfProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.LocalVariableMap;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock.PExecMode;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock.PartitionFormat;
import org.tugraz.sysds.runtime.controlprogram.Program;
import org.tugraz.sysds.runtime.controlprogram.ProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.WhileProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.tugraz.sysds.runtime.controlprogram.paramserv.SparkPSBody;
import org.tugraz.sysds.runtime.controlprogram.parfor.ParForBody;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.tugraz.sysds.runtime.instructions.CPInstructionParser;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.InstructionParser;
import org.tugraz.sysds.runtime.instructions.cp.BooleanObject;
import org.tugraz.sysds.runtime.instructions.cp.CPInstruction;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.instructions.cp.DoubleObject;
import org.tugraz.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.tugraz.sysds.runtime.instructions.cp.IntObject;
import org.tugraz.sysds.runtime.instructions.cp.ListObject;
import org.tugraz.sysds.runtime.instructions.cp.ScalarObject;
import org.tugraz.sysds.runtime.instructions.cp.SpoofCPInstruction;
import org.tugraz.sysds.runtime.instructions.cp.StringObject;
import org.tugraz.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.tugraz.sysds.runtime.instructions.gpu.GPUInstruction;
import org.tugraz.sysds.runtime.instructions.spark.SPInstruction;
import org.tugraz.sysds.runtime.lineage.Lineage;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaDataFormat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.StringTokenizer;
import java.util.stream.Collectors;

/**
 * Program converter functionalities for 
 *   (1) creating deep copies of program blocks, instructions, function program blocks, and 
 *   (2) serializing and parsing of programs, program blocks, functions program blocks.
 * 
 */
//TODO: rewrite class to instance-based invocation (grown gradually and now inappropriate design)
public class ProgramConverter 
{
	protected static final Log LOG = LogFactory.getLog(ProgramConverter.class.getName());

	//use escaped unicodes for separators in order to prevent string conflict
	public static final String NEWLINE           = "\n"; //System.lineSeparator();
	public static final String COMPONENTS_DELIM  = "\u236e"; //semicolon w/ bar; ";";
	public static final String ELEMENT_DELIM     = "\u236a"; //comma w/ bar; ",";
	public static final String ELEMENT_DELIM2    = ",";
	public static final String DATA_FIELD_DELIM  = "\u007c"; //"|";
	public static final String KEY_VALUE_DELIM   = "\u003d"; //"=";
	public static final String LEVELIN           = "\u23a8"; //variant of left curly bracket; "\u007b"; //"{";
	public static final String LEVELOUT          = "\u23ac"; //variant of right curly bracket; "\u007d"; //"}";
	public static final String EMPTY             = "null";
	public static final String DASH              = "-";
	public static final String REF               = "ref";
	public static final String LIST_ELEMENT_DELIM = "\t";

	public static final String CDATA_BEGIN = "<![CDATA[";
	public static final String CDATA_END = " ]]>";
	
	public static final String PROG_BEGIN = " PROG" + LEVELIN;
	public static final String PROG_END = LEVELOUT;
	public static final String VARS_BEGIN = "VARS: ";
	public static final String VARS_END = "";
	public static final String PBS_BEGIN = " PBS" + LEVELIN;
	public static final String PBS_END = LEVELOUT;
	public static final String INST_BEGIN = " INST: ";
	public static final String INST_END = "";
	public static final String EC_BEGIN = " EC: ";
	public static final String EC_END = "";
	public static final String PB_BEGIN = " PB" + LEVELIN;
	public static final String PB_END = LEVELOUT;
	public static final String PB_WHILE = " WHILE" + LEVELIN;
	public static final String PB_FOR = " FOR" + LEVELIN;
	public static final String PB_PARFOR = " PARFOR" + LEVELIN;
	public static final String PB_IF = " IF" + LEVELIN;
	public static final String PB_FC = " FC" + LEVELIN;
	public static final String PB_EFC = " EFC" + LEVELIN;

	public static final String CONF_STATS = "stats";

	// Used for parfor
	public static final String PARFORBODY_BEGIN  = CDATA_BEGIN + "PARFORBODY" + LEVELIN;
	public static final String PARFORBODY_END    = LEVELOUT + CDATA_END;

	// Used for paramserv builtin function
	public static final String PSBODY_BEGIN = CDATA_BEGIN + "PSBODY" + LEVELIN;
	public static final String PSBODY_END = LEVELOUT + CDATA_END;
	
	//exception msgs
	public static final String NOT_SUPPORTED_EXTERNALFUNCTION_PB = "Not supported: ExternalFunctionProgramBlock contains MR instructions. " +
																	"(ExternalFunctionPRogramBlockCP can be used)";
	public static final String NOT_SUPPORTED_SPARK_INSTRUCTION   = "Not supported: Instructions of type other than CP instructions";
	public static final String NOT_SUPPORTED_SPARK_PARFOR        = "Not supported: Nested ParFOR REMOTE_SPARK due to possible deadlocks." +
																	"(LOCAL can be used for innner ParFOR)";
	public static final String NOT_SUPPORTED_PB                  = "Not supported: type of program block";
	
	////////////////////////////////
	// CREATION of DEEP COPIES
	////////////////////////////////
	
	/**
	 * Creates a deep copy of the given execution context.
	 * For rt_platform=Hadoop, execution context has a symbol table.
	 * 
	 * @param ec execution context
	 * @return execution context
	 * @throws CloneNotSupportedException if CloneNotSupportedException occurs
	 */
	public static ExecutionContext createDeepCopyExecutionContext(ExecutionContext ec) 
		throws CloneNotSupportedException
	{
		ExecutionContext cpec = ExecutionContextFactory.createContext(false, ec.getProgram());
		cpec.setVariables((LocalVariableMap) ec.getVariables().clone());
		if( ec.getLineage() != null )
			cpec.setLineage(new Lineage(ec.getLineage()));
		
		//handle result variables with in-place update flag
		//(each worker requires its own copy of the empty matrix object)
		for( String var : cpec.getVariables().keySet() ) {
			Data dat = cpec.getVariables().get(var);
			if( dat instanceof MatrixObject && ((MatrixObject)dat).getUpdateType().isInPlace() ) {
				MatrixObject mo = (MatrixObject)dat;
				MatrixObject moNew = new MatrixObject(mo); 
				if( mo.getNnz() != 0 ){
					// If output matrix is not empty (NNZ != 0), then local copy is created so that 
					// update in place operation can be applied.
					MatrixBlock mbVar = mo.acquireRead();
					moNew.acquireModify (new MatrixBlock(mbVar));
					mo.release();
				} else {
					//create empty matrix block w/ dense representation (preferred for update in-place)
					//Creating a dense matrix block is valid because empty block not allocated and transfer 
					// to sparse representation happens in left indexing in place operation.
					moNew.acquireModify(new MatrixBlock((int)mo.getNumRows(), (int)mo.getNumColumns(), false));
				}
				moNew.release();
				cpec.setVariable(var, moNew);
			}
		}
		
		return cpec;
	}
	
	/**
	 * This recursively creates a deep copy of program blocks and transparently replaces filenames according to the
	 * specified parallel worker in order to avoid conflicts between parworkers. This happens recursively in order
	 * to support arbitrary control-flow constructs within a parfor. 
	 * 
	 * @param childBlocks child program blocks
	 * @param pid ?
	 * @param IDPrefix ?
	 * @param fnStack ?
	 * @param fnCreated ?
	 * @param plain if true, full deep copy without id replacement
	 * @param forceDeepCopy if true, force deep copy
	 * @return list of program blocks
	 */
	public static ArrayList<ProgramBlock> rcreateDeepCopyProgramBlocks(ArrayList<ProgramBlock> childBlocks, long pid, int IDPrefix, HashSet<String> fnStack, HashSet<String> fnCreated, boolean plain, boolean forceDeepCopy) 
	{
		ArrayList<ProgramBlock> tmp = new ArrayList<>();
		
		for( ProgramBlock pb : childBlocks )
		{
			Program prog = pb.getProgram();
			ProgramBlock tmpPB = null;
			
			if( pb instanceof WhileProgramBlock ) {
				tmpPB = createDeepCopyWhileProgramBlock((WhileProgramBlock) pb, pid, IDPrefix, prog, fnStack, fnCreated, plain, forceDeepCopy);
			}
			else if( pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock) ) {
				tmpPB = createDeepCopyForProgramBlock((ForProgramBlock) pb, pid, IDPrefix, prog, fnStack, fnCreated, plain, forceDeepCopy );
			}
			else if( pb instanceof ParForProgramBlock ) {
				ParForProgramBlock pfpb = (ParForProgramBlock) pb;
				if( ParForProgramBlock.ALLOW_NESTED_PARALLELISM )
					tmpPB = createDeepCopyParForProgramBlock(pfpb, pid, IDPrefix, prog, fnStack, fnCreated, plain, forceDeepCopy);
				else 
					tmpPB = createDeepCopyForProgramBlock((ForProgramBlock) pb, pid, IDPrefix, prog, fnStack, fnCreated, plain, forceDeepCopy);
			}
			else if( pb instanceof IfProgramBlock ) {
				tmpPB = createDeepCopyIfProgramBlock((IfProgramBlock) pb, pid, IDPrefix, prog, fnStack, fnCreated, plain, forceDeepCopy);
			}
			else if( pb instanceof BasicProgramBlock ) { //last-level program block
				BasicProgramBlock bpb = (BasicProgramBlock) pb;
				tmpPB = new BasicProgramBlock(prog); // general case use for most PBs
				
				//for recompile in the master node JVM
				tmpPB.setStatementBlock(createStatementBlockCopy(bpb.getStatementBlock(), pid, plain, forceDeepCopy)); 
				tmpPB.setThreadID(pid);
				
				//copy instructions
				((BasicProgramBlock)tmpPB).setInstructions(
					createDeepCopyInstructionSet(bpb.getInstructions(),
					pid, IDPrefix, prog, fnStack, fnCreated, plain, true) );
			}
			
			//copy symbol table
			//tmpPB.setVariables( pb.getVariables() ); //implicit cloning
			
			tmp.add(tmpPB);
		}
		
		return tmp;
	}

	public static WhileProgramBlock createDeepCopyWhileProgramBlock(WhileProgramBlock wpb, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, HashSet<String> fnCreated, boolean plain, boolean forceDeepCopy) {
		ArrayList<Instruction> predinst = createDeepCopyInstructionSet(wpb.getPredicate(), pid, IDPrefix, prog, fnStack, fnCreated, plain, true);
		WhileProgramBlock tmpPB = new WhileProgramBlock(prog, predinst);
		tmpPB.setStatementBlock( createWhileStatementBlockCopy((WhileStatementBlock) wpb.getStatementBlock(), pid, plain, forceDeepCopy) );
		tmpPB.setThreadID(pid);
		tmpPB.setChildBlocks(rcreateDeepCopyProgramBlocks(wpb.getChildBlocks(), pid, IDPrefix, fnStack, fnCreated, plain, forceDeepCopy));
		return tmpPB;
	}

	public static IfProgramBlock createDeepCopyIfProgramBlock(IfProgramBlock ipb, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, HashSet<String> fnCreated, boolean plain, boolean forceDeepCopy) {
		ArrayList<Instruction> predinst = createDeepCopyInstructionSet(ipb.getPredicate(), pid, IDPrefix, prog, fnStack, fnCreated, plain, true);
		IfProgramBlock tmpPB = new IfProgramBlock(prog, predinst);
		tmpPB.setStatementBlock( createIfStatementBlockCopy((IfStatementBlock)ipb.getStatementBlock(), pid, plain, forceDeepCopy ) );
		tmpPB.setThreadID(pid);
		tmpPB.setChildBlocksIfBody(rcreateDeepCopyProgramBlocks(ipb.getChildBlocksIfBody(), pid, IDPrefix, fnStack, fnCreated, plain, forceDeepCopy));
		tmpPB.setChildBlocksElseBody(rcreateDeepCopyProgramBlocks(ipb.getChildBlocksElseBody(), pid, IDPrefix, fnStack, fnCreated, plain, forceDeepCopy));
		return tmpPB;
	}

	public static ForProgramBlock createDeepCopyForProgramBlock(ForProgramBlock fpb, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, HashSet<String> fnCreated, boolean plain, boolean forceDeepCopy) {
		ForProgramBlock tmpPB = new ForProgramBlock(prog,fpb.getIterVar());
		tmpPB.setStatementBlock( createForStatementBlockCopy((ForStatementBlock)fpb.getStatementBlock(), pid, plain, forceDeepCopy));
		tmpPB.setThreadID(pid);
		tmpPB.setFromInstructions( createDeepCopyInstructionSet(fpb.getFromInstructions(), pid, IDPrefix, prog, fnStack, fnCreated, plain, true) );
		tmpPB.setToInstructions( createDeepCopyInstructionSet(fpb.getToInstructions(), pid, IDPrefix, prog, fnStack, fnCreated, plain, true) );
		tmpPB.setIncrementInstructions( createDeepCopyInstructionSet(fpb.getIncrementInstructions(), pid, IDPrefix, prog, fnStack, fnCreated, plain, true) );
		tmpPB.setChildBlocks( rcreateDeepCopyProgramBlocks(fpb.getChildBlocks(), pid, IDPrefix, fnStack, fnCreated, plain, forceDeepCopy) );
		return tmpPB;
	}

	public static ForProgramBlock createShallowCopyForProgramBlock(ForProgramBlock fpb, Program prog ) {
		ForProgramBlock tmpPB = new ForProgramBlock(prog,fpb.getIterVar());
		tmpPB.setFromInstructions( fpb.getFromInstructions() );
		tmpPB.setToInstructions( fpb.getToInstructions() );
		tmpPB.setIncrementInstructions( fpb.getIncrementInstructions() );
		tmpPB.setChildBlocks( fpb.getChildBlocks() );
		return tmpPB;
	}

	public static ParForProgramBlock createDeepCopyParForProgramBlock(ParForProgramBlock pfpb, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, HashSet<String> fnCreated, boolean plain, boolean forceDeepCopy) {
		ParForProgramBlock tmpPB = null;
		
		if( IDPrefix == -1 ) //still on master node
			tmpPB = new ParForProgramBlock(prog,pfpb.getIterVar(), pfpb.getParForParams(), pfpb.getResultVariables());
		else //child of remote ParWorker at any level
			tmpPB = new ParForProgramBlock(IDPrefix, prog, pfpb.getIterVar(), pfpb.getParForParams(), pfpb.getResultVariables());
		
		tmpPB.setStatementBlock( createForStatementBlockCopy( (ForStatementBlock) pfpb.getStatementBlock(), pid, plain, forceDeepCopy) );
		tmpPB.setThreadID(pid);
		
		tmpPB.disableOptimization(); //already done in top-level parfor
		tmpPB.disableMonitorReport(); //already done in top-level parfor
		
		tmpPB.setFromInstructions( createDeepCopyInstructionSet(pfpb.getFromInstructions(), pid, IDPrefix, prog, fnStack, fnCreated, plain, true) );
		tmpPB.setToInstructions( createDeepCopyInstructionSet(pfpb.getToInstructions(), pid, IDPrefix, prog, fnStack, fnCreated, plain, true) );
		tmpPB.setIncrementInstructions( createDeepCopyInstructionSet(pfpb.getIncrementInstructions(), pid, IDPrefix, prog, fnStack, fnCreated, plain, true) );
		
		//NOTE: Normally, no recursive copy because (1) copied on each execution in this PB anyway 
		//and (2) leave placeholders as they are. However, if plain, an explicit deep copy is requested.
		if( plain || forceDeepCopy )
			tmpPB.setChildBlocks( rcreateDeepCopyProgramBlocks(pfpb.getChildBlocks(), pid, IDPrefix, fnStack, fnCreated, plain, forceDeepCopy) ); 
		else
			tmpPB.setChildBlocks( pfpb.getChildBlocks() );
		
		return tmpPB;
	}
	
	/**
	 * This creates a deep copy of a function program block. The central reference to singletons of function program blocks
	 * poses the need for explicit copies in order to prevent conflicting writes of temporary variables (see ExternalFunctionProgramBlock.
	 * 
	 * @param namespace function namespace
	 * @param oldName ?
	 * @param pid ?
	 * @param IDPrefix ?
	 * @param prog runtime program
	 * @param fnStack ?
	 * @param fnCreated ?
	 * @param plain ?
	 */
	public static void createDeepCopyFunctionProgramBlock(String namespace, String oldName, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, HashSet<String> fnCreated, boolean plain) 
	{
		//fpb guaranteed to be non-null (checked inside getFunctionProgramBlock)
		FunctionProgramBlock fpb = prog.getFunctionProgramBlock(namespace, oldName);
		String fnameNew = (plain)? oldName :(oldName+Lop.CP_CHILD_THREAD+pid); 
		String fnameNewKey = DMLProgram.constructFunctionKey(namespace,fnameNew);

		if( prog.getFunctionProgramBlocks().containsKey(fnameNewKey) )
			return; //prevent redundant deep copy if already existent
		
		//create deep copy
		FunctionProgramBlock copy = null;
		ArrayList<DataIdentifier> tmp1 = new ArrayList<>();
		ArrayList<DataIdentifier> tmp2 = new ArrayList<>();
		if( fpb.getInputParams()!= null )
			tmp1.addAll(fpb.getInputParams());
		if( fpb.getOutputParams()!= null )
			tmp2.addAll(fpb.getOutputParams());
		
		
		if( !fnStack.contains(fnameNewKey) ) {
			fnStack.add(fnameNewKey);
			copy = new FunctionProgramBlock(prog, tmp1, tmp2);
			copy.setChildBlocks( rcreateDeepCopyProgramBlocks(fpb.getChildBlocks(), pid, IDPrefix, fnStack, fnCreated, plain, fpb.isRecompileOnce()) );
			copy.setRecompileOnce( fpb.isRecompileOnce() );
			copy.setThreadID(pid);
			fnStack.remove(fnameNewKey);
		}
		else //stop deep copy for recursive function calls
			copy = fpb;
		
		//copy.setVariables( (LocalVariableMap) fpb.getVariables() ); //implicit cloning
		//note: instructions not used by function program block
		
		//put 
		prog.addFunctionProgramBlock(namespace, fnameNew, copy);
		fnCreated.add(DMLProgram.constructFunctionKey(namespace, fnameNew));
	}

	public static FunctionProgramBlock createDeepCopyFunctionProgramBlock(FunctionProgramBlock fpb, HashSet<String> fnStack, HashSet<String> fnCreated) 
	{
		if( fpb == null )
			throw new DMLRuntimeException("Unable to create a deep copy of a non-existing FunctionProgramBlock.");
	
		//create deep copy
		FunctionProgramBlock copy = null;
		ArrayList<DataIdentifier> tmp1 = new ArrayList<>();
		ArrayList<DataIdentifier> tmp2 = new ArrayList<>();
		if( fpb.getInputParams()!= null )
			tmp1.addAll(fpb.getInputParams());
		if( fpb.getOutputParams()!= null )
			tmp2.addAll(fpb.getOutputParams());
		
		copy = new FunctionProgramBlock(fpb.getProgram(), tmp1, tmp2);
		copy.setChildBlocks( rcreateDeepCopyProgramBlocks(fpb.getChildBlocks(), 0, -1, fnStack, fnCreated, true, fpb.isRecompileOnce()) );
		copy.setStatementBlock( fpb.getStatementBlock() );
		copy.setRecompileOnce(fpb.isRecompileOnce());
		//copy.setVariables( (LocalVariableMap) fpb.getVariables() ); //implicit cloning
		//note: instructions not used by function program block
	
		return copy;
	}

	
	/**
	 * Creates a deep copy of an array of instructions and replaces the placeholders of parworker
	 * IDs with the concrete IDs of this parfor instance. This is a helper method uses for generating
	 * deep copies of program blocks.
	 * 
	 * @param instSet list of instructions
	 * @param pid ?
	 * @param IDPrefix ?
	 * @param prog runtime program
	 * @param fnStack ?
	 * @param fnCreated ?
	 * @param plain ?
	 * @param cpFunctions ?
	 * @return list of instructions
	 */
	public static ArrayList<Instruction> createDeepCopyInstructionSet(ArrayList<Instruction> instSet, long pid, int IDPrefix, Program prog, HashSet<String> fnStack, HashSet<String> fnCreated, boolean plain, boolean cpFunctions) {
		ArrayList<Instruction> tmp = new ArrayList<>();
		for( Instruction inst : instSet ) {
			if( inst instanceof FunctionCallCPInstruction && cpFunctions ) {
				FunctionCallCPInstruction finst = (FunctionCallCPInstruction) inst;
				createDeepCopyFunctionProgramBlock( finst.getNamespace(),
					finst.getFunctionName(), pid, IDPrefix, prog, fnStack, fnCreated, plain );
			}
			tmp.add( cloneInstruction( inst, pid, plain, cpFunctions ) );
		}
		return tmp;
	}

	public static Instruction cloneInstruction( Instruction oInst, long pid, boolean plain, boolean cpFunctions ) 
	{
		Instruction inst = null;
		String tmpString = oInst.toString();
		
		try
		{
			if( oInst instanceof CPInstruction || oInst instanceof SPInstruction 
				|| oInst instanceof GPUInstruction ) {
				if( oInst instanceof FunctionCallCPInstruction && cpFunctions ) {
					FunctionCallCPInstruction tmp = (FunctionCallCPInstruction) oInst;
					if( !plain ) {
						//safe replacement because target variables might include the function name
						//note: this is no update-in-place in order to keep the original function name as basis
						tmpString = tmp.updateInstStringFunctionName(tmp.getFunctionName(), tmp.getFunctionName() + Lop.CP_CHILD_THREAD+pid);
					}
					//otherwise: preserve function name
				}
				inst = InstructionParser.parseSingleInstruction(tmpString);
			}
			else
				throw new DMLRuntimeException("Failed to clone instruction: "+oInst);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		//save replacement of thread id references in instructions
		inst = saveReplaceThreadID( inst, Lop.CP_ROOT_THREAD_ID, Lop.CP_CHILD_THREAD+pid);
		
		return inst;
	}

	public static StatementBlock createStatementBlockCopy( StatementBlock sb, long pid, boolean plain, boolean forceDeepCopy )
	{
		StatementBlock ret = null;
		
		try
		{
			if( ConfigurationManager.getCompilerConfigFlag(ConfigType.ALLOW_PARALLEL_DYN_RECOMPILATION) 
				&& sb != null  //forced deep copy for function recompilation
				&& (Recompiler.requiresRecompilation( sb.getHops() ) || forceDeepCopy)  )
			{
				//create new statement (shallow copy livein/liveout for recompile, line numbers for explain)
				ret = new StatementBlock();
				ret.setDMLProg(sb.getDMLProg());
				ret.setParseInfo(sb);
				ret.setLiveIn( sb.liveIn() );
				ret.setLiveOut( sb.liveOut() );
				ret.setUpdatedVariables( sb.variablesUpdated() );
				ret.setReadVariables( sb.variablesRead() );
				
				//deep copy hops dag for concurrent recompile
				ArrayList<Hop> hops = Recompiler.deepCopyHopsDag( sb.getHops() );
				if( !plain )
					Recompiler.updateFunctionNames( hops, pid );
				ret.setHops( hops );
				ret.updateRecompilationFlag();
			}
			else {
				ret = sb;
			}
		}
		catch( Exception ex ) {
			throw new DMLRuntimeException( ex );
		}
		
		return ret;
	}

	public static IfStatementBlock createIfStatementBlockCopy( IfStatementBlock sb, long pid, boolean plain, boolean forceDeepCopy ) 
	{
		IfStatementBlock ret = null;
		
		try
		{
			if( ConfigurationManager.getCompilerConfigFlag(ConfigType.ALLOW_PARALLEL_DYN_RECOMPILATION) 
				&& sb != null //forced deep copy for function recompile
				&& (Recompiler.requiresRecompilation( sb.getPredicateHops() ) || forceDeepCopy)  )
			{
				//create new statement (shallow copy livein/liveout for recompile, line numbers for explain)
				ret = new IfStatementBlock();
				ret.setDMLProg(sb.getDMLProg());
				ret.setParseInfo(sb);
				ret.setLiveIn( sb.liveIn() );
				ret.setLiveOut( sb.liveOut() );
				ret.setUpdatedVariables( sb.variablesUpdated() );
				ret.setReadVariables( sb.variablesRead() );
				
				//shallow copy child statements
				ret.setStatements( sb.getStatements() );
				
				//deep copy predicate hops dag for concurrent recompile
				Hop hops = Recompiler.deepCopyHopsDag( sb.getPredicateHops() );
				ret.setPredicateHops( hops );
				ret.updatePredicateRecompilationFlag();
			}
			else {
				ret = sb;
			}
		}
		catch( Exception ex ) {
			throw new DMLRuntimeException( ex );
		}
		
		return ret;
	}

	public static WhileStatementBlock createWhileStatementBlockCopy( WhileStatementBlock sb, long pid, boolean plain, boolean forceDeepCopy ) 
	{
		WhileStatementBlock ret = null;
		
		try
		{
			if( ConfigurationManager.getCompilerConfigFlag(ConfigType.ALLOW_PARALLEL_DYN_RECOMPILATION) 
				&& sb != null  //forced deep copy for function recompile
				&& (Recompiler.requiresRecompilation( sb.getPredicateHops() ) || forceDeepCopy)  )
			{
				//create new statement (shallow copy livein/liveout for recompile, line numbers for explain)
				ret = new WhileStatementBlock();
				ret.setDMLProg(sb.getDMLProg());
				ret.setParseInfo(sb);
				ret.setLiveIn( sb.liveIn() );
				ret.setLiveOut( sb.liveOut() );
				ret.setUpdatedVariables( sb.variablesUpdated() );
				ret.setReadVariables( sb.variablesRead() );
				ret.setUpdateInPlaceVars( sb.getUpdateInPlaceVars() );
				
				//shallow copy child statements
				ret.setStatements( sb.getStatements() );
				
				//deep copy predicate hops dag for concurrent recompile
				Hop hops = Recompiler.deepCopyHopsDag( sb.getPredicateHops() );
				ret.setPredicateHops( hops );
				ret.updatePredicateRecompilationFlag();
			}
			else {
				ret = sb;
			}
		}
		catch( Exception ex ) {
			throw new DMLRuntimeException( ex );
		}
		
		return ret;
	}

	public static ForStatementBlock createForStatementBlockCopy( ForStatementBlock sb, long pid, boolean plain, boolean forceDeepCopy ) 
	{
		ForStatementBlock ret = null;
		
		try
		{
			if( ConfigurationManager.getCompilerConfigFlag(ConfigType.ALLOW_PARALLEL_DYN_RECOMPILATION) 
				&& sb != null 
				&& ( Recompiler.requiresRecompilation(sb.getFromHops()) ||
					 Recompiler.requiresRecompilation(sb.getToHops()) ||
					 Recompiler.requiresRecompilation(sb.getIncrementHops()) ||
					 forceDeepCopy )  )
			{
				ret = (sb instanceof ParForStatementBlock) ? new ParForStatementBlock() : new ForStatementBlock();
				
				//create new statement (shallow copy livein/liveout for recompile, line numbers for explain)
				ret.setDMLProg(sb.getDMLProg());
				ret.setParseInfo(sb);
				ret.setLiveIn( sb.liveIn() );
				ret.setLiveOut( sb.liveOut() );
				ret.setUpdatedVariables( sb.variablesUpdated() );
				ret.setReadVariables( sb.variablesRead() );
				ret.setUpdateInPlaceVars( sb.getUpdateInPlaceVars() );
				
				//shallow copy child statements
				ret.setStatements( sb.getStatements() );
				
				//deep copy predicate hops dag for concurrent recompile
				if( sb.requiresFromRecompilation() ){
					Hop hops = Recompiler.deepCopyHopsDag( sb.getFromHops() );
					ret.setFromHops( hops );
				}
				if( sb.requiresToRecompilation() ){
					Hop hops = Recompiler.deepCopyHopsDag( sb.getToHops() );
					ret.setToHops( hops );
				}
				if( sb.requiresIncrementRecompilation() ){
					Hop hops = Recompiler.deepCopyHopsDag( sb.getIncrementHops() );
					ret.setIncrementHops( hops );
				}
				ret.updatePredicateRecompilationFlags();
			}
			else {
				ret = sb;
			}
		}
		catch( Exception ex ) {
			throw new DMLRuntimeException( ex );
		}
		
		return ret;
	}
	
	
	////////////////////////////////
	// SERIALIZATION 
	////////////////////////////////

	public static String serializeSparkPSBody(SparkPSBody body, HashMap<String, byte[]> clsMap) {

		ExecutionContext ec = body.getEc();
		StringBuilder builder = new StringBuilder();
		builder.append(PSBODY_BEGIN);
		builder.append(NEWLINE);

		//handle DMLScript UUID (propagate original uuid for writing to scratch space)
		builder.append(DMLScript.getUUID());
		builder.append(COMPONENTS_DELIM);
		builder.append(NEWLINE);

		//handle DML config
		builder.append(ConfigurationManager.getDMLConfig().serializeDMLConfig());
		builder.append(COMPONENTS_DELIM);
		builder.append(NEWLINE);

		//handle additional configurations
		builder.append(CONF_STATS + "=" + DMLScript.STATISTICS);
		builder.append(COMPONENTS_DELIM);
		builder.append(NEWLINE);

		//handle program
		builder.append(PROG_BEGIN);
		builder.append(NEWLINE);
		builder.append(rSerializeFunctionProgramBlocks(ec.getProgram().getFunctionProgramBlocks(),
			new HashSet<>(ec.getProgram().getFunctionProgramBlocks().keySet()), clsMap));
		builder.append(PROG_END);
		builder.append(NEWLINE);
		builder.append(COMPONENTS_DELIM);
		builder.append(NEWLINE);

		//handle execution context
		builder.append(EC_BEGIN);
		builder.append(serializeExecutionContext(ec));
		builder.append(EC_END);
		builder.append(NEWLINE);
		builder.append(COMPONENTS_DELIM);
		builder.append(NEWLINE);

		//handle program blocks
		builder.append(PBS_BEGIN);
		builder.append(NEWLINE);
		builder.append(rSerializeProgramBlocks(ec.getProgram().getProgramBlocks(), clsMap));
		builder.append(PBS_END);
		builder.append(NEWLINE);
		builder.append(COMPONENTS_DELIM);
		builder.append(NEWLINE);

		builder.append(PSBODY_END);
		return builder.toString();
	}

	public static String serializeParForBody( ParForBody body ) {
		return serializeParForBody(body, new HashMap<String, byte[]>());
	}	
	
	public static String serializeParForBody( ParForBody body, HashMap<String,byte[]> clsMap ) 
	{
		ArrayList<ProgramBlock> pbs = body.getChildBlocks();
		ArrayList<ResultVar> rVnames = body.getResultVariables();
		ExecutionContext ec = body.getEc();
		
		if( pbs.isEmpty() )
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
		sb.append( ConfigurationManager.getDMLConfig().serializeDMLConfig() );
		sb.append( COMPONENTS_DELIM );
		sb.append( NEWLINE );
		
		//handle additional configurations
		sb.append( CONF_STATS + "=" + DMLScript.STATISTICS );
		sb.append( COMPONENTS_DELIM );
		sb.append( NEWLINE );
		
		//handle program
		sb.append(PROG_BEGIN);
		sb.append( NEWLINE );
		sb.append( serializeProgram(prog, pbs, clsMap) );
		sb.append(PROG_END);
		sb.append( NEWLINE );
		sb.append( COMPONENTS_DELIM );
		sb.append( NEWLINE );
		
		//handle result variable names
		sb.append( serializeResultVariables(rVnames) );
		sb.append( COMPONENTS_DELIM );
		
		//handle execution context
		//note: this includes also the symbol table (serialize only the top-level variable map,
		//      (symbol tables for nested/child blocks are created at parse time, on the remote side)
		sb.append(EC_BEGIN);
		sb.append( serializeExecutionContext(ec) );
		sb.append(EC_END);
		sb.append( NEWLINE );
		sb.append( COMPONENTS_DELIM );
		sb.append( NEWLINE );
		
		//handle program blocks
		sb.append(PBS_BEGIN);
		sb.append( NEWLINE );
		sb.append( rSerializeProgramBlocks(pbs, clsMap) );
		sb.append(PBS_END);
		sb.append( NEWLINE );
		
		sb.append( PARFORBODY_END );
		
		return sb.toString();
	}

	private static String serializeProgram( Program prog, ArrayList<ProgramBlock> pbs, HashMap<String, byte[]> clsMap ) {
		//note program contains variables, programblocks and function program blocks 
		//but in order to avoid redundancy, we only serialize function program blocks
		HashMap<String, FunctionProgramBlock> fpb = prog.getFunctionProgramBlocks();
		HashSet<String> cand = new HashSet<>();
		rFindSerializationCandidates(pbs, cand);
		return rSerializeFunctionProgramBlocks( fpb, cand, clsMap );
	}

	private static void rFindSerializationCandidates( ArrayList<ProgramBlock> pbs, HashSet<String> cand ) 
	{
		for( ProgramBlock pb : pbs )
		{
			if( pb instanceof WhileProgramBlock ) {
				WhileProgramBlock wpb = (WhileProgramBlock) pb;
				rFindSerializationCandidates(wpb.getChildBlocks(), cand );
			}
			else if ( pb instanceof ForProgramBlock || pb instanceof ParForProgramBlock ) {
				ForProgramBlock fpb = (ForProgramBlock) pb; 
				rFindSerializationCandidates(fpb.getChildBlocks(), cand);
			}
			else if ( pb instanceof IfProgramBlock ) {
				IfProgramBlock ipb = (IfProgramBlock) pb;
				rFindSerializationCandidates(ipb.getChildBlocksIfBody(), cand);
				if( ipb.getChildBlocksElseBody() != null )
					rFindSerializationCandidates(ipb.getChildBlocksElseBody(), cand);
			}
			else if( pb instanceof BasicProgramBlock ) { 
				BasicProgramBlock bpb = (BasicProgramBlock) pb;
				for( Instruction inst : bpb.getInstructions() )
					if( inst instanceof FunctionCallCPInstruction ) {
						FunctionCallCPInstruction fci = (FunctionCallCPInstruction) inst;
						String fkey = DMLProgram.constructFunctionKey(fci.getNamespace(), fci.getFunctionName());
						if( !cand.contains(fkey) ) { //memoization for multiple calls, recursion
							cand.add( fkey ); //add to candidates
							//investigate chains of function calls
							FunctionProgramBlock fpb = pb.getProgram().getFunctionProgramBlock(fci.getNamespace(), fci.getFunctionName());
							rFindSerializationCandidates(fpb.getChildBlocks(), cand);
						}
					}
			}
		}
	}

	private static String serializeVariables (LocalVariableMap vars) {
		StringBuilder sb = new StringBuilder();
		sb.append(VARS_BEGIN);
		sb.append( vars.serialize() );
		sb.append(VARS_END);
		return sb.toString();
	}

	public static String serializeDataObject(String key, Data dat) 
	{
		// SCHEMA: <name>|<datatype>|<valuetype>|value
		// (scalars are serialize by value, matrices by filename)
		StringBuilder sb = new StringBuilder();
		//prepare data for serialization
		String name = key;
		DataType datatype = dat.getDataType();
		ValueType valuetype = dat.getValueType();
		String value = null;
		String[] metaData = null;
		String[] listData = null;
		switch( datatype )
		{
			case SCALAR:
				ScalarObject so = (ScalarObject) dat;
				//name = so.getName();
				value = so.getStringValue();
				break;
			case MATRIX:
				MatrixObject mo = (MatrixObject) dat;
				MetaDataFormat md = (MetaDataFormat) dat.getMetaData();
				DataCharacteristics dc = md.getDataCharacteristics();
				value = mo.getFileName();
				PartitionFormat partFormat = (mo.getPartitionFormat()!=null) ? new PartitionFormat(
						mo.getPartitionFormat(),mo.getPartitionSize()) : PartitionFormat.NONE;
				metaData = new String[11];
				metaData[0] = String.valueOf( dc.getRows() );
				metaData[1] = String.valueOf( dc.getCols() );
				metaData[2] = String.valueOf( dc.getBlocksize() );
				metaData[3] = String.valueOf( dc.getBlocksize() );
				metaData[4] = String.valueOf( dc.getNonZeros() );
				metaData[5] = InputInfo.inputInfoToString( md.getInputInfo() );
				metaData[6] = OutputInfo.outputInfoToString( md.getOutputInfo() );
				metaData[7] = String.valueOf( partFormat );
				metaData[8] = String.valueOf( mo.getUpdateType() );
				metaData[9] = String.valueOf(mo.isHDFSFileExists());
				metaData[10] = String.valueOf(mo.isCleanupEnabled());
				break;
			case LIST:
				// SCHEMA: <name>|<datatype>|<valuetype>|value|<metadata>|<tab>element1<tab>element2<tab>element3 (this is the list)
				//         (for the element1) <listName-index>|<datatype>|<valuetype>|value
				//         (for the element2) <listName-index>|<datatype>|<valuetype>|value
				ListObject lo = (ListObject) dat;
				value = REF;
				metaData = new String[2];
				metaData[0] = String.valueOf(lo.getLength());
				metaData[1] = lo.getNames() == null ? EMPTY : serializeList(lo.getNames(), ELEMENT_DELIM2);
				listData = new String[lo.getLength()];
				for (int index = 0; index < lo.getLength(); index++) {
					listData[index] = serializeDataObject(name + DASH + index, lo.slice(index));
				}
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
		if( metaData != null )
			for( int i=0; i<metaData.length; i++ ) {
				sb.append(DATA_FIELD_DELIM);
				sb.append(metaData[i]);
			}
		if (listData != null) {
			sb.append(DATA_FIELD_DELIM);
			for (String ld : listData) {
				sb.append(LIST_ELEMENT_DELIM);
				sb.append(ld);
			}
		}
		
		return sb.toString();
	}

	private static String serializeExecutionContext( ExecutionContext ec ) {
		return (ec != null) ? serializeVariables( ec.getVariables() ) : EMPTY;
	}

	@SuppressWarnings("all")
	private static String serializeInstructions( ArrayList<Instruction> inst, HashMap<String, byte[]> clsMap ) 
	{
		StringBuilder sb = new StringBuilder();
		int count = 0;
		for( Instruction linst : inst ) {
			//check that only cp instruction are transmitted 
			if( !( linst instanceof CPInstruction) )
				throw new DMLRuntimeException( NOT_SUPPORTED_SPARK_INSTRUCTION + " " +linst.getClass().getName()+"\n"+linst );
			
			//obtain serialized version of generated classes
			if( linst instanceof SpoofCPInstruction ) {
				Class<?> cla = ((SpoofCPInstruction) linst).getOperatorClass();
				clsMap.put(cla.getName(), CodegenUtils.getClassData(cla.getName()));
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
	 * @param instStr instruction string
	 * @return instruction string with replacements
	 */
	private static String checkAndReplaceLiterals( String instStr )
	{
		String tmp = instStr;
		
		//1) check own delimiters (very unlikely due to special characters)
		if( tmp.contains(COMPONENTS_DELIM) ) {
			tmp = tmp.replaceAll(COMPONENTS_DELIM, ".");
			LOG.warn("Replaced special literal character sequence "+COMPONENTS_DELIM+" with '.'");
		}
		
		if( tmp.contains(ELEMENT_DELIM) ) {
			tmp = tmp.replaceAll(ELEMENT_DELIM, ".");
			LOG.warn("Replaced special literal character sequence "+ELEMENT_DELIM+" with '.'");
		}
			
		if( tmp.contains( LEVELIN ) ){
			tmp = tmp.replaceAll(LEVELIN, "("); // '\\' required if LEVELIN='{' because regex
			LOG.warn("Replaced special literal character sequence "+LEVELIN+" with '('");
		}

		if( tmp.contains(LEVELOUT) ){
			tmp = tmp.replaceAll(LEVELOUT, ")");
			LOG.warn("Replaced special literal character sequence "+LEVELOUT+" with ')'");
		}
		
		//NOTE: DATA_FIELD_DELIM and KEY_VALUE_DELIM not required
		//because those literals cannot occur in critical places.
		
		//2) check end tag of CDATA
		if( tmp.contains(CDATA_END) ){
			tmp = tmp.replaceAll(CDATA_END, "."); //prevent XML parsing issues in job.xml
			LOG.warn("Replaced special literal character sequence "+ CDATA_END +" with '.'");
		}
		
		return tmp;
	}

	private static String serializeStringHashMap(HashMap<String,String> vars) {
		return serializeList(vars.entrySet().stream().map(e ->
			e.getKey()+KEY_VALUE_DELIM+e.getValue()).collect(Collectors.toList()));
	}

	public static String serializeResultVariables( List<ResultVar> vars) {
		return serializeList(vars.stream().map(v -> v._isAccum ?
			v._name+"+" : v._name).collect(Collectors.toList()));
	}
	
	public static String serializeList(List<String> elements) {
		return serializeList(elements, ELEMENT_DELIM);
	}
	
	public static String serializeList(List<String> elements, String delim) {
		return StringUtils.join(elements, delim);
	}

	private static String serializeDataIdentifiers(List<DataIdentifier> vars) {
		return serializeList(vars.stream().map(v ->
			serializeDataIdentifier(v)).collect(Collectors.toList()));
	}

	private static String serializeDataIdentifier( DataIdentifier dat ) {
		// SCHEMA: <name>|<datatype>|<valuetype>
		StringBuilder sb = new StringBuilder();
		sb.append(dat.getName());
		sb.append(DATA_FIELD_DELIM);
		sb.append(dat.getDataType());
		sb.append(DATA_FIELD_DELIM);
		sb.append(dat.getValueType());
		return sb.toString();
	}

	private static String rSerializeFunctionProgramBlocks(HashMap<String,FunctionProgramBlock> pbs, HashSet<String> cand, HashMap<String, byte[]> clsMap) {
		StringBuilder sb = new StringBuilder();
		int count = 0;
		for( Entry<String,FunctionProgramBlock> pb : pbs.entrySet() ) {
			if( !cand.contains(pb.getKey()) ) //skip function not included in the parfor body
				continue;
			if( count>0 ) {
				sb.append( ELEMENT_DELIM );
				sb.append( NEWLINE );
			}
			sb.append( pb.getKey() );
			sb.append( KEY_VALUE_DELIM );
			sb.append( rSerializeProgramBlock(pb.getValue(), clsMap) );
			count++;
		}
		sb.append(NEWLINE);
		return sb.toString();
	}

	private static String rSerializeProgramBlocks(ArrayList<ProgramBlock> pbs, HashMap<String, byte[]> clsMap) {
		StringBuilder sb = new StringBuilder();
		int count = 0;
		for( ProgramBlock pb : pbs ) {
			if( count>0 ) {
				sb.append( ELEMENT_DELIM );
				sb.append(NEWLINE);
			}
			sb.append( rSerializeProgramBlock(pb, clsMap) );
			count++;
		}
		return sb.toString();
	}

	private static String rSerializeProgramBlock( ProgramBlock pb, HashMap<String, byte[]> clsMap ) {
		StringBuilder sb = new StringBuilder();
		
		//handle header
		if( pb instanceof WhileProgramBlock ) 
			sb.append(PB_WHILE);
		else if ( pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock) )
			sb.append(PB_FOR);
		else if ( pb instanceof ParForProgramBlock )
			sb.append(PB_PARFOR);
		else if ( pb instanceof IfProgramBlock )
			sb.append(PB_IF);
		else if ( pb instanceof FunctionProgramBlock )
			sb.append(PB_FC);
		else //all generic program blocks
			sb.append(PB_BEGIN);
		
		//handle body
		if( pb instanceof WhileProgramBlock ) {
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			sb.append(INST_BEGIN);
			sb.append( serializeInstructions( wpb.getPredicate(), clsMap ) );
			sb.append(INST_END);
			sb.append( COMPONENTS_DELIM );
			sb.append(PBS_BEGIN);
			sb.append( rSerializeProgramBlocks( wpb.getChildBlocks(), clsMap) );
			sb.append(PBS_END);
		}
		else if ( pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock ) ) {
			ForProgramBlock fpb = (ForProgramBlock) pb; 
			sb.append( fpb.getIterVar() );
			sb.append( COMPONENTS_DELIM );
			sb.append(INST_BEGIN);
			sb.append( serializeInstructions( fpb.getFromInstructions(), clsMap ) );
			sb.append(INST_END);
			sb.append( COMPONENTS_DELIM );
			sb.append(INST_BEGIN);
			sb.append( serializeInstructions(fpb.getToInstructions(), clsMap) );
			sb.append(INST_END);
			sb.append( COMPONENTS_DELIM );
			sb.append(INST_BEGIN);
			sb.append( serializeInstructions(fpb.getIncrementInstructions(), clsMap) );
			sb.append(INST_END);
			sb.append( COMPONENTS_DELIM );
			sb.append(PBS_BEGIN);
			sb.append( rSerializeProgramBlocks(fpb.getChildBlocks(), clsMap) );
			sb.append(PBS_END);
		}
		else if ( pb instanceof ParForProgramBlock ) {
			ParForProgramBlock pfpb = (ParForProgramBlock) pb; 
			
			//check for nested remote ParFOR
			if( PExecMode.valueOf( pfpb.getParForParams().get( ParForStatementBlock.EXEC_MODE )) == PExecMode.REMOTE_SPARK )
				throw new DMLRuntimeException( NOT_SUPPORTED_SPARK_PARFOR );
			
			sb.append( pfpb.getIterVar() );
			sb.append( COMPONENTS_DELIM );
			sb.append( serializeResultVariables( pfpb.getResultVariables()) );
			sb.append( COMPONENTS_DELIM );
			sb.append( serializeStringHashMap( pfpb.getParForParams()) ); //parameters of nested parfor
			sb.append( COMPONENTS_DELIM );
			sb.append(INST_BEGIN);
			sb.append( serializeInstructions(pfpb.getFromInstructions(), clsMap) );
			sb.append(INST_END);
			sb.append( COMPONENTS_DELIM );
			sb.append(INST_BEGIN);
			sb.append( serializeInstructions(pfpb.getToInstructions(), clsMap) );
			sb.append(INST_END);
			sb.append( COMPONENTS_DELIM );
			sb.append(INST_BEGIN);
			sb.append( serializeInstructions(pfpb.getIncrementInstructions(), clsMap) );
			sb.append(INST_END);
			sb.append( COMPONENTS_DELIM );
			sb.append(PBS_BEGIN);
			sb.append( rSerializeProgramBlocks( pfpb.getChildBlocks(), clsMap ) );
			sb.append(PBS_END);
		}
		else if ( pb instanceof IfProgramBlock ) {
			IfProgramBlock ipb = (IfProgramBlock) pb;
			sb.append(INST_BEGIN);
			sb.append( serializeInstructions(ipb.getPredicate(), clsMap) );
			sb.append(INST_END);
			sb.append( COMPONENTS_DELIM );
			sb.append(PBS_BEGIN);
			sb.append( rSerializeProgramBlocks(ipb.getChildBlocksIfBody(), clsMap) );
			sb.append(PBS_END);
			sb.append( COMPONENTS_DELIM );
			sb.append(PBS_BEGIN);
			sb.append( rSerializeProgramBlocks(ipb.getChildBlocksElseBody(), clsMap) );
			sb.append(PBS_END);
		}
		else if( pb instanceof FunctionProgramBlock ) {
			FunctionProgramBlock fpb = (FunctionProgramBlock) pb;
			sb.append( serializeDataIdentifiers( fpb.getInputParams() ) );
			sb.append( COMPONENTS_DELIM );
			sb.append( serializeDataIdentifiers( fpb.getOutputParams() ) );
			sb.append( COMPONENTS_DELIM );
			sb.append(PBS_BEGIN);
			sb.append( rSerializeProgramBlocks(fpb.getChildBlocks(), clsMap) );
			sb.append(PBS_END);
			sb.append( COMPONENTS_DELIM );
		}
		else if( pb instanceof BasicProgramBlock ) {
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			sb.append(INST_BEGIN);
			sb.append( serializeInstructions(
				bpb.getInstructions(), clsMap) );
			sb.append(INST_END);
		}
		
		//handle end
		sb.append(PB_END);
		
		return sb.toString();
	}

	
	////////////////////////////////
	// PARSING 
	////////////////////////////////

	public static SparkPSBody parseSparkPSBody(String in, int id) {
		SparkPSBody body = new SparkPSBody();

		//header elimination
		String tmpin = in.replaceAll(NEWLINE, ""); //normalization
		tmpin = tmpin.substring(PSBODY_BEGIN.length(), tmpin.length() - PSBODY_END.length()); //remove start/end
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(tmpin, COMPONENTS_DELIM);

		//handle DMLScript UUID (NOTE: set directly in DMLScript)
		//(master UUID is used for all nodes (in order to simply cleanup))
		DMLScript.setUUID(st.nextToken());

		//handle DML config (NOTE: set directly in ConfigurationManager)
		handleDMLConfig(st.nextToken());

		//handle additional configs
		parseAndSetAdditionalConfigurations(st.nextToken());

		//handle program
		Program prog = parseProgram(st.nextToken(), id);

		//handle execution context
		ExecutionContext ec = parseExecutionContext(st.nextToken(), prog);
		ec.setProgram(prog);

		//handle program blocks
		String spbs = st.nextToken();
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(spbs, prog, id);
		prog.getProgramBlocks().addAll(pbs);

		body.setEc(ec);
		return body;
	}

	public static ParForBody parseParForBody( String in, int id ) {
		return parseParForBody(in, id, false);
	}
	
	public static ParForBody parseParForBody( String in, int id, boolean inSpark ) {
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
		JobConf job = ConfigurationManager.getCachedJobConf();
		if( !InfrastructureAnalyzer.isLocalMode(job) ) {
			handleDMLConfig(confStr);
		}
		
		//handle additional configs
		String aconfs = st.nextToken();
		if( !inSpark )
			parseAndSetAdditionalConfigurations( aconfs );
		
		//handle program
		String progStr = st.nextToken();
		Program prog = parseProgram( progStr, id ); 
		
		//handle result variable names
		String rvarStr = st.nextToken();
		ArrayList<ResultVar> rvars = parseResultVariables(rvarStr);
		body.setResultVariables(rvars);
		
		//handle execution context
		String ecStr = st.nextToken();
		ExecutionContext ec = parseExecutionContext( ecStr, prog );
		
		//handle program blocks
		String spbs = st.nextToken();
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(spbs, prog, id);
		
		body.setChildBlocks( pbs );
		body.setEc( ec );
		
		return body;
	}

	private static void handleDMLConfig(String confStr) {
		if(confStr != null && !confStr.trim().isEmpty()) {
			DMLConfig dmlconf = DMLConfig.parseDMLConfig(confStr);
			CompilerConfig cconf = OptimizerUtils.constructCompilerConfig(dmlconf);
			ConfigurationManager.setLocalConfig(dmlconf);
			ConfigurationManager.setLocalConfig(cconf);
		}
	}

	public static Program parseProgram( String in, int id ) {
		String lin = in.substring( PROG_BEGIN.length(),in.length()- PROG_END.length()).trim();
		Program prog = new Program();
		HashMap<String,FunctionProgramBlock> fc = parseFunctionProgramBlocks(lin, prog, id);
		for( Entry<String,FunctionProgramBlock> e : fc.entrySet() ) {
			String[] keypart = e.getKey().split( Program.KEY_DELIM );
			String namespace = keypart[0];
			String name      = keypart[1];
			prog.addFunctionProgramBlock(namespace, name, e.getValue());
		}
		return prog;
	}

	private static LocalVariableMap parseVariables(String in) {
		LocalVariableMap ret = null;
		if( in.length()> VARS_BEGIN.length() + VARS_END.length()) {
			String varStr = in.substring( VARS_BEGIN.length(),in.length()- VARS_END.length()).trim();
			ret = LocalVariableMap.deserialize(varStr);
		}
		else { //empty input symbol table
			ret = new LocalVariableMap();
		}
		return ret;
	}

	private static HashMap<String,FunctionProgramBlock> parseFunctionProgramBlocks( String in, Program prog, int id ) {
		HashMap<String,FunctionProgramBlock> ret = new HashMap<>();
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer( in, ELEMENT_DELIM );
		while( st.hasMoreTokens() ) {
			String lvar  = st.nextToken(); //with ID = CP_CHILD_THREAD+id for current use
			//put first copy into prog (for direct use)
			int index = lvar.indexOf( KEY_VALUE_DELIM );
			String tmp1 = lvar.substring(0, index); // + CP_CHILD_THREAD+id;
			String tmp2 = lvar.substring(index + 1);
			ret.put(tmp1, (FunctionProgramBlock)rParseProgramBlock(tmp2, prog, id));
		}
		return ret;
	}

	private static ArrayList<ProgramBlock> rParseProgramBlocks(String in, Program prog, int id) {
		ArrayList<ProgramBlock> pbs = new ArrayList<>();
		String tmpdata = in.substring(PBS_BEGIN.length(),in.length()- PBS_END.length());
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(tmpdata, ELEMENT_DELIM);
		while( st.hasMoreTokens() )
			pbs.add( rParseProgramBlock( st.nextToken(), prog, id ) );
		return pbs;
	}

	private static ProgramBlock rParseProgramBlock( String in, Program prog, int id ) {
		ProgramBlock pb = null;
		if( in.startsWith(PB_WHILE) )
			pb = rParseWhileProgramBlock( in, prog, id );
		else if ( in.startsWith(PB_FOR) )
			pb = rParseForProgramBlock( in, prog, id );
		else if ( in.startsWith(PB_PARFOR) )
			pb = rParseParForProgramBlock( in, prog, id );
		else if ( in.startsWith(PB_IF) )
			pb = rParseIfProgramBlock( in, prog, id );
		else if ( in.startsWith(PB_FC) )
			pb = rParseFunctionProgramBlock( in, prog, id );
 		else if ( in.startsWith(PB_BEGIN) )
			pb = rParseGenericProgramBlock( in, prog, id );
		else 
			throw new DMLRuntimeException( NOT_SUPPORTED_PB+" "+in );
		return pb;
	}

	private static WhileProgramBlock rParseWhileProgramBlock( String in, Program prog, int id ) {
		String lin = in.substring( PB_WHILE.length(),in.length()- PB_END.length());
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(lin, COMPONENTS_DELIM);
		
		//predicate instructions
		ArrayList<Instruction> inst = parseInstructions(st.nextToken(),id);
		
		//program blocks
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(st.nextToken(), prog, id);
		
		WhileProgramBlock wpb = new WhileProgramBlock(prog,inst);
		wpb.setChildBlocks(pbs);
		return wpb;
	}

	private static ForProgramBlock rParseForProgramBlock( String in, Program prog, int id ) {
		String lin = in.substring( PB_FOR.length(),in.length()- PB_END.length());
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(lin, COMPONENTS_DELIM);
		
		//inputs
		String iterVar = st.nextToken();
		
		//instructions
		ArrayList<Instruction> from = parseInstructions(st.nextToken(),id);
		ArrayList<Instruction> to = parseInstructions(st.nextToken(),id);
		ArrayList<Instruction> incr = parseInstructions(st.nextToken(),id);
		
		//program blocks
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(st.nextToken(), prog, id);
		
		ForProgramBlock fpb = new ForProgramBlock(prog, iterVar);
		fpb.setFromInstructions(from);
		fpb.setToInstructions(to);
		fpb.setIncrementInstructions(incr);
		fpb.setChildBlocks(pbs);
		
		return fpb;
	}

	private static ParForProgramBlock rParseParForProgramBlock( String in, Program prog, int id ) {
		String lin = in.substring( PB_PARFOR.length(),in.length()- PB_END.length());
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(lin, COMPONENTS_DELIM);
		
		//inputs
		String iterVar = st.nextToken();
		ArrayList<ResultVar> resultVars = parseResultVariables(st.nextToken());
		HashMap<String,String> params = parseStringHashMap(st.nextToken());
		
		//instructions 
		ArrayList<Instruction> from = parseInstructions(st.nextToken(), 0);
		ArrayList<Instruction> to = parseInstructions(st.nextToken(), 0);
		ArrayList<Instruction> incr = parseInstructions(st.nextToken(), 0);
		
		//program blocks //reset id to preinit state, replaced during exec
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(st.nextToken(), prog, 0); 
		
		ParForProgramBlock pfpb = new ParForProgramBlock(id, prog, iterVar, params, resultVars);
		pfpb.disableOptimization(); //already done in top-level parfor
		pfpb.setFromInstructions(from);
		pfpb.setToInstructions(to);
		pfpb.setIncrementInstructions(incr);
		pfpb.setChildBlocks(pbs);
		
		return pfpb;
	}

	private static IfProgramBlock rParseIfProgramBlock( String in, Program prog, int id ) {
		String lin = in.substring( PB_IF.length(),in.length()- PB_END.length());
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(lin, COMPONENTS_DELIM);
		
		//predicate instructions
		ArrayList<Instruction> inst = parseInstructions(st.nextToken(),id);
		
		//program blocks: if and else
		ArrayList<ProgramBlock> pbs1 = rParseProgramBlocks(st.nextToken(), prog, id);
		ArrayList<ProgramBlock> pbs2 = rParseProgramBlocks(st.nextToken(), prog, id);
		
		IfProgramBlock ipb = new IfProgramBlock(prog,inst);
		ipb.setChildBlocksIfBody(pbs1);
		ipb.setChildBlocksElseBody(pbs2);
		
		return ipb;
	}

	private static FunctionProgramBlock rParseFunctionProgramBlock( String in, Program prog, int id ) {
		String lin = in.substring( PB_FC.length(),in.length()- PB_END.length());
		HierarchyAwareStringTokenizer st = new HierarchyAwareStringTokenizer(lin, COMPONENTS_DELIM);
		
		//inputs and outputs
		ArrayList<DataIdentifier> dat1 = parseDataIdentifiers(st.nextToken());
		ArrayList<DataIdentifier> dat2 = parseDataIdentifiers(st.nextToken());

		//program blocks
		ArrayList<ProgramBlock> pbs = rParseProgramBlocks(st.nextToken(), prog, id);

		ArrayList<DataIdentifier> tmp1 = new ArrayList<>(dat1);
		ArrayList<DataIdentifier> tmp2 = new ArrayList<>(dat2);
		FunctionProgramBlock fpb = new FunctionProgramBlock(prog, tmp1, tmp2);
		fpb.setChildBlocks(pbs);
		
		return fpb;
	}

	private static ProgramBlock rParseGenericProgramBlock( String in, Program prog, int id ) {
		String lin = in.substring( PB_BEGIN.length(),in.length()- PB_END.length());
		StringTokenizer st = new StringTokenizer(lin,COMPONENTS_DELIM);
		BasicProgramBlock pb = new BasicProgramBlock(prog);
		pb.setInstructions(parseInstructions(st.nextToken(),id));
		return pb;
	}

	private static ArrayList<Instruction> parseInstructions( String in, int id ) {
		ArrayList<Instruction> insts = new ArrayList<>();
		String lin = in.substring( INST_BEGIN.length(),in.length()- INST_END.length());
		StringTokenizer st = new StringTokenizer(lin, ELEMENT_DELIM);
		while(st.hasMoreTokens()) {
			//Note that at this point only CP instructions and External function instruction can occur
			String instStr = st.nextToken();
			try {
				Instruction tmpinst = CPInstructionParser.parseSingleInstruction(instStr);
				tmpinst = saveReplaceThreadID(tmpinst, Lop.CP_ROOT_THREAD_ID, Lop.CP_CHILD_THREAD+id );
				insts.add( tmpinst );
			}
			catch(Exception ex) {
				throw new DMLRuntimeException("Failed to parse instruction: " + instStr, ex);
			}
		}
		return insts;
	}
	
	private static ArrayList<ResultVar> parseResultVariables(String in) {
		ArrayList<ResultVar> ret = new ArrayList<>();
		for(String var : parseStringArrayList(in)) {
			boolean accum = var.endsWith("+");
			ret.add(new ResultVar(accum ? var.substring(0, var.length()-1) : var, accum));
		}
		return ret;
	}

	private static HashMap<String,String> parseStringHashMap( String in ) {
		HashMap<String,String> vars = new HashMap<>();
		StringTokenizer st = new StringTokenizer(in,ELEMENT_DELIM);
		while( st.hasMoreTokens() ) {
			String lin = st.nextToken();
			int index = lin.indexOf( KEY_VALUE_DELIM );
			String tmp1 = lin.substring(0, index);
			String tmp2 = lin.substring(index + 1);
			vars.put(tmp1, tmp2);
		}
		return vars;
	}
	
	private static ArrayList<String> parseStringArrayList(String in) {
		return parseStringArrayList(in, ELEMENT_DELIM);
	}

	private static ArrayList<String> parseStringArrayList(String in, String delim) {
		StringTokenizer st = new StringTokenizer(in, delim);
		ArrayList<String> vars = new ArrayList<>(st.countTokens());
		while( st.hasMoreTokens() )
			vars.add(st.nextToken());
		return vars;
	}

	private static ArrayList<DataIdentifier> parseDataIdentifiers( String in ) {
		ArrayList<DataIdentifier> vars = new ArrayList<>();
		StringTokenizer st = new StringTokenizer(in, ELEMENT_DELIM);
		while( st.hasMoreTokens() ) {
			String tmp = st.nextToken();
			DataIdentifier dat = parseDataIdentifier( tmp );
			vars.add(dat);
		}
		return vars;
	}

	private static DataIdentifier parseDataIdentifier( String in ) {
		StringTokenizer st = new StringTokenizer(in, DATA_FIELD_DELIM);
		DataIdentifier dat = new DataIdentifier(st.nextToken());
		dat.setDataType(DataType.valueOf(st.nextToken()));
		dat.setValueType(ValueType.valueOf(st.nextToken()));
		return dat;
	}
	
	/**
	 * NOTE: MRJobConfiguration cannot be used for the general case because program blocks and
	 * related symbol tables can be hierarchically structured.
	 * 
	 * @param in data object as string
	 * @return array of objects
	 */
	public static Object[] parseDataObject(String in) {
		Object[] ret = new Object[2];
	
		StringTokenizer st = new StringTokenizer(in, DATA_FIELD_DELIM );
		String name = st.nextToken();
		DataType datatype = DataType.valueOf( st.nextToken() );
		ValueType valuetype = ValueType.valueOf( st.nextToken() );
		String valString = st.hasMoreTokens() ? st.nextToken() : "";
		Data dat = null;
		switch( datatype )
		{
			case SCALAR: {
				switch ( valuetype ) {
					case INT64:     dat = new IntObject(Long.parseLong(valString)); break;
					case FP64:  dat = new DoubleObject(Double.parseDouble(valString)); break;
					case BOOLEAN: dat = new BooleanObject(Boolean.parseBoolean(valString)); break;
					case STRING:  dat = new StringObject(valString); break;
					default:
						throw new DMLRuntimeException("Unable to parse valuetype "+valuetype);
				}
				break;
			}
			case MATRIX: {
				MatrixObject mo = new MatrixObject(valuetype,valString);
				long rows = Long.parseLong( st.nextToken() );
				long cols = Long.parseLong( st.nextToken() );
				int blen = Integer.parseInt( st.nextToken() );
				long nnz = Long.parseLong( st.nextToken() );
				InputInfo iin = InputInfo.stringToInputInfo( st.nextToken() );
				OutputInfo oin = OutputInfo.stringToOutputInfo( st.nextToken() );
				PartitionFormat partFormat = PartitionFormat.valueOf( st.nextToken() );
				UpdateType inplace = UpdateType.valueOf( st.nextToken() );
				MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, blen, nnz);
				MetaDataFormat md = new MetaDataFormat( mc, oin, iin );
				mo.setMetaData( md );
				if( partFormat._dpf != PDataPartitionFormat.NONE )
					mo.setPartitioned( partFormat._dpf, partFormat._N );
				mo.setUpdateType(inplace);
				mo.setHDFSFileExists(Boolean.valueOf(st.nextToken()));
				mo.enableCleanup(Boolean.valueOf(st.nextToken()));
				dat = mo;
				break;
			}
			case LIST:
				int size = Integer.parseInt(st.nextToken());
				String namesStr = st.nextToken();
				List<String> names = namesStr.equals(EMPTY) ? null :
					parseStringArrayList(namesStr, ELEMENT_DELIM2);
				List<Data> data = new ArrayList<>(size);
				st.nextToken(LIST_ELEMENT_DELIM);
				for (int i = 0; i < size; i++) {
					String dataStr = st.nextToken();
					Object[] obj = parseDataObject(dataStr);
					data.add((Data) obj[1]);
				}
				dat = new ListObject(data, names);
				break;
			default:
				throw new DMLRuntimeException("Unable to parse datatype "+datatype);
		}
		
		ret[0] = name;
		ret[1] = dat;
		return ret;
	}

	private static ExecutionContext parseExecutionContext(String in, Program prog) {
		ExecutionContext ec = null;
		String lin = in.substring(EC_BEGIN.length(),in.length()- EC_END.length()).trim();
		if( !lin.equals( EMPTY ) ) {
			LocalVariableMap vars = parseVariables(lin);
			ec = ExecutionContextFactory.createContext( false, prog );
			ec.setVariables(vars);
		}
		return ec;
	}
	
	private static void parseAndSetAdditionalConfigurations(String conf) {
		String[] statsFlag = conf.split("=");
		DMLScript.STATISTICS = Boolean.parseBoolean(statsFlag[1]);
	}

	//////////
	// CUSTOM SAFE LITERAL REPLACEMENT
	
	/**
	 * In-place replacement of thread ids in filenames, functions names etc
	 * 
	 * @param inst instruction
	 * @param pattern ?
	 * @param replacement string replacement
	 * @return instruction
	 */
	private static Instruction saveReplaceThreadID( Instruction inst, String pattern, String replacement ) {
		if ( inst instanceof VariableCPInstruction ) { //createvar, setfilename
			//update in-memory representation
			inst.updateInstructionThreadID(pattern, replacement);
		}
		//NOTE> //Rand, seq in CP not required
		return inst;
	}
	
	public static String saveReplaceFilenameThreadID(String fname, String pattern, String replace) {
		//save replace necessary in order to account for the possibility that read variables have our prefix in the absolute path
		//replace the last match only, because (1) we have at most one _t0 and (2) always concatenated to the end.
		int pos = fname.lastIndexOf(pattern);
		return ( pos < 0 ) ? fname : fname.substring(0, pos)
			+ replace + fname.substring(pos+pattern.length());
	}
	
	
	//////////
	// CUSTOM HIERARCHICAL TOKENIZER
	
	
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
		
		public HierarchyAwareStringTokenizer( String in, String delim ) {
			//super(in);
			_str = in;
			_del = delim;
			_off = delim.length();
		}

		public boolean hasMoreTokens() {
			return (_str.length() > 0);
		}

		public String nextToken() {
			int nextDelim = determineNextSameLevelIndexOf(_str, _del);
			String token = null;
			if(nextDelim < 0) {
				nextDelim = _str.length();
				_off = 0;
			}
			token = _str.substring(0,nextDelim);
			_str = _str.substring( nextDelim + _off );
			return token;
		}
		
		private static int determineNextSameLevelIndexOf( String data, String pattern  )
		{
			String tmpdata = data;
			int index      = 0;
			int count      = 0;
			int off=0,i1,i2,i3,min;
			
			while(true) {
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
				else if( i2 == min ) {
					count++;
					off = LEVELIN.length();
				}
				else if( i3 == min ) {
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

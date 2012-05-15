package dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import dml.lops.LopProperties;
import dml.lops.Lops;
import dml.lops.compile.Dag;
import dml.runtime.controlprogram.CVProgramBlock;
import dml.runtime.controlprogram.ELProgramBlock;
import dml.runtime.controlprogram.ELUseProgramBlock;
import dml.runtime.controlprogram.ExternalFunctionProgramBlock;
import dml.runtime.controlprogram.ExternalFunctionProgramBlockCP;
import dml.runtime.controlprogram.ForProgramBlock;
import dml.runtime.controlprogram.FunctionProgramBlock;
import dml.runtime.controlprogram.IfProgramBlock;
import dml.runtime.controlprogram.ParForProgramBlock;
import dml.runtime.controlprogram.Program;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.controlprogram.WhileProgramBlock;
import dml.runtime.controlprogram.parfor.ProgramConverter;
import dml.runtime.instructions.Instruction;
import dml.utils.DMLException;
import dml.utils.LanguageException;
import dml.utils.LopsException;
import dml.utils.configuration.DMLConfig;


public class DMLProgram {

	private ArrayList<StatementBlock> _blocks;
	private HashMap<String, FunctionStatementBlock> _functionBlocks;
	private HashMap<String,DMLProgram> _namespaces;
	public static String DEFAULT_NAMESPACE = ".defaultNS";
	
	public DMLProgram(){
		_blocks = new ArrayList<StatementBlock>();
		_functionBlocks = new HashMap<String,FunctionStatementBlock>();
		_namespaces = new HashMap<String,DMLProgram>();
		_namespaces.put(DMLProgram.DEFAULT_NAMESPACE,this);
	}
	
	public HashMap<String,DMLProgram> getNamespaces(){
		return _namespaces;
	}
	
	public void addStatementBlock(StatementBlock b, int pos) {
		_blocks.add(pos,b) ;
	}
	
	public void addStatementBlock(StatementBlock b){
		_blocks.add(b);
	}
	
	public int getNumStatementBlocks(){
		return _blocks.size();
	}
	
	/**
	 * getFunctionStatementBlock: retrieve function statement block for specified function in specified namespace
	 * @param namespaceKey namespace name
	 * @param functionName function name
	 * @return the function statementblock for the specified function in the specified namespace
	 * @throws LanguageException 
	 */
	public FunctionStatementBlock getFunctionStatementBlock(String namespaceKey, String functionName) throws LanguageException {
		DMLProgram namespaceProgram = this.getNamespaces().get(namespaceKey);
		if (namespaceProgram == null)
			throw new LanguageException("namespace " + namespaceKey + " is underfined");
	
		// for the namespace DMLProgram, get the specified function (if exists) in its current namespace
		FunctionStatementBlock retVal = namespaceProgram._functionBlocks.get(functionName);
		if (retVal == null)
			throw new LanguageException("function " + functionName + " is not defined in namespace " + namespaceKey);
		return retVal;
	}
	
	public HashMap<String, FunctionStatementBlock> getFunctionStatementBlocks(String namespaceKey) throws LanguageException{
		DMLProgram namespaceProgram = this.getNamespaces().get(namespaceKey);
		if (namespaceProgram == null)
			throw new LanguageException("namespace " + namespaceKey + " is underfined");
			
		// for the namespace DMLProgram, get the functions in its current namespace
		return namespaceProgram._functionBlocks;
	}
	
	public ArrayList<StatementBlock> getBlocks(){
		return _blocks;
	}
	
	public void setBlocks(ArrayList<StatementBlock> passed){
		_blocks = passed;
	}
	
	public StatementBlock getStatementBlock(int i){
		return _blocks.get(i);
	}

	public void mergeStatementBlocks(){
		_blocks = StatementBlock.mergeStatementBlocks(_blocks);
	}
	
	public String toString(){
		StringBuffer sb = new StringBuffer();
		
		// for each namespace, display all functions
		for (String namespaceKey : this.getNamespaces().keySet()){
			
			sb.append("******** NAMESPACE : " + namespaceKey + " ******** \n ");
			DMLProgram namespaceProg = this.getNamespaces().get(namespaceKey);
			
			
			sb.append("**** FUNCTIONS ***** \n");
			sb.append("\n");
			for (FunctionStatementBlock fsb : namespaceProg._functionBlocks.values()){
				sb.append(fsb);
				sb.append("\n");
			}
		
		}
		
		sb.append("******** MAIN SCRIPT BODY ******** \n");
		for (StatementBlock b : _blocks){
			sb.append(b);
			sb.append("\n");
		}
		return sb.toString();
	}
	
	
	public Program getRuntimeProgram(boolean debug, DMLConfig config) throws DMLException {
		
		// constructor resets the set of registered functions
		Program rtprog = new Program();
		
		// for all namespaces, translate function statement blocks into function program blocks
		for (String namespace : _namespaces.keySet()){
		
			for (String fname : getFunctionStatementBlocks(namespace).keySet()){
				// add program block to program
				FunctionStatementBlock fsb = getFunctionStatementBlocks(namespace).get(fname);
				FunctionProgramBlock rtpb = (FunctionProgramBlock)createRuntimeProgramBlock(rtprog, fsb, debug, config);
				rtprog.addFunctionProgramBlock(namespace, fname,rtpb);
			}
		}
		
		// for each top-level block
		for (StatementBlock sb : _blocks) {
		
			// add program block to program
			ProgramBlock rtpb = createRuntimeProgramBlock(rtprog, sb, debug, config);
			rtprog.addProgramBlock(rtpb);

		}
		return rtprog ;
	}
	
	
	protected ProgramBlock createRuntimeProgramBlock(Program prog, StatementBlock sb, boolean debug, DMLConfig config) throws DMLException {
		Dag<Lops> dag, pred_dag = null;

		ArrayList<Instruction> instruct;
		ArrayList<Instruction> pred_instruct = null;
		
		// process While Statement - add runtime program blocks to program
		if (sb instanceof WhileStatementBlock){
		
			// create DAG for loop predicates
			pred_dag = new Dag<Lops>();
			((WhileStatementBlock) sb).get_predicateLops().addToDag(pred_dag);
			
			// create instructions for loop predicates
			pred_instruct = new ArrayList<Instruction>();
			ArrayList<Instruction> pInst = pred_dag.getJobs(debug,config);
			for (Instruction i : pInst ) {
				pred_instruct.add(i);
			}
			
			// create while program block
			WhileProgramBlock rtpb = new WhileProgramBlock(prog, pred_instruct);
			
			if (rtpb.getPredicateResultVar() == null) {
				// e.g case : WHILE(continue)
				if ( ((WhileStatementBlock) sb).get_predicateLops().getExecLocation() == LopProperties.ExecLocation.Data ) {
					String resultVar = ((WhileStatementBlock) sb).get_predicateLops().getOutputParameters().getLabel();
					rtpb.setPredicateResultVar( resultVar );
				}
				else
					throw new LopsException("Error in translating the WHILE predicate."); 
			}
			
			//// process the body of the while statement block ////
			
			WhileStatementBlock wsb = (WhileStatementBlock)sb;
			if (wsb.getNumStatements() > 1)
				throw new LopsException("WhileStatementBlock should only have 1 statement");
			
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			for (StatementBlock sblock : wstmt.getBody()){
				
				// process the body
				ProgramBlock childBlock = createRuntimeProgramBlock(prog, sblock, debug, config);
				rtpb.addProgramBlock(childBlock);
			}
			
			// check there are actually Lops in to process (loop stmt body will not have any)
			if (wsb.get_lops() != null && wsb.get_lops().size() > 0){
				throw new LopsException("WhileStatementBlock should have no Lops");
			}
			
			// initialize local _matrices hashmap using metadata of input matrices (as computed in piggybacking)
			// these values are updated (whenever necessary) by the metadata computed at runtime 
			rtpb.initInputMatrixMetadata ( );
			return rtpb;
		}
		
		// process If Statement - add runtime program blocks to program
		else if (sb instanceof IfStatementBlock){
		
			// create DAG for loop predicates
			pred_dag = new Dag<Lops>();
			((IfStatementBlock) sb).get_predicateLops().addToDag(pred_dag);
			
			// create instructions for loop predicates
			pred_instruct = new ArrayList<Instruction>();
			ArrayList<Instruction> pInst = pred_dag.getJobs(debug,config);
			for (Instruction i : pInst ) {
				pred_instruct.add( i);
			}
			
			// create if program block
			IfProgramBlock rtpb = new IfProgramBlock(prog, pred_instruct);
			
			if (rtpb.getPredicateResultVar() == null ) {
				// e.g case : If(continue)
				if ( ((IfStatementBlock) sb).get_predicateLops().getExecLocation() == LopProperties.ExecLocation.Data ) {
					String resultVar = ((IfStatementBlock) sb).get_predicateLops().getOutputParameters().getLabel();
					rtpb.setPredicateResultVar( resultVar );
				}
				else
					throw new LopsException("Error in translating the WHILE predicate."); 
			}
			
			// process the body of the if statement block
			IfStatementBlock isb = (IfStatementBlock)sb;
			if (isb.getNumStatements() > 1)
				throw new LopsException("IfStatementBlock should have only 1 statement");
			
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			
			// process the if body
			for (StatementBlock sblock : istmt.getIfBody()){
				ProgramBlock childBlock = createRuntimeProgramBlock(prog, sblock, debug, config);
				rtpb.addProgramBlockIfBody(childBlock);
			}
			
			// process the else body
			for (StatementBlock sblock : istmt.getElseBody()){
				ProgramBlock childBlock = createRuntimeProgramBlock(prog, sblock, debug, config);
				rtpb.addProgramBlockElseBody(childBlock); 
			}
			
			// check there are actually Lops in to process (loop stmt body will not have any)
			if (isb.get_lops() != null && isb.get_lops().size() > 0){
				throw new LopsException("IfStatementBlock should have no Lops");
			}
			
			// initialize local _matrices hashmap using metadata of input matrices (as computed in piggybacking)
			// these values are updated (whenever necessary) by the metadata computed at runtime 
			rtpb.initInputMatrixMetadata ( );
			return rtpb;
		}
		
		// process For Statement - add runtime program blocks to program
		// NOTE: applies to ForStatementBlock and ParForStatementBlock
		else if (sb instanceof ForStatementBlock) 
		{ 
			ForStatementBlock fsb = (ForStatementBlock) sb;
			
			// create DAGs for loop predicates 
			Dag<Lops> fromDag = new Dag<Lops>();
			Dag<Lops> toDag = new Dag<Lops>();
			Dag<Lops> incrementDag = new Dag<Lops>();
			if( fsb.getFromHops()!=null )
				fsb.getFromLops().addToDag(fromDag);
			if( fsb.getToHops()!=null )
				fsb.getToLops().addToDag(toDag);		
			if( fsb.getIncrementHops()!=null )
				fsb.getIncrementLops().addToDag(incrementDag);		
				
			// create instructions for loop predicates			
			ArrayList<Instruction> fromInstructions = fromDag.getJobs(debug,config);
			ArrayList<Instruction> toInstructions = toDag.getJobs(debug,config);
			ArrayList<Instruction> incrementInstructions = incrementDag.getJobs(debug,config);		

			// create for program block
			String sbName = null;
			ForProgramBlock rtpb = null;
			IterablePredicate iterPred = fsb.getIterPredicate();
			String [] iterPredData= IterablePredicate.createIterablePredicateVariables(iterPred.getIterVar().getName(),
					                                                                   fsb.getFromLops(), fsb.getToLops(), fsb.getIncrementLops()); 
			
			if( sb instanceof ParForStatementBlock )
			{
				sbName = "ParForStatementBlock";
				rtpb = new ParForProgramBlock(prog, iterPredData,iterPred.getParForParams());
				((ParForProgramBlock)rtpb).setResultVariables( ((ParForStatementBlock)sb).getResultVariables() );
			}
			else //ForStatementBlock
			{
				sbName = "ForStatementBlock";
				rtpb = new ForProgramBlock(prog, iterPredData);
			}
			 
			rtpb.setFromInstructions(      fromInstructions      );
			rtpb.setToInstructions(        toInstructions        );
			rtpb.setIncrementInstructions( incrementInstructions );
			
			rtpb.setIterablePredicateVars( iterPredData );
			
			// process the body of the for statement block
			if (fsb.getNumStatements() > 1)
				throw new LopsException( sbName+" should have 1 statement" );
			
			ForStatement fs = (ForStatement)fsb.getStatement(0);
			for (StatementBlock sblock : fs.getBody()){
				ProgramBlock childBlock = createRuntimeProgramBlock(prog, sblock, debug, config);
				rtpb.addProgramBlock(childBlock); 
			}
		
			// check there are actually Lops in to process (loop stmt body will not have any)
			if (fsb.get_lops() != null && fsb.get_lops().size() > 0){
				throw new LopsException( sbName+" should have no Lops" );
			}
			
			// initialize local _matrices hashmap using metadata of input matrices (as computed in piggybacking)
			// these values are updated (whenever necessary) by the metadata computed at runtime 
			rtpb.initInputMatrixMetadata ( );
			return rtpb;
		
		}
		
		// process function statement block - add runtime program blocks to program
		if (sb instanceof FunctionStatementBlock){
			
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			if (fsb.getNumStatements() > 1)
				throw new LopsException("FunctionStatementBlock should only have 1 statement");
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			FunctionProgramBlock rtpb = null;
			
			if (fstmt instanceof ExternalFunctionStatement) {
				 // create external function program block
				
				String execLoc = ((ExternalFunctionStatement) fstmt)
                				  .getOtherParams().get("execlocation");
				boolean isCP = (execLoc!=null && execLoc.equals("CP")) ? true : false;
				
				if( isCP )
				{
					rtpb = new ExternalFunctionProgramBlockCP(prog, 
							fstmt.getInputParams(), fstmt.getOutputParams(), 
							((ExternalFunctionStatement) fstmt).getOtherParams(),
							config.getTextValue("scratch")+ProgramConverter.CP_ROOT_THREAD_SEPARATOR + 
                                                           ProgramConverter.CP_ROOT_THREAD_ID + 
                                                           ProgramConverter.CP_ROOT_THREAD_SEPARATOR);					
				}
				else
				{
					rtpb = new ExternalFunctionProgramBlock(prog, 
									fstmt.getInputParams(), fstmt.getOutputParams(), 
									((ExternalFunctionStatement) fstmt).getOtherParams());
				}
				if (fstmt.getBody().size() > 0){
					throw new LopsException("ExternalFunctionStatementBlock should have no statement blocks in body");
				}
			}
			else 
			{
		
				// create function program block
				rtpb = new FunctionProgramBlock(prog, fstmt.getInputParams(), fstmt.getOutputParams());
				
				// process the function statement body
				for (StatementBlock sblock : fstmt.getBody()){	
					// process the body
					ProgramBlock childBlock = createRuntimeProgramBlock(prog, sblock, debug, config);
					rtpb.addProgramBlock(childBlock);
				}
			}
			
			// check there are actually Lops in to process (loop stmt body will not have any)
			if (fsb.get_lops() != null && fsb.get_lops().size() > 0){
				throw new LopsException("FunctionStatementBlock should have no Lops");
			}
			
			// initialize local _matrices hashmap using metadata of input matrices (as computed in piggybacking)
			// these values are updated (whenever necessary) by the metadata computed at runtime 
			rtpb.initInputMatrixMetadata ( );
			return rtpb;
		}
		
		else if (sb instanceof CVStatementBlock) {
			// handle CV case
			CVStatementBlock cvsb = ((CVStatementBlock)sb) ;
			
			CVProgramBlock cvpb = null; //new CVProgramBlock( prog, cvsb.getPartitionParams(), ((CVStatement)cvsb.getStatement(0)).getFunctionParameters(), config) ;		
		
			// check there are actually Lops in to process (loop stmt body will not have any)
			if (cvsb.get_lops() != null && cvsb.get_lops().size() > 0){
				
				// DAGs for Lops
				dag = new Dag<Lops>();
				
				for (Lops l : cvsb.get_lops()) {
					l.addToDag(dag);
				}
				// Instructions for Lobs DAGs
				instruct = dag.getJobs(debug,config);
				for (Instruction i : instruct) {
					cvpb.addInstruction(i);
				}
				
				// add instruction for a function call
				if (sb.getFunctionCallInst() != null){
					cvpb.addInstruction(sb.getFunctionCallInst());
				}
			}

			// initialize local _matrices hashmap using metadata of input matrices (as computed in piggybacking)
			// these values are updated (whenever necessary) by the metadata computed at runtime 
			cvpb.initInputMatrixMetadata ( );
			return cvpb;
		}
		
		else if (sb instanceof ELStatementBlock) {
			// handle EL case
			ELStatementBlock esb = ((ELStatementBlock)sb) ;
			
			ELProgramBlock epb = null; // new ELProgramBlock( prog, esb.getPartitionParams(), ((ELStatement)esb.getStatement(0)).getFunctionParameters(), config) ;		
		
			// check there are actually Lops in to process (loop stmt body will not have any)
			if (esb.get_lops() != null && esb.get_lops().size() > 0){
				
				// DAGs for Lops
				dag = new Dag<Lops>();
				
				for (Lops l : esb.get_lops()) {
					l.addToDag(dag);
				}
				// Instructions for Lobs DAGs
				instruct = dag.getJobs(debug,config);
				for (Instruction i : instruct) {
					epb.addInstruction(i);
				}
				
				// add instruction for a function call
				if (sb.getFunctionCallInst() != null){
					epb.addInstruction(sb.getFunctionCallInst());
				}
			}

			// initialize local _matrices hashmap using metadata of input matrices (as computed in piggybacking)
			// these values are updated (whenever necessary) by the metadata computed at runtime 
			epb.initInputMatrixMetadata ( );
			return epb;
		}
		
		else if (sb instanceof ELUseStatementBlock) {
			// handle EL Use case
			ELUseStatementBlock eusb = ((ELUseStatementBlock)sb) ;
			
			ELUseProgramBlock eupb = null; //new ELUseProgramBlock( prog, eusb.getPartitionParams(), ((ELUseStatement)eusb.getStatement(0)).getFunctionParameters(), config) ;		
		
			// check there are actually Lops in to process (loop stmt body will not have any)
			if (eusb.get_lops() != null && eusb.get_lops().size() > 0){
				
				// DAGs for Lops
				dag = new Dag<Lops>();
				
				for (Lops l : eusb.get_lops()) {
					l.addToDag(dag);
				}
				// Instructions for Lobs DAGs
				instruct = dag.getJobs(debug,config);
				for (Instruction i : instruct) {
					eupb.addInstruction(i);
				}
				
				// add instruction for a function call
				if (sb.getFunctionCallInst() != null){
					eupb.addInstruction(sb.getFunctionCallInst());
				}
			}

			// initialize local _matrices hashmap using metadata of input matrices (as computed in piggybacking)
			// these values are updated (whenever necessary) by the metadata computed at runtime 
			eupb.initInputMatrixMetadata ( );
			return eupb;
		}
		else {
			// handle general case
			ProgramBlock rtpb = new ProgramBlock(prog);
		
			// DAGs for Lops
			dag = new Dag<Lops>();

			// check there are actually Lops in to process (loop stmt body will not have any)
			if (sb.get_lops() != null && sb.get_lops().size() > 0){
			
				for (Lops l : sb.get_lops()) {
					l.addToDag(dag);
				}
				// Instructions for Lobs DAGs
				instruct = dag.getJobs(debug,config);
				for (Instruction i : instruct) {
					rtpb.addInstruction(i);
				}
			}
			
			// add instruction for a function call
			if (sb.getFunctionCallInst() != null){
				rtpb.addInstruction(sb.getFunctionCallInst());
			}
			
			// initialize local _matrices hashmap using metadata of input matrices (as computed in piggybacking)
			// these values are updated (whenever necessary) by the metadata computed at runtime 
			rtpb.initInputMatrixMetadata ( );
			return rtpb ;	
		}
	}	
}


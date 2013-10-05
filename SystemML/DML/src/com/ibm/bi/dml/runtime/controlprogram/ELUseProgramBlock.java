/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

//package com.ibm.bi.dml.runtime.controlprogram;
//
//import com.ibm.bi.dml.meta.PartitionParams;
//import com.ibm.bi.dml.parser.MetaLearningFunctionParameters;
//import com.ibm.bi.dml.runtime.instructions.Instruction;
//import com.ibm.bi.dml.utils.DMLRuntimeException;
//import com.ibm.bi.dml.utils.configuration.DMLConfig;
//
// 
//public class ELUseProgramBlock extends ProgramBlock {
//	
//	public void printMe() {
//		for (Instruction i : this._inst)
//			i.printMe();
//	}
//
//	MetaLearningFunctionParameters _params ;
//	PartitionParams _pp ;
//	
//	public ELUseProgramBlock(Program prog, PartitionParams pp, MetaLearningFunctionParameters params, DMLConfig passedConfig)
//	throws DMLRuntimeException
//	{
//		super(prog);
//		_prog = prog;
//		_params = params; 
//		_pp = pp ;
//	}
//
//
//	/*protected void executePartition() throws DMLRuntimeException, DMLUnsupportedOperationException {
//		if (LOG.isTraceEnabled()){
//			LOG.trace("Variables: " + _variables.toString ());
//		}
//		
//		
//		for (int i = 0; i < _inst.size(); i++) {
//			Instruction currInst = _inst.get(i);
//			if (currInst instanceof MRJobInstruction) {
//				MRJobInstruction currMRInst = (MRJobInstruction) currInst;
//					
//				JobReturn jb = RunMRJobs.submitJob(currMRInst, this);
//				
//				 Populate returned stats into symbol table of matrices 
//				for ( int index=0; index < jb.getMetaData().length; index++) {
//					// TODO: Fix This
//					//_matrices.put(currMRInst.getIv_outputs()[index], jb.getMetaData(index));
//					
//					// TODO: DRB: need to add binding to variables here
//					
//				}
//				
//				Statistics.setNoOfExecutedMRJobs(Statistics.getNoOfExecutedMRJobs() + 1);
//			} else if (currInst instanceof CPInstruction) {
//				String updInst = RunMRJobs.updateLabels(currInst.toString(), _variables);
//				CPInstruction si = CPInstructionParser.parseSingleInstruction(updInst);
//				si.processInstruction(this);
//			} 
//		}
//	}
//	
//	
//	public void execute(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException{
//		
//		*//********************* Set partitions ************************************************//*
//
//		// insert call to partitionParam function
//		
//		//mapList = (_pp.toReplicate) ?	executePartition(_partitionInst, _pp.getOutputStrings1()) :  executePartition(_partitionInst, _pp.getOutputStrings2());
//		
//		// _inst in CV program block are partition instructions (from lop dag)
//		// outputs from partition 
//		executePartition();
//	
//		System.out.println("Finished executing partition") ;
//		
//		String outputVar = _params.getErrorAggFormalParams().get(0);
//		this.setVariable(outputVar, new DoubleObject(0)) ;
//		
//		int foldCount = -1;
//		if (_pp.cvt == PartitionParams.CrossvalType.holdout)
//			foldCount = _pp.numIterations;
//		else if (_pp.cvt == PartitionParams.CrossvalType.kfold)
//			foldCount = _pp.numFolds;
//		else 
//			foldCount = _pp.numFolds;
//					
//		for(int foldId = 0 ; foldId < foldCount; foldId++) {
//		
//			*//*************************** Execute test function block *******************************//*	
//			// THIS FUNCTION CALL NEEDS TO BE UPDATED
//			FunctionProgramBlock testpb = null; //_prog.getFunctionProgramBlock(_params.getTestFunctionName());
//
//			// create bindings to formal parameters for training function call
//			
//			// These are the bindings passed to the FunctionProgramBlock for function execution 
//			LocalVariableMap functionVariables = setFunctionVariables(testpb, _params.getTestFunctionFormalParams());
//			
//			// execute the function block
//			testpb.setVariables(functionVariables);
//			// TODO: Fix This
//			//testpb.setMetaData(this.getMetaData());
//			testpb.execute(ec);
//			LocalVariableMap returnedVariables = testpb.getVariables(); 
//			
//			
//			// add the updated binding for each return variable to the CV program block variables
//			for (int i=0; i< testpb.getOutputParams().size(); i++){
//			
//				String boundVarName = _params.getTrainFunctionReturnParams().get(i); //  _boundOutputParamNames.get(i); 
//				Data boundValue = returnedVariables.get(testpb.getOutputParams().get(i).getName());
//				if (boundValue == null)
//					throw new DMLUnsupportedOperationException(boundVarName + " was not assigned a return value");
//			
//				this.getVariables().put(boundVarName, boundValue);
//			}
//
//			// this is the place where we can check if the function wrote any files out.
//			// TODO: Fix This
//			//this.setMetaData(testpb.getMetaData());
//			
//			
//			*//*************** aggregate errors ********************************************************//*
//			
//			// set the error outputs for aggregation
//			
//			// constraint is that the	
//			String testOutputName = _params.getTestFunctionReturnParams().get(0) ;
//			
//			VariableCPInstruction getError = (VariableCPInstruction) InstructionParser.parseSingleInstruction("assigndoublevar" + Instruction.OPERAND_DELIM + testOutputName + foldId + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM + "iter" + foldId + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE);
//			CPInstruction ssi = null;
//			try {
//				ssi = (CPInstruction) InstructionParser.parseSingleInstruction("+" + Instruction.OPERAND_DELIM + "iter" + foldId + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM + outputVar + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM + outputVar + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE) ;
//			}
//			catch ( Exception e ) {
//				e.printStackTrace();
//			}
//			getError.processInstruction(this) ;
//			ssi.processInstruction(this) ;
//			
//		} // end for each fold
//		
//		// handle the aggregation of the errors across the folds to compute final error for CV
//		if(_params.getAgg() == CVStatement.AGG.avg) {
//			CPInstruction ssi = null;
//			try {
//				//ssi = new ScalarSimpleInstructions("/:::" + outputVar + ":::" + pp.numFolds + ":::" + outputVar) ;
//				// ssi = (ScalarCPInstruction) InstructionParser.parseSingleInstruction("/:::" + outputVar + ":::" + pp.numFolds + ":::" + outputVar) ;
//				ssi = (CPInstruction) InstructionParser.parseSingleInstruction("/" + Instruction.OPERAND_DELIM + outputVar + Instruction.OPERAND_DELIM + _pp.numFolds + Instruction.OPERAND_DELIM + outputVar) ;
//			}
//			catch (Exception e) {
//				e.printStackTrace();
//			}
//			ssi.processInstruction(this) ;
//		}
//
//	} // end execute
//	
//	
//	@Override
//	protected SymbolTable createSymbolTable() {
//		// TODO: override this function whenever CV implementation is revisited
//		return null;
//	}
//
//	*//**
//	 * 
//	 * @param fpb Function program block for function being called
//	 * @param formalParams the formal parameters function is being called with [NOTE: these are string values, so arbitrary expressions as formal parameters are not supported]
//	 * @return the binding of data values 
//	 * @throws DMLUnsupportedOperationException
//	 * @throws DMLRuntimeException 
//	 *//*
//	LocalVariableMap setFunctionVariables(FunctionProgramBlock fpb, ArrayList<String> formalParams) throws DMLUnsupportedOperationException, DMLRuntimeException{
//	
//		LocalVariableMap retVal = new LocalVariableMap (); 
//		
//		for (int i=0; i<fpb.getInputParams().size();i++) {
//			
//			String currFormalParamName = fpb.getInputParams().get(i).getName();
//			Data currFormalParamValue = null; 
//			ValueType valType = fpb.getInputParams().get(i).getValueType();
//			
//			if (i > formalParams.size() || (fpb.getVariables().get(formalParams.get(i)) == null)){
//				
//				if (valType == ValueType.BOOLEAN){
//					boolean defaultVal = (i > formalParams.size()) ? new Boolean(fpb.getInputParams().get(i).getDefaultValue()).booleanValue() : new Boolean(formalParams.get(i)).booleanValue();
//					currFormalParamValue = new BooleanObject(defaultVal);
//				}
//				else if (valType == ValueType.DOUBLE){
//					double defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? new Double(fpb.getInputParams().get(i).getDefaultValue()).doubleValue() : new Double(formalParams.get(i)).doubleValue();
//					currFormalParamValue = new DoubleObject(defaultVal);
//				}
//				else if (valType == ValueType.INT){
//					int defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? new Integer(fpb.getInputParams().get(i).getDefaultValue()).intValue() : new Integer(formalParams.get(i)).intValue();
//					currFormalParamValue = new IntObject(defaultVal);
//				}
//				else if (valType == ValueType.STRING){
//					String defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? fpb.getInputParams().get(i).getDefaultValue() : formalParams.get(i);
//					currFormalParamValue = new StringObject(defaultVal);
//				}
//				else{
//					throw new DMLUnsupportedOperationException(currFormalParamValue + " has inapporpriate value type");
//				}
//			}
//		
//			else {
//				currFormalParamValue = this.getVariables().get(formalParams.get(i));
//			}
//				
//			retVal.put(currFormalParamName,currFormalParamValue);	
//		}
//		return retVal;
//	} // end method setFunctionVariables
//*/	
//	
//} // end class
//	
//	
//	
//	
//	
//	
//	
//
//	
//	
//	
//	
//	
//	
//	
//	

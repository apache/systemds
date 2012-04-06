package dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

import dml.api.DMLScript;
import dml.lops.runtime.RunMRJobs;
import dml.meta.PartitionParams;
import dml.parser.CVStatement;
import dml.parser.MetaLearningFunctionParameters;
import dml.parser.Expression.ValueType;
import dml.runtime.instructions.CPInstructionParser;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionParser;
import dml.runtime.instructions.MRJobInstruction;
import dml.runtime.instructions.CPInstructions.BooleanObject;
import dml.runtime.instructions.CPInstructions.CPInstruction;
import dml.runtime.instructions.CPInstructions.Data;
import dml.runtime.instructions.CPInstructions.DoubleObject;
import dml.runtime.instructions.CPInstructions.IntObject;
import dml.runtime.instructions.CPInstructions.StringObject;
import dml.runtime.instructions.CPInstructions.VariableCPInstruction;
import dml.runtime.matrix.JobReturn;
import dml.runtime.matrix.MetaData;
import dml.sql.sqlcontrolprogram.ExecutionContext;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;
import dml.utils.Statistics;
import dml.utils.configuration.DMLConfig;
 
public class ELUseProgramBlock extends ProgramBlock {
	
	public void printMe() {
		for (Instruction i : this._inst)
			i.printMe();
	}

	MetaLearningFunctionParameters _params ;
	PartitionParams _pp ;
	
	public ELUseProgramBlock(Program prog, PartitionParams pp, MetaLearningFunctionParameters params, DMLConfig passedConfig)
	{
		super(prog);
		_prog = prog;
		_params = params; 
		_pp = pp ;
	}


	protected void executePartition() throws DMLRuntimeException, DMLUnsupportedOperationException {
		if ( DMLScript.DEBUG ) {
			// print _variables map
			System.out.println("____________________________________");
			System.out.println("___ Variables ____");
			Iterator<Entry<String, Data>> it = _variables.entrySet().iterator();
			while (it.hasNext()) {
				Entry<String,Data> pairs = it.next();
			    System.out.println("  " + pairs.getKey() + " = " + pairs.getValue());
			}
			System.out.println("___ Matrices ____");
			// TODO: Fix This
			Iterator<Entry<String, MetaData>> mit = null; // _matrices.entrySet().iterator();
			while (mit.hasNext()) {
				Entry<String,MetaData> pairs = mit.next();
			    System.out.println("  " + pairs.getKey() + " = " + pairs.getValue());
			}
			System.out.println("____________________________________");
		}
		updateMatrixLabels();
		for (int i = 0; i < _inst.size(); i++) {
			Instruction currInst = _inst.get(i);
			if (currInst instanceof MRJobInstruction) {
				MRJobInstruction currMRInst = (MRJobInstruction) currInst;
					
				currMRInst.setInputLabelValueMapping(_variables);
				currMRInst.setOutputLabelValueMapping(_variables);
				
				JobReturn jb = RunMRJobs.submitJob(currMRInst, this);
				
				/* Populate returned stats into symbol table of matrices */
				for ( int index=0; index < jb.getMetaData().length; index++) {
					// TODO: Fix This
					//_matrices.put(currMRInst.getIv_outputs()[index], jb.getMetaData(index));
					
					// TODO: DRB: need to add binding to variables here
					
				}
				
				Statistics.setNoOfExecutedMRJobs(Statistics.getNoOfExecutedMRJobs() + 1);
			} else if (currInst instanceof CPInstruction) {
				String updInst = RunMRJobs.updateLabels(currInst.toString(), _variables);
				CPInstruction si = CPInstructionParser.parseSingleInstruction(updInst);
				si.processInstruction(this);
			} 
		}
	}
	
	
	public void execute(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException{
		
		/********************* Set partitions ************************************************/

		// insert call to partitionParam function
		
		//mapList = (_pp.toReplicate) ?	executePartition(_partitionInst, _pp.getOutputStrings1()) :  executePartition(_partitionInst, _pp.getOutputStrings2());
		
		// _inst in CV program block are partition instructions (from lop dag)
		// outputs from partition 
		executePartition();
	
		System.out.println("Finished executing partition") ;
		
		String outputVar = _params.getErrorAggFormalParams().get(0);
		this.setVariable(outputVar, new DoubleObject(0)) ;
		
		int foldCount = -1;
		if (_pp.cvt == PartitionParams.CrossvalType.holdout)
			foldCount = _pp.numIterations;
		else if (_pp.cvt == PartitionParams.CrossvalType.kfold)
			foldCount = _pp.numFolds;
		else 
			foldCount = _pp.numFolds;
					
		for(int foldId = 0 ; foldId < foldCount; foldId++) {
		
			/*************************** Execute test function block *******************************/	
		
			FunctionProgramBlock testpb = _prog.getFunctionProgramBlock(_params.getTestFunctionName());

			// create bindings to formal parameters for training function call
			
			// These are the bindings passed to the FunctionProgramBlock for function execution 
			HashMap<String,Data> functionVariables = setFunctionVariables(testpb, _params.getTestFunctionFormalParams());
			
			// execute the function block
			testpb.setVariables(functionVariables);
			// TODO: Fix This
			//testpb.setMetaData(this.getMetaData());
			testpb.execute(ec);
			HashMap<String, Data> returnedVariables = testpb.getVariables(); 
			
			
			// add the updated binding for each return variable to the CV program block variables
			for (int i=0; i< testpb.getOutputParams().size(); i++){
			
				String boundVarName = _params.getTrainFunctionReturnParams().get(i); //  _boundOutputParamNames.get(i); 
				Data boundValue = returnedVariables.get(testpb.getOutputParams().get(i).getName());
				if (boundValue == null)
					throw new DMLUnsupportedOperationException(boundVarName + " was not assigned a return value");
			
				this.getVariables().put(boundVarName, boundValue);
			}

			// this is the place where we can check if the function wrote any files out.
			// TODO: Fix This
			//this.setMetaData(testpb.getMetaData());
			
			
			/*************** aggregate errors ********************************************************/
			
			// set the error outputs for aggregation
			
			// constraint is that the	
			String testOutputName = _params.getTestFunctionReturnParams().get(0) ;
			
			VariableCPInstruction getError = (VariableCPInstruction) InstructionParser.parseSingleInstruction("assigndoublevar" + Instruction.OPERAND_DELIM + testOutputName + foldId + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM + "iter" + foldId + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE);
			CPInstruction ssi = null;
			try {
				ssi = (CPInstruction) InstructionParser.parseSingleInstruction("+" + Instruction.OPERAND_DELIM + "iter" + foldId + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM + outputVar + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM + outputVar + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE) ;
			}
			catch ( Exception e ) {
				e.printStackTrace();
			}
			getError.processInstruction(this) ;
			ssi.processInstruction(this) ;
			
		} // end for each fold
		
		// handle the aggregation of the errors across the folds to compute final error for CV
		if(_params.getAgg() == CVStatement.AGG.avg) {
			CPInstruction ssi = null;
			try {
				//ssi = new ScalarSimpleInstructions("/:::" + outputVar + ":::" + pp.numFolds + ":::" + outputVar) ;
				// ssi = (ScalarCPInstruction) InstructionParser.parseSingleInstruction("/:::" + outputVar + ":::" + pp.numFolds + ":::" + outputVar) ;
				ssi = (CPInstruction) InstructionParser.parseSingleInstruction("/" + Instruction.OPERAND_DELIM + outputVar + Instruction.OPERAND_DELIM + _pp.numFolds + Instruction.OPERAND_DELIM + outputVar) ;
			}
			catch (Exception e) {
				e.printStackTrace();
			}
			ssi.processInstruction(this) ;
		}

	} // end execute
	
	
	/**
	 * 
	 * @param fpb Function program block for function being called
	 * @param formalParams the formal parameters function is being called with [NOTE: these are string values, so arbitrary expressions as formal parameters are not supported]
	 * @return the binding of data values 
	 * @throws DMLUnsupportedOperationException
	 */
	HashMap<String, Data> setFunctionVariables(FunctionProgramBlock fpb, ArrayList<String> formalParams) throws DMLUnsupportedOperationException{
	
		HashMap<String, Data> retVal = new HashMap<String, Data>(); 
		
		for (int i=0; i<fpb.getInputParams().size();i++) {
			
			String currFormalParamName = fpb.getInputParams().get(i).getName();
			Data currFormalParamValue = null; 
			ValueType valType = fpb.getInputParams().get(i).getValueType();
			
			if (i > formalParams.size() || (fpb.getVariables().get(formalParams.get(i)) == null)){
				
				if (valType == ValueType.BOOLEAN){
					boolean defaultVal = (i > formalParams.size()) ? new Boolean(fpb.getInputParams().get(i).getDefaultValue()).booleanValue() : new Boolean(formalParams.get(i)).booleanValue();
					currFormalParamValue = new BooleanObject(defaultVal);
				}
				else if (valType == ValueType.DOUBLE){
					double defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? new Double(fpb.getInputParams().get(i).getDefaultValue()).doubleValue() : new Double(formalParams.get(i)).doubleValue();
					currFormalParamValue = new DoubleObject(defaultVal);
				}
				else if (valType == ValueType.INT){
					int defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? new Integer(fpb.getInputParams().get(i).getDefaultValue()).intValue() : new Integer(formalParams.get(i)).intValue();
					currFormalParamValue = new IntObject(defaultVal);
				}
				else if (valType == ValueType.STRING){
					String defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? fpb.getInputParams().get(i).getDefaultValue() : formalParams.get(i);
					currFormalParamValue = new StringObject(defaultVal);
				}
				else{
					throw new DMLUnsupportedOperationException(currFormalParamValue + " has inapporpriate value type");
				}
			}
		
			else {
				currFormalParamValue = this.getVariables().get(formalParams.get(i));
			}
				
			retVal.put(currFormalParamName,currFormalParamValue);	
		}
		return retVal;
	} // end method setFunctionVariables
	
	
} // end class
	
	
	
	
	
	
	

	
	
	
	
	
	
	
	

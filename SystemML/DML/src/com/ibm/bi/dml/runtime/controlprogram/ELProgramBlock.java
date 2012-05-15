package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.parser.FunctionStatement;
import com.ibm.bi.dml.parser.MetaLearningFunctionParameters;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.instructions.CPInstructionParser;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FileCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FileObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.Statistics;
import com.ibm.bi.dml.utils.configuration.DMLConfig;


public class ELProgramBlock extends ProgramBlock {

	public void printMe() {
		for (Instruction i : this._inst) {
			i.printMe();
		}
	}

	MetaLearningFunctionParameters _params ;
	PartitionParams _pp ;
	
	public ELProgramBlock(Program prog, PartitionParams pp, MetaLearningFunctionParameters params, DMLConfig passedConfig) {
		super(prog);
		_prog = prog;
		_params = params; 
		_pp = pp ;
	}

	protected void executePartition() throws DMLRuntimeException, DMLUnsupportedOperationException {
		updateMatrixLabels();		//basically replaces ##..## stuff with actual file names of matr varbls
		for (int i = 0; i < _inst.size(); i++) {	//only one itern occurs though
			Instruction currInst = _inst.get(i);
			if (currInst instanceof MRJobInstruction) {
				MRJobInstruction currMRInst = (MRJobInstruction) currInst;
				//populate varbls table with output matrix filepaths
				for ( int index=0; index < currMRInst.getIv_outputs().length; index++) {
					//Arun: now, the output matrices (A1, A2...) have filepaths "./data/A1".. - now A1 
					_variables.put(currMRInst.getIv_outputs()[index], new FileObject(currMRInst.getIv_outputs()[index], 
																			"" + currMRInst.getIv_outputs()[index])); 
				}	
				currMRInst.setInputLabelValueMapping(_variables);
				currMRInst.setOutputLabelValueMapping(_variables);
				
				JobReturn jb = RunMRJobs.submitJob(currMRInst, this);
				//Note that submitjob has the varblnames as inputs; runjob call takes in filepathsnames after updatelabels on varblname inputs
				if(jb.getMetaData().length != currMRInst.getIv_outputs().length) {
					System.out.println("Error after partitioning in cv progm blk - no. matrices don't match!");
					System.exit(1);
				}
				//Populate returned stats into symbol table of matrices
				for ( int index=0; index < jb.getMetaData().length; index++) {
					// TODO: Fix This
					//_matrices.put(new String("" + currMRInst.getIv_outputs()[index]), jb.getMetaData(index));
				}
				Statistics.setNoOfExecutedMRJobs(Statistics.getNoOfExecutedMRJobs() + 1);
			} else if (currInst instanceof CPInstruction) {
				String updInst = RunMRJobs.updateLabels(currInst.toString(), _variables);
				CPInstruction si = CPInstructionParser.parseSingleInstruction(updInst);
				si.processInstruction(this);
			} 
		}
	}
	
	
	public void executeReblocks(long nr, long nc, int bnr, int bnc) throws DMLRuntimeException {		
		//getReblksInst(reblksmr, nr, nc, bnr, bnc);		
		MRJobInstruction reblksmr = new MRJobInstruction(JobType.REBLOCK_BINARY);
		//input (-> input re) -> partop_i (done) -> partop_i re (output now)
		String [] reblksinps = _pp.getOutputStrings();
		int nummats = reblksinps.length;
		String [] reblksouts = new String[nummats];	//will become ##INPre##
		InputInfo [] reblksinpsinfos = new InputInfo[nummats];
		OutputInfo [] reblksoutsinfos = new OutputInfo[nummats];
		long[] numrows = new long[nummats];
		long[] numcols = new long[nummats];
		int[] numrpbs = new int[nummats];
		int[] numcpbs = new int[nummats];		//TODO Arun: This means #######max blk dimensn is integer limit vs long for nr/nc!!
		byte[] resinds = new byte[nummats];		//TODO Arun: This means onyl ###########256/2 result folds can be done in one-go reblock stmt!!!
		byte[] resdims = new byte[nummats];
		//reblock inst rblk:::0:DOUBLE:::3:DOUBLE:::1000:::1000,rblk:::1:DOUBLE:::4:DOUBLE:::1000:::1000,rblk... //as per statiko
		String reblksinsts = "rblk:::0:DOUBLE:::"+nummats+":DOUBLE:::1000:::1000";
		for(int q=0; q<nummats; q++) {
			reblksouts[q] = reblksinps[q]+"re";	//add re to varblname for output, and enclose in ##,## e.g. ##A0re##
			//populate varbls table with outmatx filepaths //(A1re, A2re...) have filepaths "./data/A1re"... -> now A1re
			_variables.put(reblksouts[q], new FileObject(reblksouts[q], "" + reblksouts[q])); 
			reblksinps[q] = "##" + reblksinps[q] + "##";
			reblksouts[q] = "" + reblksouts[q] + "";	//no need for ## here! since on return we populate them!
			reblksinpsinfos[q] = InputInfo.BinaryBlockInputInfo;
			reblksoutsinfos[q] = OutputInfo.BinaryBlockOutputInfo;
			numrows[q] = numcols[q] = -1;	//unknown dims
			resdims[q] = 1;			//unknown dims
			resinds[q] = (byte)(nummats + q);	//k, k+1...	 (onyl 256/2 result folds can be handled!!)
			if(_pp.pt == PartitionParams.PartitionType.row && _pp.isColumn == false) {		//TODO: submatrix and cell - TODO!
				numrpbs[q] = 1;
//				numcpbs[q] = (_pp.apt == PartitionParams.AccessPath.RB) ? (int) nc : bnc;	//rowblks vs subrowblks - ******cast
				numcpbs[q] = (int)nc;	//new jr outputs fullrowcolblks
			}
			else if(_pp.pt == PartitionParams.PartitionType.row && _pp.isColumn == true) {
				numcpbs[q] = 1;
//				numrpbs[q] = (_pp.apt == PartitionParams.AccessPath.RB) ? (int) nr : bnr;	//colblks vs subcolblks	- cast might cause probs!
				numrpbs[q] = (int) nr;
			}
			if(q > 0)
				reblksinsts += ",rblk:::"+q+":DOUBLE:::"+(q+nummats)+":DOUBLE:::1000:::1000";
		}
		/*public void setReBlockInstructions(String [] input, InputInfo [] inputInfo, long [] numRows, long [] numCols, 
		 * int [] num_rows_per_block, int [] num_cols_per_block, String mapperInstructions, 
		 * String reblockInstructions, String otherInstructions, String [] output, OutputInfo [] outputInfo, byte [] resultIndex, 
		 * byte[] resultDimsUnknown, int numReducers, int replication, HashSet <String> inLabels, HashSet <String> outLabels)*/
		reblksmr.setReBlockInstructions(reblksinps, reblksinpsinfos, numrows, numcols, numrpbs, numcpbs, "", reblksinsts, "",
				reblksouts, reblksoutsinfos, resinds, resdims, 1, 1, null, null);	//as per statiko
		
		reblksmr.setInputLabelValueMapping(_variables);
		reblksmr.setOutputLabelValueMapping(_variables);
		JobReturn jb = RunMRJobs.submitJob(reblksmr, this);
		if(jb.getMetaData().length != nummats) {
			System.out.println("Error after reblocking in cv progm blk - no. matrices don't match!");
			System.exit(1);
		}
		//Populate returned stats into symbol table of matrices; also delete pre-reblk files from hdfs and entries in vars and mats
		for ( int index=0; index < jb.getMetaData().length; index++) {
			// TODO: Fix This
			//_matrices.put(new String("" + _pp.getOutputStrings()[index] + "re"), jb.getMetaData(index));
			
			String filepathname = "" + _pp.getOutputStrings()[index];
			FileCPInstruction instDelFile = null;
			try {
				instDelFile = (FileCPInstruction) FileCPInstruction.parseInstruction("rm:::" + filepathname);
			} catch (DMLUnsupportedOperationException e) {
				e.printStackTrace();
			}
			instDelFile.processInstruction(this); //this autom removes the matrices entry!
			//_matrices.remove(((FileObject)_variables.get(_pp.getOutputStrings()[index])).getFilePath());
			_variables.remove(_pp.getOutputStrings()[index]);
		}
		Statistics.setNoOfExecutedMRJobs(Statistics.getNoOfExecutedMRJobs() + _pp.getOutputStrings().length);
	}

	//TODO: THe following has to be updated!!!!
	public PartitionParams.AccessPath computeAccessPath(long nr, long nc, int bnr, int bnc) {
		PartitionParams.AccessPath retapt = PartitionParams.AccessPath.JR;	//defaults to JR
		Runtime runtime = Runtime.getRuntime();
		long maxMemory = runtime.maxMemory();  
		long allocatedMemory = runtime.totalMemory();  
		long freeMemory = runtime.freeMemory();  
		long availMemory = freeMemory + (maxMemory - allocatedMemory);	//current avail mem in bytes
		//######### TODO: we cannot check memsize on each mapper indeply!! so, should we check here? or assume fixed size? ###########
		//long availmem = 100 * (1 << 20); 	
		long availmem = availMemory / 2;	//use only 50% of avail memory
		
		//based on rough experiments, the hashmap size formula is roughly 138 N + 8 NV
		long N = (_pp.isColumn == false) ? nr : nc;	//numrows or numcols as numkeys	; note that rsm autom sets iscol true 
		long V = (_pp.toReplicate == false) ? 1 : _pp.numIterations;
		//long hashmapsize = 138 * N + 8 * N * V; - abandoned java hashmap
long hashmapsize = N * V * 8 + 15000; //(for small vector overhead)
		System.out.println("$$$$$$$$$$$$$ Avail memory: " + ((double)availmem/1024/1024) + 
				" and hashmap size: " + ((double)hashmapsize/1024/1024) + "MB $$$$$$$$$$$$$");
//availmem = 1;	//FOR DEBUGGING!
		if(hashmapsize <= availmem)	//we can simply use the hashmap MR job!
			retapt = PartitionParams.AccessPath.HM;
		else if (_pp.et == PartitionParams.EnsembleType.bagging)	//hashmap doesnt fit in memory; so only JR for bootstrap
			retapt = PartitionParams.AccessPath.JR;
		else if(_pp.et == PartitionParams.EnsembleType.rsm) {	 //hashmap doesnt fit in memory! so use row cost formulae
			retapt = PartitionParams.AccessPath.JR;		//by default, use JR
			if (nr * _pp.frac < 2.5) 	//shuffle cost - check paper
				_pp.apt = PartitionParams.AccessPath.RB;	//accesspath = 1;
			else if ((nr * _pp.frac == 2.5) && (nr <= (V + 1) / 2))		//rare case; need to check io cost - paper
				_pp.apt = PartitionParams.AccessPath.RB;	//accesspath = 1;
		}
		else if (_pp.et == PartitionParams.EnsembleType.rowholdout) {	//similar to cv holdout
			if(nc * _pp.frac < 2.5)
				_pp.apt = PartitionParams.AccessPath.RB;
			else
				_pp.apt = PartitionParams.AccessPath.JR;
		}
		else {	//last default
			retapt = PartitionParams.AccessPath.JR;
			//### Boosting?? ###
		}
		
/*if(hashmapsize <= availmem)	//we can simply use the hashmap MR job!
return PartitionParams.AccessPath.HM;	//FOR DEBUGGING
else {
System.out.println("Not enough memory, HM dies!");
System.exit(1);
} //FORDEBUGGING
*/
return retapt;
//return PartitionParams.AccessPath.JR;	//FOR DEBUGGING
//return PartitionParams.AccessPath.RB;	//FOR DEBUGGING
	}
	
	public void execute() throws DMLRuntimeException, DMLUnsupportedOperationException{
		//the accesspath has to be computed here (and not in runmrjobs) since we need to know if post partition reblocks are needed!
		long nr = ((MRJobInstruction) _inst.get(0)).getIv_rows()[0];	//only one instcn exists, so use it for input matr stats
		long nc = ((MRJobInstruction) _inst.get(0)).getIv_cols()[0];
		int bnr = ((MRJobInstruction) _inst.get(0)).getIv_num_rows_per_block()[0];
		int bnc = ((MRJobInstruction) _inst.get(0)).getIv_num_cols_per_block()[0];
		_pp.apt = computeAccessPath(nr, nc, bnr, bnc);
		((MRJobInstruction) _inst.get(0)).getPartitionParams().apt = _pp.apt;
		System.out.println("$$$$$$$$$$$$$ Chosen accesspath: " + _pp.apt + "$$$$$$$$$$$$$");
		//based on torepl, we need to execute partition once outside the for loop vs repeatedly inside the loop		
		/********** Construct folds for partition with replication ****************/ 
		if(_pp.toReplicate == true) {	//partition w repl
			executePartition();
			updateMatrixLabels();
			System.out.println("Finished executing partition w repl!");
			if(_pp.apt == PartitionParams.AccessPath.RB || _pp.apt == PartitionParams.AccessPath.JR) {	//do post partn reblocks for RB and JR
				executeReblocks(nr, nc, bnr, bnc);
				updateMatrixLabels();
				System.out.println("Finished executing reblocks on outputs for JR/RB w repl!");
			}
		}
		
		//iterate thro folds and do train/test/agg/writeout if w repl; for wo repl, we need to create fold by invoking partition again!
		int foldCount = _pp.numIterations;
		
		//EL Metadata file for writeout!!! -> Perhaps a simple seqfile? With key value stores!;
		//The variablename of the ensemble is given a HashMap<String, String> value! and this is passed into _variables of this blk!
		//Later, when we Write() on the ensemble varblname, we can write out the hashmap as a seq file / pmml file / etc.
		HashMap<String,String> ensembleValue = new HashMap<String, String>();
		ensembleValue.put("ensembleName", _pp.ensembleName);
		if(_pp.et == PartitionParams.EnsembleType.rsm)
			ensembleValue.put("ensembleType", "rsm");
		else if(_pp.et == PartitionParams.EnsembleType.rowholdout)
			ensembleValue.put("ensembleType", "rowholdout");
		else if(_pp.et == PartitionParams.EnsembleType.bagging)
			ensembleValue.put("ensembleType", "bagging");
		else if(_pp.et == PartitionParams.EnsembleType.adaboost)
			ensembleValue.put("ensembleType", "adaboost");
		ensembleValue.put("numIterations", new String("" + _pp.numIterations));
		ensembleValue.put("frac", new String("" + _pp.frac));
		ensembleValue.put("trainFunctionName", _params.getTrainFunctionName());
		ArrayList<String> trainrets = _params.getTrainFunctionReturnParams();
		ensembleValue.put("numTrainFunctionReturnParams", new String("" + trainrets.size()));
		//for(int tr=0; tr<trainrets.size(); tr++)
		//	ensembleValue.put("trainFunctionReturnParams" + tr, trainrets.get(tr));		//not really useful
		if(_pp.et == PartitionParams.EnsembleType.rsm)
			ensembleValue.put("hashFile", _pp.sfmapfile);	//can be used later for re-partitioning new unlabeled data along corresp cols  
		//### for boosting, we also need to add the zeta (importance) vector over train folds output models
		
		/******************* Iterate over the folds **************************/ 
		for(int foldId = 0 ; foldId < foldCount; foldId++) {
			/********** Construct folds for partition without replication ****************/
			if(_pp.toReplicate == false) {	//partition wo repl - invoke it and do post partn reblock if ncsry
				executePartition();
				updateMatrixLabels();
				System.out.println("Finished executing partition wo repl for fold "+foldId+"!");
				if(_pp.apt == PartitionParams.AccessPath.RB || _pp.apt == PartitionParams.AccessPath.JR) {	//do post partn reblocks for RB and JR
					executeReblocks(nr, nc, bnr, bnc);
					updateMatrixLabels();
					System.out.println("Finished executing reblock on fold "+foldId+" for JR/RB wo repl!");
				}
			}
			//for mappings to trainer/test, several cases can arise dep on access path, el type and repl: (after stmt level reconciln) 
			/* for both bagging and rsm, w repl, HM sends outputs[i] to train
			 * Wo repl, it is outputs[0] alone! Also, if apt is RB or JR, append "re" to resp. fold aliters
			 * #### boosting: here, w repl is meaningless; so we need to track eahc matr, and weights; also, there's tesing on orig input! 
			 *
			//reconcile formal funcnt argmt varblnames with elstmt varbl names with actual filepathnames!
			System.out.println("$$$$$ In el pgmblk, et:"+ _pp.et + ", repl:"+_pp.toReplicate + "$$$$$$$$$$$$$");
			if(_pp.et == PartitionParams.EnsembleType.rsm || _pp.et == PartitionParams.EnsembleType.rowholdout || 
					_pp.et == PartitionParams.EnsembleType.bagging) {
				String trainformalname = _pp.partitionOutputs.get(0);
				String trainaliter = _pp.getOutputStrings()[foldId];	//wo rpel, foldid will be 0!
				if(_pp.apt == PartitionParams.AccessPath.RB || _pp.apt == PartitionParams.AccessPath.JR) {
					trainaliter += "re";
				}
				_variables.put(trainformalname, new FileObject(trainformalname, ((FileObject)_variables.get(trainaliter)).getFilePath()));
				_variables.remove(trainaliter);
				System.out.println("$$$$$$ Setting up bindings : trainformalname " + trainformalname +	" binds to " + trainaliter);
			}
			
			//#### We assume that the trainer does the projection on its own (ie split labeled data matrix!)
			//Also, we assume that prev defined varbls get passed on properly (matrix, constants, etc) - prev pgm blks
			//8*************************** Execute train function block *******************************8/
			FunctionProgramBlock trainpb = _prog.getFunctionProgramBlock(_params.getTrainFunctionName());	//get train pgm func blk
			
			// create bindings to formal parameters for training function call -> contains logic to bind
			HashMap<String, Data> functionVariables = setFunctionVariables(trainpb, _params.getTrainFunctionFormalParams());
						
			trainpb.setVariables(functionVariables);	//sets the _variables
			trainpb.setMetaData(this.getMetaData());	//sets the _matrices
			trainpb.execute();		// execute training function block
			
			HashMap<String,Data> returnedVariables = trainpb.getVariables(); //add updated binding for retvars to the EL pgmblk vars
			for (int i=0; i< trainpb.getOutputParams().size(); i++){
				String boundVarName = _params.getTrainFunctionReturnParams().get(i); 
				Data boundValue = returnedVariables.get(trainpb.getOutputParams().get(i).getName());
				if (boundValue == null)
					throw new DMLUnsupportedOperationException(boundVarName + " was not assigned a return value");
				this.getVariables().put(boundVarName, boundValue); // update variables in EL program block
				//mv the output files to new paths that wont be overwritten in next fold itern!
				String filepathname0 = ((FileObject)(returnedVariables.get(trainpb.getOutputParams().get(i).getName()))).getFilePath();
				String filepathname1 = _pp.sfmapfile.replaceAll("hashfile", "model-")+foldId+"-"+i;	//the hdfs filename for the train output
				FileCPInstruction instMoveFile = null;
				try {
					instMoveFile = (FileCPInstruction) FileCPInstruction.parseInstruction("mv:::" + filepathname0 + ":::" + filepathname1);
				} catch (DMLUnsupportedOperationException e) {
					e.printStackTrace();
				}
				instMoveFile.processInstruction(this); //this autom updates the matrices entry!
				System.out.println("$$$$ Adding as trainfuncretparams foldID:"+foldId+" paramNo:"+i+" varbl name: " + boundVarName + 
						" with new filepath: " + filepathname1);
				ensembleValue.put("trainFunctionOutputs-"+foldId+"-"+i, filepathname1);
			}
			this.setMetaData(trainpb.getMetaData()); 
			
			System.out.println("____________________________________");
			System.out.println("___ Variables in train after____");
			Iterator<Entry<String, Data>> it = trainpb.getVariables().entrySet().iterator();
			while (it.hasNext()) {
				Entry<String,Data> pairs = it.next();
			    System.out.println("  " + pairs.getKey() + " = " + pairs.getValue());
			}
			System.out.println("___ Matrices ____");
			Iterator<Entry<String, MetaData>> mit = trainpb.getMetaData().entrySet().iterator();
			while (mit.hasNext()) {
				Entry<String,MetaData> pairs = mit.next();
			    System.out.println("  " + pairs.getKey() + " = " + pairs.getValue());
			}
			System.out.println("____________________________________");
			
			System.out.println("$$$$$$$$$$ Finished training $$$$$$$$$$$$");
			
			//8*********** Execute test function block **************** -> needed for boosting (update _vars after train op file mv above)
			FunctionProgramBlock testpb = _prog.getFunctionProgramBlock(_params.getTestFunctionName());

			// create bindings to formal parameters for training function call
			functionVariables = setFunctionVariables(testpb, _params.getTestFunctionFormalParams());
			
			testpb.setVariables(functionVariables);	//sets its _variables 
			testpb.setMetaData(this.getMetaData());	//sets its _matrices
			testpb.execute();
			
			returnedVariables = testpb.getVariables();  //add updated binding for retvars to the CV pgmblk vars
			for (int i=0; i< testpb.getOutputParams().size(); i++){
				String boundVarName = _params.getTestFunctionReturnParams().get(i); //  _boundOutputParamNames.get(i); 
				Data boundValue = returnedVariables.get(testpb.getOutputParams().get(i).getName());
				if (boundValue == null)
					throw new DMLUnsupportedOperationException(boundVarName + " was not assigned a return value");
				this.getVariables().put(boundVarName, boundValue); // update variables in CV program block
			}
			this.setMetaData(testpb.getMetaData()); //DRB: this is the place where we can check if the function wrote any files out.
			
			if (_pp.partitionOutputs.size() == 1) {		//only train fold
				String trainformalname = _pp.partitionOutputs.get(0);
				if(_pp.toReplicate == true) {	//if no replcn, files are automatically deleted and overwritten in MR jobs
					String filepathname = new String(((FileObject)_variables.get(trainformalname)).getFilePath()) ;
					FileCPInstruction instDelFile = null;
					try {
						instDelFile = (FileCPInstruction) FileCPInstruction.parseInstruction("rm:::" + filepathname);
					} catch (DMLUnsupportedOperationException e) {
						e.printStackTrace();
					}
					instDelFile.processInstruction(this); //this autom removes the matrices entry!
				}
				//_matrices.remove(((FileObject)_variables.get(trainformalname)).getFilePath());
				_variables.remove(trainformalname);
			}
			else {
				//## for boosting we have train and test folds!
			}	
			System.out.println("$$$$$$$$$$ Deleted temp files and entries $$$$$$$$$$$$";*/
			
		} // end for each fold
		
		/************** Put ensemble file in the varbls list of this pgm blk, to be used for Write() later! *********/
		//this._variables.put(_pp.ensembleName, ensembleValue); - this doenst work, since cant cast as Data
		//It is not of Matrix / Scalar types! So, we need a new generic Object datatype in DML!
		//instead we write it out as a tmp file somewhere? later, for write sttmt, we use it? TODO #####
		
		//***** for expts only, delete the hashmap file
		String filepathname = new String(_pp.sfmapfile);
		FileCPInstruction instDelFile = null;
		try {
			instDelFile = (FileCPInstruction) FileCPInstruction.parseInstruction("rm:::" + filepathname);
		} catch (DMLUnsupportedOperationException e) {
			e.printStackTrace();
		}
		instDelFile.processInstruction(this); //this autom removes the matrices entry!

		Statistics.stopRunTimer();
		System.out.println(Statistics.display()); //always print the time stats
		
	} // end execute
	
	
	/**
	 * 
	 * @param fpb Function program block for function being called
	 * @param formalParams the formal parameters function is being called with [NOTE: these are string values, 
	 * 			so arbitrary expressions as formal parameters are not supported]
	 * @return the binding of data values 
	 * @throws DMLUnsupportedOperationException
	 */
	HashMap<String, Data> setFunctionVariables(FunctionProgramBlock fpb, ArrayList<String> formalParams) throws DMLUnsupportedOperationException{
	
		HashMap<String, Data> retVal = new HashMap<String, Data>(); 
		
		for (int i=0; i<fpb.getInputParams().size();i++) {
			
			String currFormalParamName = fpb.getInputParams().get(i).getName();
			Data currFormalParamValue = null; 
			ValueType valType = fpb.getInputParams().get(i).getValueType();
			
			if (i > formalParams.size() || (this.getVariables().get(formalParams.get(i)) == null)){
				
				if (valType == ValueType.BOOLEAN){
					boolean defaultVal = (i > formalParams.size()) ? 
												new Boolean(fpb.getInputParams().get(i).getDefaultValue()).booleanValue() : 
													new Boolean(formalParams.get(i)).booleanValue();
					currFormalParamValue = new BooleanObject(defaultVal);
				}
				else if (valType == ValueType.DOUBLE){
					double defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? 
												new Double(fpb.getInputParams().get(i).getDefaultValue()).doubleValue() : 
													new Double(formalParams.get(i)).doubleValue();
					currFormalParamValue = new DoubleObject(defaultVal);
				}
				else if (valType == ValueType.INT){
					int defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? 
												new Integer(fpb.getInputParams().get(i).getDefaultValue()).intValue() : 
													new Integer(formalParams.get(i)).intValue();
					currFormalParamValue = new IntObject(defaultVal);
				}
				else if (valType == ValueType.STRING){
					String defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? 
												fpb.getInputParams().get(i).getDefaultValue() : formalParams.get(i);
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

	
	/**
	 * 
	 * @param fpb Function program block for function being called
	 * @param formalParams the formal parameters function is being called with 
	 * 								[NOTE: these are string values, so arbitrary expressions as formal parameters are not supported]
	 * @return the binding of data values 
	 * @throws DMLUnsupportedOperationException
	 */
	HashMap<String, Data> setFunctionVariables(FunctionStatement fstmt, ArrayList<String> formalParams) throws DMLUnsupportedOperationException{
	
		HashMap<String, Data> retVal = new HashMap<String, Data>(); 
		
		for (int i=0; i<fstmt.getInputParams().size();i++) {
			
			String currFormalParamName = fstmt.getInputParams().get(i).getName();
			Data currFormalParamValue = null; 
			ValueType valType = fstmt.getInputParams().get(i).getValueType();
			
			if (i > formalParams.size() || (this.getVariables().get(formalParams.get(i)) == null)){
				
				if (valType == ValueType.BOOLEAN){
					boolean defaultVal = (i > formalParams.size()) ? 
												new Boolean(fstmt.getInputParams().get(i).getDefaultValue()).booleanValue() : 
													new Boolean(formalParams.get(i)).booleanValue();
					currFormalParamValue = new BooleanObject(defaultVal);
				}
				else if (valType == ValueType.DOUBLE){
					double defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? 
												new Double(fstmt.getInputParams().get(i).getDefaultValue()).doubleValue() : 
													new Double(formalParams.get(i)).doubleValue();
					currFormalParamValue = new DoubleObject(defaultVal);
				}
				else if (valType == ValueType.INT){
					int defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? 
												new Integer(fstmt.getInputParams().get(i).getDefaultValue()).intValue() : 
													new Integer(formalParams.get(i)).intValue();
					currFormalParamValue = new IntObject(defaultVal);
				}
				else if (valType == ValueType.STRING){
					String defaultVal = (i > _params.getTrainFunctionFormalParams().size()) ? 
												fstmt.getInputParams().get(i).getDefaultValue() : formalParams.get(i);
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
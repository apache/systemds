/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import com.ibm.bi.dml.runtime.DMLRuntimeException;


public class CVProgramBlock extends ProgramBlock 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public CVProgramBlock(Program prog) throws DMLRuntimeException {
		super(prog);
		// TODO Auto-generated constructor stub
	}

	/*public void printMe() {
		for (Instruction i : this._inst) {
			i.printMe();
		}
	}

	MetaLearningFunctionParameters _params ;
	PartitionParams _pp ;
	
	public CVProgramBlock(Program prog, PartitionParams pp, MetaLearningFunctionParameters params, DMLConfig passedConfig)
	throws DMLRuntimeException {
		super(prog);
		_prog = prog;
		_params = params; 
		_pp = pp ;
	}

	protected void executePartition() throws DMLRuntimeException, DMLUnsupportedOperationException {
		LOG.trace("Variables: " + _variables.toString());

		for (int i = 0; i < _inst.size(); i++) {	//only one itern occurs though
			Instruction currInst = _inst.get(i);
			if (currInst instanceof MRJobInstruction) {
				MRJobInstruction currMRInst = (MRJobInstruction) currInst;
				//populate varbls table with output matrix filepaths
				for ( int index=0; index < currMRInst.getOutputVars().length; index++) {
					//Arun: now, the output matrices (A1, A2...) have filepaths "./data/A1"... 
					_variables.put(currMRInst.getOutputVars()[index], new FileObject(currMRInst.getOutputVars()[index], 
																			"" + currMRInst.getOutputVars()[index])); 
				}	
//TODO: shirish: here the outputs doesnt have "./data/", and so uses curr working dir, viz scripts/!
//better to set it in one place, rather than eveywhere! ok, let me skip using ./data/!
				
				JobReturn jb = RunMRJobs.submitJob(currMRInst, this);
				//Note that submitjob has the varblnames as inputs; runjob call takes in filepathsnames after updatelabels on varblname inputs
				if(jb.getMetaData().length != currMRInst.getOutputVars().length) {
					System.out.println("Error after partitioning in cv progm blk - no. matrices don't match!");
					System.exit(1);
				}
				//Populate returned stats into symbol table of matrices
				for ( int index=0; index < jb.getMetaData().length; index++) {
					// TODO: Fix This
					//_matrices.put(new String("" + currMRInst.getOutputVars()[index]), jb.getMetaData(index));
				}
				Statistics.setNoOfExecutedMRJobs(Statistics.getNoOfExecutedMRJobs() + 1);
			} else if (currInst instanceof CPInstruction) {
				String updInst = RunMRJobs.updateLabels(currInst.toString(), _variables);
				CPInstruction si = CPInstructionParser.parseSingleInstruction(updInst);
				si.processInstruction(this);
			} 
		}
		//delte the hashmap file!
		//### TODO
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
			//populate varbls table out matx filepaths //(A1re, A2re...) have filepaths "./data/A1re"... - now A1re itself 
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
				numrpbs[q] = (int)nr;	//new jr outputs fullrowcolblks
			}
			if(q > 0)
				reblksinsts += ",rblk:::"+q+":DOUBLE:::"+(q+nummats)+":DOUBLE:::1000:::1000";
		}
		public void setReBlockInstructions(String [] input, InputInfo [] inputInfo, long [] numRows, long [] numCols, 
		 * int [] num_rows_per_block, int [] num_cols_per_block, String mapperInstructions, 
		 * String reblockInstructions, String otherInstructions, String [] output, OutputInfo [] outputInfo, byte [] resultIndex, 
		 * byte[] resultDimsUnknown, int numReducers, int replication, HashSet <String> inLabels, HashSet <String> outLabels)
		
		
		// TODO: following setReblockInstructions() is commented since this does not adhere to the new method of setting up MR jobs.
		// TODO: One must create input and output variables and use them to set up the jobs, instead of reblkinps and reblkouts
		//reblksmr.setReBlockInstructions(reblksinps, reblksinpsinfos, numrows, numcols, numrpbs, numcpbs, "", reblksinsts, "",
		//		reblksouts, reblksoutsinfos, resinds, resdims, 1, 1, null, null);	//as per statiko
		
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
			//if(filexists on hdfs)	//#### how?! TODO ####
			//	delete file on hdfs;
			FileCPInstruction instDelFile = null;
			try {
				instDelFile = (FileCPInstruction) FileCPInstruction.parseInstruction("rm:::" + filepathname);
			} catch (DMLUnsupportedOperationException e) {
				// TODO Auto-generated catch block
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
		PartitionParams.AccessPath retapt;
		Runtime runtime = Runtime.getRuntime();
		long maxMemory = runtime.maxMemory();  
		long allocatedMemory = runtime.totalMemory();  
		long freeMemory = runtime.freeMemory();  
		long availMemory = freeMemory + (maxMemory - allocatedMemory);	//current avail mem in bytes
		//######### TODO: we cannot check memsize on each mapper indeply!! so, should we check here? or assume fixed size? ###########
		//long availmem = 100 * (1 << 20); 	
		long availmem = availMemory / 2;	//TODO: use only 50% of avail memory
		
		//based on rough experiments, the hashmap size formula is roughly 138 N + 8 NV
		long N = (_pp.isColumn == false) ? nr : nc;	//numrows or numcols as numkeys
		long V = 1;
		if(_pp.cvt == PartitionParams.CrossvalType.kfold)
			V = _pp.numFolds;		//tuple has k values
		else if ((_pp.cvt == PartitionParams.CrossvalType.holdout) || 
						(_pp.cvt == PartitionParams.CrossvalType.bootstrap)) {
			V = (_pp.toReplicate == false) ? 1 : _pp.numIterations;
		}
		else {
			System.out.println("Unsupported method in Run MR jobs!");
			System.exit(1);
		}
		//long hashmapsize = 138 * N + 8 * N * V; - abandoned java hashmap
long hashmapsize = N * V * 8 + 15000; //(for small vector overhead)
		System.out.println("$$$$$$$$$$$$$ Avail memory: " + ((double)availmem/1024/1024) + 
				" and hashmap size: " + ((double)hashmapsize/1024/1024) + "MB $$$$$$$$$$$$$");
//availmem = 1;	//FOR DEBUGGING!
		if(hashmapsize <= availmem)	//we can simply use the hashmap MR job!
			retapt = PartitionParams.AccessPath.HM;
		else if (_pp.cvt == PartitionParams.CrossvalType.bootstrap)	//hashmap doesnt fit in memory; so only JR for bootstrap
			retapt = PartitionParams.AccessPath.JR;
		else {		 //hashmap doesnt fit in memory! so use row cost formulae
			retapt = PartitionParams.AccessPath.JR;		//by default, use JR
			long Nother = (_pp.isColumn == false) ? nc : nr;		//the other dimension
			if(Nother < 2.5)	//based on shuffle cost formulae for all cv, el bagging
				retapt = PartitionParams.AccessPath.RB;		//accesspath = 1;
		}
		
if(hashmapsize <= availmem)	//we can simply use the hashmap MR job!
return PartitionParams.AccessPath.HM;	//FOR DEBUGGING
else {
System.out.println("Not enough memory, HM dies!");
System.exit(1);
} //FORDEBUGGING

return retapt;
//return PartitionParams.AccessPath.JR;	//FOR DEBUGGING
//return PartitionParams.AccessPath.RB;	//FOR DEBUGGING
	}
	
	public void execute() throws DMLRuntimeException, DMLUnsupportedOperationException{
		//the accesspath has to be computed here (and not in runmrjobs) since we need to know if post partition reblocks are needed!
		
		// TODO: following code for setting up nr, nc, bnr, bnc are commented. These values 
		// must be pulled from the symbol table.
		
		long nr = -1; // ((MRJobInstruction) _inst.get(0)).getIv_rows()[0];	//only one instcn exists, so use it for input matr stats
		long nc = -1; // ((MRJobInstruction) _inst.get(0)).getIv_cols()[0];
		int bnr = -1; // ((MRJobInstruction) _inst.get(0)).getIv_num_rows_per_block()[0];
		int bnc = -1; // ((MRJobInstruction) _inst.get(0)).getIv_num_cols_per_block()[0];
		_pp.apt = computeAccessPath(nr, nc, bnr, bnc);
		((MRJobInstruction) _inst.get(0)).getPartitionParams().apt = _pp.apt;	//#######modify pp in partn instrcn - TODO: chk if this is allowed!
		System.out.println("$$$$$$$$$$$$$ Chosen accesspath: " + _pp.apt + "$$$$$$$$$$$$$");
		//based on torepl, we need to execute partition once outside the for loop vs repeatedly inside the loop		
		*//********** Construct folds for partition with replication ****************//* 
		if(_pp.toReplicate == true) {	//partition w repl
			executePartition();
			System.out.println("Finished executing partition w repl!");
			if(_pp.apt == PartitionParams.AccessPath.RB || _pp.apt == PartitionParams.AccessPath.JR) {	//do post partn reblocks for RB and JR
				executeReblocks(nr, nc, bnr, bnc);
				System.out.println("Finished executing reblocks on outputs for JR/RB w repl!");
			}
		}
		
		String outputVar = _params.getErrorAggFormalParams().get(0);
		this.setVariable(outputVar, new DoubleObject(0)) ;
		
		//iterate thro folds and do train/test/agg/writeout if w repl; for wo repl, we need to create fold by invoking partition again!
		int foldCount = -1;
		//if (_pp.isEL == true || (_pp.isEL == false && _pp.cvt != PartitionParams.CrossvalType.kfold))
		if (_pp.cvt == PartitionParams.CrossvalType.holdout || _pp.cvt == PartitionParams.CrossvalType.bootstrap)
			foldCount = _pp.numIterations;
		else if (_pp.cvt == PartitionParams.CrossvalType.kfold)
			foldCount = _pp.numFolds;
		
		//TODO Arun: need to handle ############# submatrix and cell ########!! current focus is only row!!!!
		//TODO Arun: need to handle ######### kfold (row) ###########!! if we produce only test folds; should we append matrcies there???
		
		*//******************* Iterate over the folds **************************//* 
		for(int foldId = 0 ; foldId < foldCount; foldId++) {
			*//********** Construct folds for partition without replication ****************//*
			if(_pp.toReplicate == false) {	//partition wo repl - invoke it and do post partn reblock if ncsry
				executePartition();
				System.out.println("Finished executing partition wo repl for fold "+foldId+"!");
				if(_pp.apt == PartitionParams.AccessPath.RB || _pp.apt == PartitionParams.AccessPath.JR) {	//do post partn reblocks for RB and JR
					executeReblocks(nr, nc, bnr, bnc);
					System.out.println("Finished executing reblock on fold "+foldId+" for JR/RB wo repl!");
				}
			}
			//for mappings to trainer/test, several cases can arise dep on access path, partn type and repl: (after stmt level reconciln) 
			 row partngn with repl:
			 * 		holdout / kfold: HM: feed in outputs[T+i] to train and outputs[i] to test (since test folds occur first by convention!)
			 * 		boostrap: HM: outputs[i] to train, original input to test!
			 * In all the above, wo repl means T is 1 and i is 0! Also, if apt is RB or JR, append "re" to fold aliters
			 *	TODO: kfold wo repl - we need so 'stitch' together test folds to get train fold! also TODO submatrix and cell partng TODO
			 *
			//reconcile formal funcnt argmt varblnames with cvstmt varbl names with actual filepathnames!
			//the func varbls mapping below maps from function's inp varbls in cv stmt to func's inp varbls in func dml
			//the inputs to func in cv stmt basically resuses (some of) partn output varbls
			//so, we only need to bind partn output varbl in this fold to correct filepathname and metadata in _variables and _matrices
			//thus, we use above list of cases, and enact this binding for this fold!
			System.out.println("$$$$$ In cv pgmblk, pt:" + _pp.pt+ ", cvt:"+ _pp.cvt + ", repl:"+_pp.toReplicate + "$$$$$$$$$$$$$");
			if(_pp.pt == PartitionParams.PartitionType.row) {
				if(_pp.cvt == PartitionParams.CrossvalType.holdout || (_pp.cvt == PartitionParams.CrossvalType.kfold && 
																			_pp.toReplicate == true)) {	//kfold w repl or holdout w/wo repl
					String testformalname = _pp.partitionOutputs.get(0);
					String trainformalname = _pp.partitionOutputs.get(1);
					String testaliter = _pp.getOutputStrings()[2*foldId]; 
					String trainaliter = _pp.getOutputStrings()[2*foldId + 1];	//if wo repl, foldid will be 0!
					if(_pp.apt == PartitionParams.AccessPath.RB || _pp.apt == PartitionParams.AccessPath.JR) {
						testaliter += "re";
						trainaliter += "re";
					}
					//get map values of aliters and give it to formals //no need to update _matrices since filepath is same!
					_variables.put(testformalname, new FileObject(testformalname, ((FileObject)_variables.get(testaliter)).getFilePath()));
					_variables.put(trainformalname, new FileObject(trainformalname, ((FileObject)_variables.get(trainaliter)).getFilePath()));
					_variables.remove(testaliter);									//remove aliter varbls, not needed hence
					_variables.remove(trainaliter);
					System.out.println("$$$$$$ Setting up bindings for holdout/kfoldwrepl: trainformalname " + trainformalname +
							" binds to " + trainaliter + " and testformalname " + testformalname + " binds to " + testaliter); 
				}
				else if(_pp.cvt == PartitionParams.CrossvalType.bootstrap) {	//only train outputs of partition; test is orig input matrix!
					String trainformalname = _pp.partitionOutputs.get(0);
					String trainaliter = _pp.getOutputStrings()[foldId];
					if(_pp.apt == PartitionParams.AccessPath.JR) {
						trainaliter += "re";
					}
					_variables.put(trainformalname, new FileObject(trainformalname, ((FileObject)_variables.get(trainaliter)).getFilePath()));
					_variables.remove(trainaliter);
					System.out.println("$$$$$$ Setting up bindings for bootstrap: trainformalname " + trainformalname +
							" binds to " + trainaliter);
				}
				else {	//kfold wo repl - needs 'stitching' of test matrices into train matrix!
					//##### TODO ######
				}
			}
			else {	//submatrix and cell
				//##### TODO #####
			}
			
			//#### TODO #### We assume that the trainer does the projection on its own (ie split labeled data matrix!)
			//Also, we assume that prev defined varbls get passed on properly (matrix, constants, etc)
			//8*************************** Execute train function block *******************************8/
			FunctionProgramBlock trainpb = _prog.getFunctionProgramBlock(_params.getTrainFunctionName());	//get train pgm func blk
			
			// create bindings to formal parameters for training function call -> contains logic to bind
			HashMap<String, Data> functionVariables = setFunctionVariables(trainpb, _params.getTrainFunctionFormalParams());
						
			trainpb.setVariables(functionVariables);	//sets the _variables
			trainpb.setMetaData(this.getMetaData());	//sets the _matrices
			trainpb.execute();		// execute training function block
			
			HashMap<String,Data> returnedVariables = trainpb.getVariables(); //add updated binding for retvars to the CV pgmblk vars
			for (int i=0; i< trainpb.getOutputParams().size(); i++){
				String boundVarName = _params.getTrainFunctionReturnParams().get(i); 
				Data boundValue = returnedVariables.get(trainpb.getOutputParams().get(i).getName());
				if (boundValue == null)
					throw new DMLUnsupportedOperationException(boundVarName + " was not assigned a return value");
				this.getVariables().put(boundVarName, boundValue); // update variables in CV program block
			}
			this.setMetaData(trainpb.getMetaData()); //DRB: this is the place where we can check if the function wrote any files out.
			System.out.println("$$$$$$$$$$ Finished training on fold " + foldId + " $$$$$$$$$$$$");
			
			//8*************************** Execute test function block *******************************8/	
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

			System.out.println("$$$$$$$$$$ Finished testing on fold " + foldId + " $$$$$$$$$$$$");

			//#### delete aliter fold matrices files and remove corresp entries in vars and mats;
			// DRB: do I need to handle this?	//deleteTempFiles();
			if (_pp.partitionOutputs.size() == 1) {		//only train fold
				String trainformalname = _pp.partitionOutputs.get(0);
				if(_pp.toReplicate == true) {	//if no replcn, files are automatically deleted and overwritten in MR jobs
					String filepathname = new String(((FileObject)_variables.get(trainformalname)).getFilePath()) ;
					//if(filexists on hdfs)	//#### how?! TODO ####
					//	delete file on hdfs;
					FileCPInstruction instDelFile = null;
					try {
						instDelFile = (FileCPInstruction) FileCPInstruction.parseInstruction("rm:::" + filepathname);
					} catch (DMLUnsupportedOperationException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					instDelFile.processInstruction(this); //this autom removes the matrices entry!
				}
				//_matrices.remove(((FileObject)_variables.get(trainformalname)).getFilePath());
				_variables.remove(trainformalname);
			} 
			else if ((_pp.partitionOutputs.size() == 2) && (_pp.cvt == PartitionParams.CrossvalType.holdout || 
					(_pp.cvt == PartitionParams.CrossvalType.kfold && _pp.toReplicate == true))) {	//test+train for holdout w/wo repl, kfold w repl
				String testformalname = _pp.partitionOutputs.get(0);
				String trainformalname = _pp.partitionOutputs.get(1);
				if(_pp.toReplicate == true) {
					String filepathname0 = new String(((FileObject)_variables.get(testformalname)).getFilePath()) ;
					String filepathname1 = new String(((FileObject)_variables.get(trainformalname)).getFilePath()) ;
					//if(filexists on hdfs)	//#### how?! TODO ####
					//	delete file on hdfs;
					FileCPInstruction instDelFile0 = null, instDelFile1 = null;
					try {
						instDelFile0 = (FileCPInstruction) FileCPInstruction.parseInstruction("rm:::" + filepathname0);
					} catch (DMLUnsupportedOperationException e) {
						e.printStackTrace();
					}
					instDelFile0.processInstruction(this); //this autom removes the matrices entry!
					try {
						instDelFile1 = (FileCPInstruction) FileCPInstruction.parseInstruction("rm:::" + filepathname1);
					} catch (DMLUnsupportedOperationException e) {
						e.printStackTrace();
					}
					instDelFile1.processInstruction(this); //this autom removes the matrices entry!
				}
				//_matrices.remove(((FileObject)_variables.get(testformalname)).getFilePath());
				//_matrices.remove(((FileObject)_variables.get(trainformalname)).getFilePath());
				_variables.remove(testformalname);
				_variables.remove(trainformalname);
			}
			else { //kfold wo repl - stitching of matrices? also, submatrix and cell!
				//#### TODO ####
			}

			System.out.println("$$$$$$$$$$ Deleted temp files and entries on fold " + foldId + " $$$$$$$$$$$$");

			//#### TODO #####  Copy all the above to EL Pgm Blk as well! and modify accordingly!!;
			//#### TODO #####  EL Metadata file writeout!!! -> Perhaps a simple seqfile? With key value stores!;
			//#### TODO #### chk out why join MR job cups!! also why distr cache for hashmap cups!
			//#### TODO CV Error aggregn handling #######;
			
			//8********************** aggregate errors *************************8/
			// set the error outputs for aggregation		// constraint is that the - ???
			String testOutputName = _params.getTestFunctionReturnParams().get(0) ;
			VariableCPInstruction getError = (VariableCPInstruction) InstructionParser.parseSingleInstruction("assigndoublevar" + 
					Instruction.OPERAND_DELIM + testOutputName + foldId + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + 
					Instruction.OPERAND_DELIM + "iter" + foldId + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE);
			ScalarCPInstruction ssi = null;
			try {
				ssi = (ScalarCPInstruction) InstructionParser.parseSingleInstruction("+" + Instruction.OPERAND_DELIM + "iter" + foldId + 
						Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM + outputVar + Instruction.VALUETYPE_PREFIX + 
						ValueType.DOUBLE + Instruction.OPERAND_DELIM + outputVar + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE) ;
			}
			catch ( Exception e ) {
				e.printStackTrace();
			}
			getError.processInstruction(this) ;
			ssi.processInstruction(this) ;
			
			System.out.println("$$$$$$$$$$ End of for loop iteration " + foldId + " on folds! $$$$$$$$$$$$");

		} //8*********** end of for loop on folds *************8/ 
		
		// handle the aggregation of the errors across the folds to compute final error for CV
		if(_params.getAgg() == CVStatement.AGG.avg) {
			ScalarCPInstruction ssi = null;
			try {
				//ssi = new ScalarSimpleInstructions("/:::" + outputVar + ":::" + pp.numFolds + ":::" + outputVar) ;
				// ssi = (ScalarCPInstruction) InstructionParser.parseSingleInstruction("/:::" + outputVar + ":::" + pp.numFolds + ":::" + outputVar);
				ssi = (ScalarCPInstruction) InstructionParser.parseSingleInstruction("/" + Instruction.OPERAND_DELIM + outputVar + 
						Instruction.OPERAND_DELIM + _pp.numFolds + Instruction.OPERAND_DELIM + outputVar) ;
			}
			catch (Exception e) {
				e.printStackTrace();
			}
			ssi.processInstruction(this) ;
		}
		
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
	
	@Override
	protected SymbolTable createSymbolTable() {
		// TODO: override this function whenever CV implementation is revisited
		return null;
	}
	
	*//**
	 * 
	 * @param fpb Function program block for function being called
	 * @param formalParams the formal parameters function is being called with [NOTE: these are string values, 
	 * 			so arbitrary expressions as formal parameters are not supported]
	 * @return the binding of data values 
	 * @throws DMLUnsupportedOperationException
	 *//*
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

	
	*//**
	 * 
	 * @param fpb Function program block for function being called
	 * @param formalParams the formal parameters function is being called with 
	 * 								[NOTE: these are string values, so arbitrary expressions as formal parameters are not supported]
	 * @return the binding of data values 
	 * @throws DMLUnsupportedOperationException
	 *//*
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
*/
} // end class
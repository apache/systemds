package dml.meta;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;


import dml.lops.runtime.RunMRJobs;
import dml.parser.DataIdentifier;
import dml.runtime.instructions.CPInstructions.Data;
import dml.runtime.instructions.CPInstructions.FileObject;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;
import dml.utils.configuration.DMLConfig;

public class PartitionParams {
//<Arun> TODO was integrating column sampling!!! TODO need to change the MR stuff and other files for column!!
	//the el type implicitly determines partition type:
	//bagging is row boostrap; rsm is column holdout; boosting is row wtdsampling - all produce only train sets 
	//all 3 use the numIterations attribute below; bagging and rsm also use the frac and toTeplicate attributes
	public boolean isEL = false;	//flag to check if partn is part of a cv clause or el clause; default cv
	public enum EnsembleType {bagging, rsm, rowholdout, adaboost};	//supported el methods (adaboost not yet impl)
	public EnsembleType et;				//the current stmt's el type
	public String ensembleName;			//the name of the ensemble variable
	public boolean isColumn = false;	//this indicates whether we are doing row sampling or column sampling
										//column sampling is currently allowed only for holdout in cv, and rsm in el
										//the user explicitly states element=column only in cv;
										//col MR methods are integrated with row's MR methods using this flag (easier)
	public boolean isSupervised = true;	//flag used only for column sampling in cv, and rsm in el 
										//-> cv col doesnt make sensse; rsm is assumed to be supervised; if not use this flag!
	public String sfmapfile;	//the name of the hashmap file / join file used for partitioning
								//also this is used as prefix for the train output matrices
	public enum AccessPath {HM, JR, RB};
	public AccessPath apt;		//used for partition post processing (reblocks after RB, JR)
//</Arun>
	public enum CrossvalType {kfold, holdout, bootstrap} ; // For now 0,1,2.
	public enum PartitionType {row,cell,submatrix} ; // For now 0,1,2 -> used only for cv; in el, this is implicit in method
	
	public CrossvalType cvt ;
	public PartitionType pt ;

	private ArrayList<String> inputs; // the amtrix we want to partition
	public int numFolds = -1 ; //k-fold cross validation
	public int numRowGroups ; // h X l partitioning
	public int numColGroups ;
	public double frac ; // fraction in test
	public int idToStratify = -1; // column id on which to stratify our sampling
	public int numIterations = 1 ; // number of times to repeat, only for holdout right now - also bootstrapping
	public int idDomainSize ;
	public ArrayList<String> partitionOutputs ;
	public boolean toReplicate = false ;
	
	String scratch = "";
	
	public int rows_in_block = DMLConfig.DEFAULT_BLOCK_SIZE; 
	public int columns_in_block = DMLConfig.DEFAULT_BLOCK_SIZE; 
	
	public int get_rows_in_block() {
		return rows_in_block ;
	}
	public int get_columns_in_block() {
		return columns_in_block ;
	}

	public ArrayList<String> getInputs(){
		return this.inputs;
	}
	
	public int getNumSeedsPerMapper() {
		// For holdout -- numIterations
		// For stratify -- domainSize
		if(cvt == CrossvalType.kfold) {
			if(idToStratify == -1)
				return 1 ;
			else
				return idDomainSize ;
		}
		else if (cvt == CrossvalType.holdout) {
			if(idToStratify == -1)
				return 1 ;
			else
				return idDomainSize ;
		}

		return 1;
	}

	/* Needs to be called before for submatrix */
	public void numFoldsForSubMatrix(long numRows, long numColumns) {
		int rSize = (int) Math.ceil((double)numRows/(double)numRowGroups) ;
		int cSize = (int) Math.ceil((double)numColumns/(double)numColGroups) ;

		int ActualNumRowGroups = (int) Math.ceil((double)numRows/(double)rSize) ;
		int ActualNumColGroups = (int) Math.ceil((double)numColumns/(double)cSize) ;		

		numFolds = ActualNumRowGroups * ActualNumColGroups ;
	}
	/**
	 * getOutputStrings1(): output strings with replication 
	 * getOutputStrings2(): output strings without replication 
	 * @return
	 */
	
	public String[] getOutputStrings() {
		if (this.toReplicate)
			return getOutputStrings1();
		else
			return getOutputStrings2();
		
	}
	
	private String[] getOutputStrings1() {
		String[] outputs = null ;
		
		if(isEL == true) {	//el rather than cv; all el methods have numiterations as an attribute
			outputs = new String[numIterations] ;
			for(int i = 0 ; i < numIterations; i++) {
				outputs[i] = partitionOutputs.get(0) + i ;	//in el, partition has only one output (train set)
			}
			return outputs;
		}
		
		if(cvt == CrossvalType.kfold) {
			if(pt == PartitionType.row) {
				outputs = new String[2*numFolds] ;
				for(int i = 0 ; i <numFolds; i++) {
					// First test, then train
					outputs[2*i] = partitionOutputs.get(0) + i ;
					outputs[2*i+1] = partitionOutputs.get(1) + i ;
				}
			}
			else if(pt == PartitionType.submatrix) {
				outputs = new String[4 * numFolds] ;
				for(int i = 0 ; i < outputs.length; i++) {
					// A,B,C,D -- int that order
					if(i % 4 == 0)
						outputs[i] = partitionOutputs.get(0) + (i/4) ; 
					else if(i % 4 == 1)	
						outputs[i] = partitionOutputs.get(1) + (i/4) ;
					else if(i % 4 == 2)
						outputs[i] = partitionOutputs.get(2) + (i/4); 
					else
						outputs[i] = partitionOutputs.get(3) + (i/4);
				}
			}
		}

		else if (cvt == CrossvalType.holdout) {
			if(pt == PartitionType.row) {
				outputs = new String[2*numIterations] ;
				for(int i = 0 ; i < numIterations; i++) {
					// First test, then train
					outputs[2*i] = partitionOutputs.get(0)  + i ;
					outputs[2*i+1] = partitionOutputs.get(1)  + i ;
				}
			}
			
			else if(pt == PartitionType.cell) {
				outputs = new String[numIterations] ;
				for(int i = 0 ; i < numIterations; i++)
					outputs[i] = "S" + i ;
			}
		}

		else if (cvt == CrossvalType.bootstrap) {
			if(pt == PartitionType.row) {
				if(isColumn == true) {
					System.out.println("Sorry, bootstrap on columns not supported!");
					System.exit(1);
				}
				outputs = new String[numIterations] ;
				for(int i = 0 ; i <numIterations; i++)
					outputs[i] = partitionOutputs.get(0) + i;
				//outputs[i] = "ensemble" + i ;
			}	
			else {
				System.out.println("Sorry, bootstrap supported only for rows!");
				System.exit(1);
			}
		
		}
		return outputs ;
	}

	/**
	 * getOutputStrings2: Outputs of partition when replication is not used 
	 * @return
	 */
	private String[] getOutputStrings2() {
		String[] outputs = null ;
		
		if(isEL == true) {	//el rather than cv
			outputs = new String[1];
			outputs[0] = partitionOutputs.get(0) + "0";	//in el, partition has only one output (train set)
			return outputs;
		}
		
		if(cvt == CrossvalType.kfold) {
			if(pt == PartitionType.row) {
				outputs = new String[numFolds] ;
				for(int i = 0 ; i <numFolds; i++) {
					outputs[i] = partitionOutputs.get(0) + i ;
				}
			}
			else if(pt == PartitionType.submatrix) {
				outputs = new String[numFolds] ;
				for(int i = 0 ; i < outputs.length; i++) {
						outputs[i] = partitionOutputs.get(0) + i; 
				}
			}
		}

		else if (cvt == CrossvalType.holdout) {
			if(pt == PartitionType.row) {
				outputs = new String[2] ;
				outputs[0] = partitionOutputs.get(0) + "0";
				outputs[1] = partitionOutputs.get(1) + "0";	//first test, then train
				
			}
		
			// TODO xxx DRB suspecious???
			else if(pt == PartitionType.cell) {
				outputs = new String[numIterations ] ;
				for(int i = 0 ; i < numIterations; i++)
					outputs[i] = "S" + i ;
			}
		}

		else if (cvt == CrossvalType.bootstrap) {
			if(pt == PartitionType.row) {
				outputs = new String[1];
				outputs[0] = partitionOutputs.get(0) + "0";
			}
			else {
				System.out.println("Sorry, Bootstrap supports only row!");
				System.exit(1);
			}
			/*outputs = new String[numFolds] ;
			for(int i = 0 ; i <numFolds; i++)
				outputs[i] = "ensemble" + i ;
			*/
		}
		return outputs ;
	}
	
	private int getResultIndexesLength1() {
		if(isEL == true) {
			return numIterations;
		}
		int resultIndexesLength = 1 ;
		if(cvt == CrossvalType.kfold) {
			if(pt == PartitionType.row)
				resultIndexesLength = 2*numFolds ;
			else if(pt == PartitionType.submatrix)
				resultIndexesLength = 4*numFolds ;
		}

		else if (cvt == CrossvalType.holdout) {
			if(pt == PartitionType.row)
				resultIndexesLength = 2*numIterations ;
			else if(pt == PartitionType.cell)
				resultIndexesLength = numIterations ;
		}

		else if (cvt == CrossvalType.bootstrap)
			resultIndexesLength = numIterations;

		return resultIndexesLength ;
	}

	private int getResultIndexesLength2() {
		if(isEL == true) {
			return 1;
		}
		int resultIndexesLength = 1 ;
		if(cvt == CrossvalType.kfold) {
			if(pt == PartitionType.row || pt == PartitionType.submatrix)
				resultIndexesLength = numFolds;
		}

		else if (cvt == CrossvalType.holdout) {
			if(pt == PartitionType.row)
				resultIndexesLength = 2;//*numIterations ; TODO Arun: without replcn, so only twice!
			else if(pt == PartitionType.cell)
				resultIndexesLength = numIterations ;
		}

		else if (cvt == CrossvalType.bootstrap)
			resultIndexesLength = 1;

		return resultIndexesLength ;
	}
	
	/*public long[] getCounters() {
		return new long[getResultIndexesLength()] ;
	}*/

	public byte[] getResultIndexes() {
		int resultIndexesLength = (this.toReplicate) ? getResultIndexesLength1() : getResultIndexesLength2();

		byte[] resultIndexes = new byte[resultIndexesLength] ;
		for(int i = 0 ; i < resultIndexes.length; i++)
			resultIndexes[i] = (byte) i;
		return resultIndexes ;
	}

	public byte[] getResultDimsUnknown() {
		int resultIndexesLength = (this.toReplicate) ? getResultIndexesLength1() : getResultIndexesLength2();

		byte[] resultDimsUnknown = new byte[resultIndexesLength] ;
		for(int i = 0 ; i < resultDimsUnknown.length; i++)
			resultDimsUnknown[i] = (byte) 1; // the result dimensions are unknown for every output 
		return resultDimsUnknown ;
	}

	public void set(ArrayList<String> inputs, CrossvalType cvt, PartitionType pt) {
		this.inputs = inputs;
		this.cvt = cvt ;
		this.pt = pt ;
	}

	public PartitionParams(ArrayList<String> inputs, CrossvalType cvt, PartitionType pt, int param1, int param2) {
		set(inputs, cvt, pt) ;
		if(pt == PartitionType.row) {
			this.numFolds = param1 ;
			this.idToStratify = param2 ;
		}
		else if(pt == PartitionType.submatrix) {
			this.numRowGroups = param1 ;
			this.numColGroups = param2 ;
		}
	}

	public PartitionParams(ArrayList<String> input, CrossvalType cvt, PartitionType pt, double frac, int idToStratify){
		set(input,cvt,pt) ;
		this.frac = frac ;
		if(frac >= 1) {
			System.out.println("The value of frac must be less than 1") ;
			System.exit(-1) ;
		}
		this.idToStratify = idToStratify ;
	}
	
	public PartitionParams(ArrayList<String> input, EnsembleType et, int numiter, double frac){
		this.inputs = input;
		this.et = et;
		this.numIterations = numiter;
		this.frac = frac;
		if(frac >= 1) {
			System.out.println("The value of frac must be less than 1") ;
			System.exit(-1) ;
		}
	}
	
	public String toString(){
		StringBuffer sb = new StringBuffer();
		sb.append("Data = ");
		for (String input : inputs){
			sb.append(input + " ");
		}
		if(isEL == false) 
			sb.append("," + "Type = " + cvt + "," + "Element = " + pt);
		else
			sb.append("," + "Method = " + et);
		return sb.toString();
	}

	public PartitionParams() {}

	public HashMap<String, Data> getOutputLabelValueMapping() {
		HashMap<String, Data> output = new HashMap<String, Data>() ;
		String[] opStrings = getOutputStrings();
		for(int i = 0 ; i < opStrings.length; i++)
			output.put(opStrings[i], new FileObject(opStrings[i], "" + opStrings[i])) ;	//varbl name maps to path "/data/varblname"

		return output ;
	}
	
	public Pair<long[],long[]> getRowAndColumnLengths(long numRows, long numColumns, int brlen, int bclen) {
		int rSize = (int) Math.ceil((double)numRows/(double)numRowGroups) ;
		int cSize = (int) Math.ceil((double)numColumns/(double)numColGroups) ;
		int x = rSize % brlen ;
		if(x != 0) rSize = rSize + x ;
		int y = cSize % bclen ;
		if(y != 0) cSize = cSize + y ;
		return getLengths(numRows, numColumns, rSize, cSize) ;
	}
	
	public Pair<long[],long[]> getRowAndColumnLengths(long numRows, long numColumns) {
		int rSize = (int) Math.ceil((double)numRows/(double)numRowGroups) ;
		int cSize = (int) Math.ceil((double)numColumns/(double)numColGroups) ;
		return getLengths(numRows, numColumns, rSize, cSize) ;
	}
	
	/**
	 * DRB: suspecious
	 */
	public Pair<long[],long[]> getLengths(long numRows, long numColumns, int rSize, int cSize) {
		/*int rSize = (int) Math.ceil((double)numRows/(double)numRowGroups) ;
		int cSize = (int) Math.ceil((double)numColumns/(double)numColGroups) ;*/

		int ActualNumRowGroups = (int) Math.ceil((double)numRows/(double)rSize) ;
		int ActualNumColGroups = (int) Math.ceil((double)numColumns/(double)cSize) ;		

		numFolds = ActualNumRowGroups * ActualNumColGroups ;

		//HashMap<Integer, DataIdentifier> map = new HashMap<Integer, DataIdentifier>() ;
		long[] rowArray = new long[numFolds * 4] ;
		long[] colArray = new long[numFolds * 4] ;

		long rowA, colA ;
		int ctr = 0 ;
		for(int i = 0 ; i < ActualNumRowGroups; i++) {
			for(int j = 0 ; j < ActualNumColGroups; j++) {
				// Suppose matrix block i,j is picked as A.
				if(i != ActualNumRowGroups -1)
					rowA = rSize;
				else {
					rowA = numRows % rSize ;
					if(rowA == 0)
						rowA = rSize ;
				}

				if(j != ActualNumRowGroups -1)
					colA = cSize;
				else {
					colA = numColumns % cSize ;
					if(colA == 0)
						colA = cSize ;
				}

				rowArray[4*(i*ActualNumColGroups + j)] = rowA  ;
				rowArray[4*(i*ActualNumColGroups + j) + 1] = rowA  ;
				rowArray[4*(i*ActualNumColGroups + j) + 2] = (numRows-rowA) ;
				rowArray[4*(i*ActualNumColGroups + j) + 3] = (numRows-rowA) ;

				colArray[4*(i*ActualNumColGroups + j)] =  colA ;
				colArray[4*(i*ActualNumColGroups + j) + 1] = (numColumns-colA) ;
				colArray[4*(i*ActualNumColGroups + j) + 2] =  colA ;
				colArray[4*(i*ActualNumColGroups + j) + 3] = (numColumns-colA) ;

			}	
		}
		return new Pair<long[],long[]>(rowArray, colArray) ;
	}
	public void setScratchPrefix(String scratch_dir) {

		scratch = scratch_dir;
		
	}

}
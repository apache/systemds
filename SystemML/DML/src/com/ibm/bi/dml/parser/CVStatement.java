package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.utils.LanguageException;


public class CVStatement extends Statement {
	
	private ArrayList<String> _inputNames;	
	private PartitionParams _pp;
	private MetaLearningFunctionParameters _params;
	private AGG _agg;
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		LOG.error(this.printErrorLocation() + "should not call rewriteStatement for CVStatement");
		throw new LanguageException(this.printErrorLocation() + "should not call rewriteStatement for CVStatement");
	}
	
	public enum AGG {
		sum, avg
	};

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append(_pp.toString());
		sb.append("train: " + _params.getTrainFunctionName());
		sb.append("test: " + _params.getTestFunctionName());
		sb.append("agg: " + _agg);
		sb.append("\n");
		return sb.toString();
	}

	public ArrayList<String> getInputNames(){
		return _inputNames;
	}
	
	public void initializePartitionParams(HashMap<String, String> map) {
		// Initialize _pp using map; //TODO: handle column sampling 
		if(map.containsKey("method") == false || map.containsKey("element") == false) {
			LOG.error("Error in cv statement: Need to specify both method and element!");
			System.exit(1);
		}
		String element = map.get("element");
		LOG.debug("$$$$$$$ cv element is " + element + ", type is " + map.get("method") + " $$$$$$$$$$");
		
		// case: k-fold
		if (map.get("method").equals("kfold")) {
			int numFolds, numRowGroups, numColGroups;
			
			if (element.equals("row")) {
				numFolds = (new Integer(map.get("numfolds"))).intValue();
				_pp = new PartitionParams(_inputNames, PartitionParams.CrossvalType.kfold,
						PartitionParams.PartitionType.row, numFolds, -1);
			} 
			else if (element.equals("submatrix")) {
				numRowGroups = (new Integer(map.get("numrowgroups"))).intValue();
				numColGroups = (new Integer(map.get("numcolgroups"))).intValue();
				_pp = new PartitionParams(_inputNames, PartitionParams.CrossvalType.kfold,
						PartitionParams.PartitionType.submatrix, numRowGroups, numColGroups);
			} 
			else if (element.equals("cell")) {
				LOG.error("Partitioning method currently unsupported in the framework");
				System.exit(-1);
			}
			else {
				LOG.error("kfold with column not supported!");
				System.exit(1);
			}
		}//end if kfold

		else if (map.get("method").equals("holdout")) {
			double frac;
			int numIterations = 1;
			if (element.equals("row")) {
				frac = new Double(map.get("frac")).doubleValue();
				numIterations = new Integer(map.get("numiterations")).intValue();
				_pp = new PartitionParams(_inputNames, PartitionParams.CrossvalType.holdout,
						PartitionParams.PartitionType.row, frac, -1);
				_pp.numIterations = numIterations;
			}			
			else if (element.equals("column")) {
				LOG.error("Column sampling for CV doesn't make sense; not supported!");
				System.exit(1);
				/*frac = new Double(map.get("frac")).doubleValue();
				numIterations = new Integer(map.get("numiterations")).intValue();
				_pp = new PartitionParams(_inputNames, PartitionParams.CrossvalType.holdout,
						PartitionParams.PartitionType.row, frac, -1);
				_pp.numIterations = numIterations;
				_pp.isColumn = true;
				//since it is columnar, check if it is supervised or not (default is yes)
				if(map.containsKey("supervised")) {
					if(map.get("supervised").equals("yes"))
						_pp.isSupervised = true;
					else if(map.get("supervised").equals("no"))
						_pp.isSupervised = 	false;
					else {
						System.out.println("Unrecognized value for supervised!");
						System.exit(1);
					}
				}
				else {
					System.out.println("Need to specify if supervised or not for column!");
					System.exit(1);
				}*/
			}
			else if (element.equals("submatrix")) {
				LOG.error("Partitioning method currently unsupported in the framework");
				System.exit(-1);
			}
			else if (element.equals("cell")) {
				frac = (new Double(map.get("frac"))).doubleValue();
				numIterations = new Integer(map.get("numiterations")).intValue();
				_pp = new PartitionParams(_inputNames, PartitionParams.CrossvalType.holdout,
						PartitionParams.PartitionType.cell, frac, -1);
				_pp.numIterations = numIterations;
			}
		}//end if holdout
		else if (map.get("method").equals("bootstrap")) {
			if (element.equals("row")) {
				double frac = new Double(map.get("frac")).doubleValue();
				int numIterations = new Integer(map.get("numiterations")).intValue();
				_pp = new PartitionParams(_inputNames, PartitionParams.CrossvalType.bootstrap,
						PartitionParams.PartitionType.row, frac, -1);
				_pp.numIterations = numIterations;
			} 
			else {
				LOG.error("Bootstrapping supported only for row partitions");
				System.exit(-1);
			}
		}
		if(map.containsKey("replicate")) {
			if(map.get("replicate").equals("true")) {
				_pp.toReplicate = true;
				LOG.debug ("$$$$$$$$ Replication set to true $$$$$$$$");
			}
		}
		_pp.partitionOutputs = new ArrayList<String>();
		_pp.partitionOutputs = _params.getPartitionReturnParams();
		_pp.sfmapfile = "cv-" + System.currentTimeMillis() + "-hashfile";
	}

	/**
	 * 
	 * @param input 	name of the data being cross-validated
	 * @param map 	 	stores the partition parameters
	 * @param params 	CV operator parameters -- listed below:
	 * 
	 */
	public CVStatement(ArrayList<String> inputs, HashMap<String, String> map, MetaLearningFunctionParameters params) {
		_inputNames = inputs;
		_params = params;			
		initializePartitionParams(map);
		LOG.debug("Input[0] is " + inputs.get(0));
	}

	public PartitionParams getPartitionParams() {
		return _pp;
	}

	@Override
	public boolean controlStatement() {
		return false;
	}

	@Override
	public VariableSet initializebackwardLV(VariableSet lo) {
		return null;
	}

	@Override
	public void initializeforwardLV(VariableSet activeIn) {
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet set = new VariableSet();
		for (String input : _inputNames){
			set.addVariable(input, new DataIdentifier(input));
		}
		return set;
	}

	@Override
	// only update 
	public VariableSet variablesUpdated() {
		VariableSet set = new VariableSet();
		for (String var : _params.getErrorAggReturnParams())
			set.addVariable(var, new DataIdentifier(var));
		return set;
	}

	public MetaLearningFunctionParameters getFunctionParameters() {
		return _params;
	}
}

package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.utils.LanguageException;

public class ELStatement extends Statement {
	
	// variable name of the ensemble that is created
	private String _ensembleName;
	
	// variable name of the input datasets 
	private ArrayList<String> _inputNames;	
	
	// partition parameters
	private PartitionParams _pp;
	
	// function parameters for ensemble learning
	private MetaLearningFunctionParameters _params;
	
	public String getEnsembleName(){
		return _ensembleName;
	}
	
	public MetaLearningFunctionParameters getFunctionParameters(){
		return _params;
	}
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		throw new LanguageException(this.printErrorLocation() + "should not call rewriteStatement for ELStatement");
	}
	
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append(_pp.toString());
		sb.append("train: " + _params.getTrainFunctionName());
		if (_params.getTestFunctionName() != null)
			sb.append(" test: " + _params.getTestFunctionName());
		sb.append("\n");
		return sb.toString();
	}

	public ArrayList<String> getInputNames(){
		return _inputNames;
	}
	
	public void initializePartitionParams(HashMap<String, String> map) {
		// Initialize _pp using map; //TODO error catching handling
		String elmethod = map.get("method");
		LOG.debug("$$$$$$$ el method is " + elmethod + " $$$$$$$$$$");
		
		if (elmethod.equals("bagging") || elmethod.equals("rsm") || elmethod.equals("rowholdout")) {	//new el method rowholdout
			//numiterations and frac expected
			int numiter = (new Integer(map.get("numiterations"))).intValue();
			double frac = new Double(map.get("frac")).doubleValue();
			if(elmethod.equals("bagging") == false)	//but for bagging, the stmt gives train fraction!
				frac = 1 - frac; //we invert the frac to be consistent with cv (in cv frac is %test; in el it is %train as per dml)
			PartitionParams.EnsembleType et = elmethod.equals("rsm") ? PartitionParams.EnsembleType.rsm : 
					(elmethod.equals("bagging") ? PartitionParams.EnsembleType.bagging : PartitionParams.EnsembleType.rowholdout);
			_pp = new PartitionParams(_inputNames,et, numiter, frac);
			if(elmethod.equals("rsm")) {
				_pp.isColumn = true;		//el rsm is implicitly colwise
				_pp.isSupervised = true;	//el rsm is implicitly supervised (last col labels), so never used; if not, change this!
			}
		}
		else if (elmethod.equals("adaboost")) { 
			LOG.error("Adaboost not yet implemented!");
			System.exit(1);
		}
		
		if(map.containsKey("replicate") == true) {
			if(map.get("replicate").equals("true")) {
				_pp.toReplicate = true;
				LOG.debug ("$$$$$$$$ Replication set to true! $$$$$$$$");
			}
		}
		_pp.partitionOutputs = new ArrayList<String>();
		_pp.partitionOutputs = _params.getPartitionReturnParams();
		_pp.isEL = true;
		_pp.ensembleName = _ensembleName;
		_pp.sfmapfile = "el-" + System.currentTimeMillis() + "-hashfile";
	}

	/**
	 * 
	 * @param ename 	names of the ensemble being created
	 * @param inputs 	names of the data sets used to build the ensemble
	 * @param map 	 	stores the partition parameters
	 * @param params 	CV operator parameters -- listed below:
	 * 
	 */
	public ELStatement(String ename, ArrayList<String> inputs, HashMap<String, String> map, MetaLearningFunctionParameters params) {
		_ensembleName = ename;
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
		VariableSet set = new VariableSet() ;
		for (String input : _inputNames){
			set.addVariable(input, new DataIdentifier(input));	
		}
		return set;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet set = new VariableSet();
		set.addVariable(_ensembleName, new DataIdentifier(_ensembleName));
		return set;
	}
}

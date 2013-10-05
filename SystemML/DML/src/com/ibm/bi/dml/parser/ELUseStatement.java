/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

//package com.ibm.bi.dml.parser;
//
//import java.util.ArrayList;
//import java.util.HashMap;
//
//import com.ibm.bi.dml.meta.PartitionParams;
//import com.ibm.bi.dml.utils.LanguageException;
//
//
//public class ELUseStatement extends Statement {
//	
//	private String _ensembleName;
//	private ArrayList<String> _inputNames;	
//	private PartitionParams _pp;
//	private MetaLearningFunctionParameters _params;
//	private AGG _agg;
//	
//	public Statement rewriteStatement(String prefix) throws LanguageException{
//		LOG.error(this.printErrorLocation() + "should not call rewriteStatement for CVStatement");
//		throw new LanguageException(this.printErrorLocation() + "should not call rewriteStatement for CVStatement");
//	}
//	
//	public enum AGG {
//		sum, avg
//	};
//
//	@Override
//	public String toString() {
//		StringBuffer sb = new StringBuffer();
//		sb.append(_pp.toString());
//		sb.append("test: " + _params.getTestFunctionName());
//		sb.append("agg: " + _agg);
//		sb.append("\n");
//		return sb.toString();
//	}
//
//	public String getEnsembleName(){
//		return _ensembleName;
//	}
//	
//	public void setEnsembleName(String name){
//		_ensembleName = name;
//	}
//	
//	public ArrayList<String> getInputNames(){
//		return _inputNames;
//	}
//	
//	public void initializePartitionParams(HashMap<String, String> map) {
//		
//		// Initialize _pp using map
//		String element = map.get("element");
//		
//		// case: k-fold
//		if (map.get("method").equals("kfold")) {
//			
//			int numFolds, numRowGroups, numColGroups;
//			
//			if (element.equals("row")) {
//				numFolds = (new Integer(map.get("numfolds"))).intValue();
//				_pp = new PartitionParams(_inputNames, PartitionParams.CrossvalType.kfold,
//						PartitionParams.PartitionType.row, numFolds, -1);
//			} else if (element.equals("submatrix")) {
//				numRowGroups = (new Integer(map.get("numrowgroups"))).intValue();
//				numColGroups = (new Integer(map.get("numcolgroups"))).intValue();
//				_pp = new PartitionParams(_inputNames, PartitionParams.CrossvalType.kfold,
//						PartitionParams.PartitionType.submatrix, numRowGroups, numColGroups);
//
//			//	if (map.get("rows_in_block") != null)
//			//		_pp.rows_in_block = (new Integer(map.get("rows_in_block"))).intValue();
//			//	if (map.get("columns_in_block") != null)
//			//		_pp.columns_in_block = (new Integer(map.get("columns_in_block"))).intValue();
//			} else if (element.equals("cell")) {
//				LOG.error("Partitioning method currently unsupported in the framework");
//				System.exit(-1);
//			}
//		}
//
//		else if (map.get("method").equals("holdout")) {
//			double frac;
//			int numIterations = 1;
//			if (element.equals("row")) {
//				frac = new Double(map.get("frac")).doubleValue();
//				numIterations = new Integer(map.get("numIterations")).intValue();
//				_pp = new PartitionParams(_inputNames, PartitionParams.CrossvalType.holdout,
//						PartitionParams.PartitionType.row, frac, -1);
//				_pp.numIterations = numIterations;
//			}
//
//			else if (element.equals("submatrix")) {
//				LOG.error("Partitioning method currently unsupported in the framework");
//				System.exit(-1);
//			}
//
//			else if (element.equals("cell")) {
//				frac = (new Double(map.get("frac"))).doubleValue();
//				numIterations = new Integer(map.get("numIterations")).intValue();
//				_pp = new PartitionParams(_inputNames, PartitionParams.CrossvalType.holdout,
//						PartitionParams.PartitionType.cell, frac, -1);
//				_pp.numIterations = numIterations;
//			}
//		}
//
//		else if (map.get("method").equals("boostrap")) {
//			if (element.equals("row")) {
//				int numIterations = new Integer(map.get("numIterations")).intValue();
//				_pp = new PartitionParams(_inputNames, PartitionParams.CrossvalType.bootstrap,
//						PartitionParams.PartitionType.row, -1, -1);
//				_pp.numIterations = numIterations;
//			} else {
//				LOG.error("Bootstrapping supported only for row partitions");
//				System.exit(-1);
//			}
//		}
//		_pp.partitionOutputs = _params.getPartitionReturnParams();
//	}
//
//	/**
//	 * @param ensembleName 	name
//	 * @param inputs 		name of the data sets being used by the ensemble
//	 * @param map 	 		stores the partition parameters
//	 * @param params 		EL operator parameters -- listed below:
//	 * 
//	 */
//	public ELUseStatement(String eName, ArrayList<String> inputs, HashMap<String, String> map, MetaLearningFunctionParameters params) {
//		_ensembleName = eName;
//		_inputNames = inputs;
//		_params = params;			
//		initializePartitionParams(map);
//	}
//
//	public PartitionParams getPartitionParams() {
//		return _pp;
//	}
//
//	@Override
//	public boolean controlStatement() {
//		return false;
//	}
//
//	@Override
//	public VariableSet initializebackwardLV(VariableSet lo) {
//		return null;
//	}
//
//	@Override
//	public void initializeforwardLV(VariableSet activeIn) {
//	}
//
//	@Override
//	public VariableSet variablesRead() {
//		VariableSet set = new VariableSet() ;
//		for (String input : _inputNames)
//			set.addVariable(input, new DataIdentifier(input));
//		return set;
//	}
//
//	@Override
//	/*
//	public VariableSet variablesUpdated() {
//		VariableSet set = new VariableSet();
//		for (String var : _params.getErrorAggReturnParams())
//			set.addVariable(var, new DataIdentifier(var));
//		return set;
//	}
//	*/
//	
//	public VariableSet variablesUpdated() {
//		VariableSet set = new VariableSet();
//		return set;
//	}
//	
//	public MetaLearningFunctionParameters getFunctionParameters() {
//		return _params;
//	}
//}

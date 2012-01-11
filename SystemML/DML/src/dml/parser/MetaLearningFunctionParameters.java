package dml.parser;

import java.util.ArrayList;

import dml.parser.CVStatement.AGG;

public class MetaLearningFunctionParameters {
	private String _trainFunctionName ;
	private ArrayList<String> _trainFunctionFormalParams ;
	private ArrayList<String> _trainFunctionReturnParams ;

	private String _testFunctionName ;
	private ArrayList<String> _testFunctionFormalParams ;
	private ArrayList<String> _testFunctionReturnParams ;
	
	private ArrayList<String> _partitionReturnParams ;
	
	private AGG _agg ;
	private ArrayList<String> _errorAggFormalParams ;
	private ArrayList<String> _errorAggReturnParams ;
	
	public MetaLearningFunctionParameters(
			String trainFunctionName, ArrayList<String> trainFunctionParams, ArrayList<String> trainFunctionOutputs, 
			String testFunctionName, ArrayList<String> testFunctionParams, ArrayList<String> testFunctionOutputs, 
			ArrayList<String> partitionOutputs, 
			AGG agg, ArrayList<String> errorAggFormalParams, ArrayList<String> errorAggReturnParams) {
		
		_trainFunctionName = trainFunctionName;
		_trainFunctionFormalParams = trainFunctionParams;
		_trainFunctionReturnParams = trainFunctionOutputs;
		
		_testFunctionName = testFunctionName;
		_testFunctionFormalParams = testFunctionParams;
		_testFunctionReturnParams = testFunctionOutputs;
	
		_partitionReturnParams = partitionOutputs ;
	
		_agg = agg;
		_errorAggFormalParams = errorAggFormalParams;
		_errorAggReturnParams = errorAggReturnParams;	
	
	}
	
	public String getTrainFunctionName() {
		return _trainFunctionName;
	}

	public void setTrainFunctionName(String trainFunctionName) {
		_trainFunctionName = trainFunctionName;
	}

	public ArrayList<String> getTrainFunctionFormalParams() {
		return _trainFunctionFormalParams;
	}

	public void setTrainFunctionFormalParams(
			ArrayList<String> trainFunctionFormalParams) {
		_trainFunctionFormalParams = trainFunctionFormalParams;
	}

	public ArrayList<String> getTrainFunctionReturnParams() {
		return _trainFunctionReturnParams;
	}

	public void setTrainFunctionReturnParams(
			ArrayList<String> trainFunctionReturnParams) {
		_trainFunctionReturnParams = trainFunctionReturnParams;
	}

	public String getTestFunctionName() {
		return _testFunctionName;
	}

	public void setTestFunctionName(String testFunctionName) {
		_testFunctionName = testFunctionName;
	}

	public ArrayList<String> getTestFunctionFormalParams() {
		return _testFunctionFormalParams;
	}

	public void setTestFunctionFormalParams(ArrayList<String> testFunctionFormalParams) {
		_testFunctionFormalParams = testFunctionFormalParams;
	}

	public ArrayList<String> getTestFunctionReturnParams() {
		return _testFunctionReturnParams;
	}

	public void setTestFunctionReturnParams(
			ArrayList<String> testFunctionReturnParams) {
		_testFunctionReturnParams = testFunctionReturnParams;
	}

	public ArrayList<String> getPartitionReturnParams() {
		return _partitionReturnParams;
	}

	public void setPartitionReturnParams(ArrayList<String> partitionReturnParams) {
		_partitionReturnParams = partitionReturnParams;
	}

	public AGG getAgg() {
		return _agg;
	}

	public void setAgg(AGG agg) {
		_agg = agg;
	}

	public ArrayList<String> getErrorAggFormalParams() {
		return _errorAggFormalParams;
	}

	public void setErrorAggFormalParams(ArrayList<String> errorAggFormalParams) {
		_errorAggFormalParams = errorAggFormalParams;
	}

	public ArrayList<String> getErrorAggReturnParams() {
		return _errorAggReturnParams;
	}

	public void setErrorAggReturnParams(ArrayList<String> errorAggReturnParams) {
		_errorAggReturnParams = errorAggReturnParams;
	}
	
}
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.api.mlcontext;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.instructions.cp.Data;

import scala.Tuple2;
import scala.Tuple3;
import scala.collection.JavaConversions;

/**
 * A Script object encapsulates a DML or PYDML script.
 *
 */
public class Script {

	/**
	 * The type of script ({@code ScriptType.DML} or {@code ScriptType.PYDML}).
	 */
	private ScriptType scriptType;
	/**
	 * The script content.
	 */
	private String scriptString;
	/**
	 * The optional name of the script.
	 */
	private String name;
	/**
	 * All inputs (input parameters ($) and input variables).
	 */
	private Map<String, Object> inputs = new LinkedHashMap<String, Object>();
	/**
	 * The input parameters ($).
	 */
	private Map<String, Object> inputParameters = new LinkedHashMap<String, Object>();
	/**
	 * The input variables.
	 */
	private Set<String> inputVariables = new LinkedHashSet<String>();
	/**
	 * The input matrix or frame metadata if present.
	 */
	private Map<String, Metadata> inputMetadata = new LinkedHashMap<String, Metadata>();
	/**
	 * The output variables.
	 */
	private Set<String> outputVariables = new LinkedHashSet<String>();
	/**
	 * The symbol table containing the data associated with variables.
	 */
	private LocalVariableMap symbolTable = new LocalVariableMap();
	/**
	 * The ScriptExecutor which is used to define the execution of the script.
	 */
	private ScriptExecutor scriptExecutor;
	/**
	 * The results of the execution of the script.
	 */
	private MLResults results;

	/**
	 * Script constructor, which by default creates a DML script.
	 */
	public Script() {
		scriptType = ScriptType.DML;
	}

	/**
	 * Script constructor, specifying the type of script ({@code ScriptType.DML}
	 * or {@code ScriptType.PYDML}).
	 *
	 * @param scriptType
	 *            {@code ScriptType.DML} or {@code ScriptType.PYDML}
	 */
	public Script(ScriptType scriptType) {
		this.scriptType = scriptType;
	}

	/**
	 * Script constructor, specifying the script content. By default, the script
	 * type is DML.
	 *
	 * @param scriptString
	 *            the script content as a string
	 */
	public Script(String scriptString) {
		this.scriptString = scriptString;
		this.scriptType = ScriptType.DML;
	}

	/**
	 * Script constructor, specifying the script content and the type of script
	 * (DML or PYDML).
	 *
	 * @param scriptString
	 *            the script content as a string
	 * @param scriptType
	 *            {@code ScriptType.DML} or {@code ScriptType.PYDML}
	 */
	public Script(String scriptString, ScriptType scriptType) {
		this.scriptString = scriptString;
		this.scriptType = scriptType;
	}

	/**
	 * Obtain the script type.
	 *
	 * @return {@code ScriptType.DML} or {@code ScriptType.PYDML}
	 */
	public ScriptType getScriptType() {
		return scriptType;
	}

	/**
	 * Set the type of script (DML or PYDML).
	 *
	 * @param scriptType
	 *            {@code ScriptType.DML} or {@code ScriptType.PYDML}
	 */
	public void setScriptType(ScriptType scriptType) {
		this.scriptType = scriptType;
	}

	/**
	 * Obtain the script string.
	 *
	 * @return the script string
	 */
	public String getScriptString() {
		return scriptString;
	}

	/**
	 * Set the script string.
	 *
	 * @param scriptString
	 *            the script string
	 * @return {@code this} Script object to allow chaining of methods
	 */
	public Script setScriptString(String scriptString) {
		this.scriptString = scriptString;
		return this;
	}

	/**
	 * Obtain the input variable names as an unmodifiable set of strings.
	 *
	 * @return the input variable names
	 */
	public Set<String> getInputVariables() {
		return Collections.unmodifiableSet(inputVariables);
	}

	/**
	 * Obtain the output variable names as an unmodifiable set of strings.
	 *
	 * @return the output variable names
	 */
	public Set<String> getOutputVariables() {
		return Collections.unmodifiableSet(outputVariables);
	}

	/**
	 * Obtain the symbol table, which is essentially a
	 * {@code HashMap<String, Data>} representing variables and their values.
	 *
	 * @return the symbol table
	 */
	public LocalVariableMap getSymbolTable() {
		return symbolTable;
	}

	/**
	 * Obtain an unmodifiable map of all inputs (parameters ($) and variables).
	 *
	 * @return all inputs to the script
	 */
	public Map<String, Object> getInputs() {
		return Collections.unmodifiableMap(inputs);
	}

	/**
	 * Obtain an unmodifiable map of input matrix/frame metadata.
	 *
	 * @return input matrix/frame metadata
	 */
	public Map<String, Metadata> getInputMetadata() {
		return Collections.unmodifiableMap(inputMetadata);
	}

	/**
	 * Pass a map of inputs to the script.
	 *
	 * @param inputs
	 *            map of inputs (parameters ($) and variables).
	 * @return {@code this} Script object to allow chaining of methods
	 */
	public Script in(Map<String, Object> inputs) {
		for (Entry<String, Object> input : inputs.entrySet()) {
			in(input.getKey(), input.getValue());
		}

		return this;
	}

	/**
	 * Pass a Scala Map of inputs to the script.
	 * <p>
	 * Note that the {@code Map} value type is not explicitly specified on this
	 * method because {@code [String, Any]} can't be recognized on the Java side
	 * since {@code Any} doesn't have an equivalent in the Java class hierarchy
	 * ({@code scala.Any} is a superclass of {@code scala.AnyRef}, which is
	 * equivalent to {@code java.lang.Object}). Therefore, specifying
	 * {@code scala.collection.Map<String, Object>} as an input parameter to
	 * this Java method is not encompassing enough and would require types such
	 * as a {@code scala.Double} to be cast using {@code asInstanceOf[AnyRef]}.
	 *
	 * @param inputs
	 *            Scala Map of inputs (parameters ($) and variables).
	 * @return {@code this} Script object to allow chaining of methods
	 */
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public Script in(scala.collection.Map<String, ?> inputs) {
		Map javaMap = JavaConversions.mapAsJavaMap(inputs);
		in(javaMap);

		return this;
	}

	/**
	 * Pass a Scala Seq of inputs to the script. The inputs are either two-value
	 * or three-value tuples, where the first value is the variable name, the
	 * second value is the variable value, and the third optional value is the
	 * metadata.
	 *
	 * @param inputs
	 *            Scala Seq of inputs (parameters ($) and variables).
	 * @return {@code this} Script object to allow chaining of methods
	 */
	public Script in(scala.collection.Seq<Object> inputs) {
		List<Object> list = JavaConversions.seqAsJavaList(inputs);
		for (Object obj : list) {
			if (obj instanceof Tuple3) {
				@SuppressWarnings("unchecked")
				Tuple3<String, Object, MatrixMetadata> t3 = (Tuple3<String, Object, MatrixMetadata>) obj;
				in(t3._1(), t3._2(), t3._3());
			} else if (obj instanceof Tuple2) {
				@SuppressWarnings("unchecked")
				Tuple2<String, Object> t2 = (Tuple2<String, Object>) obj;
				in(t2._1(), t2._2());
			} else {
				throw new MLContextException("Only Tuples of 2 or 3 values are permitted");
			}
		}
		return this;
	}

	/**
	 * Obtain an unmodifiable map of all input parameters ($).
	 *
	 * @return input parameters ($)
	 */
	public Map<String, Object> getInputParameters() {
		return inputParameters;
	}

	/**
	 * Register an input (parameter ($) or variable).
	 *
	 * @param name
	 *            name of the input
	 * @param value
	 *            value of the input
	 * @return {@code this} Script object to allow chaining of methods
	 */
	public Script in(String name, Object value) {
		return in(name, value, null);
	}

	/**
	 * Register an input (parameter ($) or variable) with optional matrix
	 * metadata.
	 *
	 * @param name
	 *            name of the input
	 * @param value
	 *            value of the input
	 * @param metadata
	 *            optional matrix/frame metadata
	 * @return {@code this} Script object to allow chaining of methods
	 */
	public Script in(String name, Object value, Metadata metadata) {
		MLContextUtil.checkInputValueType(name, value);
		if (inputs == null) {
			inputs = new LinkedHashMap<String, Object>();
		}
		inputs.put(name, value);

		if (name.startsWith("$")) {
			MLContextUtil.checkInputParameterType(name, value);
			if (inputParameters == null) {
				inputParameters = new LinkedHashMap<String, Object>();
			}
			inputParameters.put(name, value);
		} 
		else {
			Data data = MLContextUtil.convertInputType(name, value, metadata);
			if (data != null) {
				//store input variable name and data
				symbolTable.put(name, data);
				inputVariables.add(name);
				
				//store matrix/frame meta data and disable variable cleanup
				if( data instanceof CacheableData ) {
					if( metadata != null )
						inputMetadata.put(name, metadata);
					((CacheableData<?>)data).enableCleanup(false);
				}
			}
		}
		return this;
	}

	/**
	 * Register an output variable.
	 *
	 * @param outputName
	 *            name of the output variable
	 * @return {@code this} Script object to allow chaining of methods
	 */
	public Script out(String outputName) {
		outputVariables.add(outputName);
		return this;
	}

	/**
	 * Register output variables.
	 *
	 * @param outputNames
	 *            names of the output variables
	 * @return {@code this} Script object to allow chaining of methods
	 */
	public Script out(String... outputNames) {
		outputVariables.addAll(Arrays.asList(outputNames));
		return this;
	}

	/**
	 * Clear the inputs, outputs, and symbol table.
	 */
	public void clearIOS() {
		clearInputs();
		clearOutputs();
		clearSymbolTable();
	}

	/**
	 * Clear the inputs and outputs, but not the symbol table.
	 */
	public void clearIO() {
		clearInputs();
		clearOutputs();
	}

	/**
	 * Clear the script string, inputs, outputs, and symbol table.
	 */
	public void clearAll() {
		scriptString = null;
		clearIOS();
	}

	/**
	 * Clear the inputs.
	 */
	public void clearInputs() {
		inputs.clear();
		inputParameters.clear();
		inputVariables.clear();
		inputMetadata.clear();
	}

	/**
	 * Clear the outputs.
	 */
	public void clearOutputs() {
		outputVariables.clear();
	}

	/**
	 * Clear the symbol table.
	 */
	public void clearSymbolTable() {
		symbolTable.removeAll();
	}

	/**
	 * Obtain the results of the script execution.
	 *
	 * @return the results of the script execution.
	 */
	public MLResults results() {
		return results;
	}

	/**
	 * Obtain the results of the script execution.
	 *
	 * @return the results of the script execution.
	 */
	public MLResults getResults() {
		return results;
	}

	/**
	 * Set the results of the script execution.
	 *
	 * @param results
	 *            the results of the script execution.
	 */
	public void setResults(MLResults results) {
		this.results = results;
	}

	/**
	 * Obtain the script executor used by this Script.
	 *
	 * @return the ScriptExecutor used by this Script.
	 */
	public ScriptExecutor getScriptExecutor() {
		return scriptExecutor;
	}

	/**
	 * Set the ScriptExecutor used by this Script.
	 *
	 * @param scriptExecutor
	 *            the script executor
	 */
	public void setScriptExecutor(ScriptExecutor scriptExecutor) {
		this.scriptExecutor = scriptExecutor;
	}

	/**
	 * Is the script type DML?
	 *
	 * @return {@code true} if the script type is DML, {@code false} otherwise
	 */
	public boolean isDML() {
		return scriptType.isDML();
	}

	/**
	 * Is the script type PYDML?
	 *
	 * @return {@code true} if the script type is PYDML, {@code false} otherwise
	 */
	public boolean isPYDML() {
		return scriptType.isPYDML();
	}

	/**
	 * Generate the script execution string, which adds read/load/write/save
	 * statements to the beginning and end of the script to execute.
	 *
	 * @return the script execution string
	 */
	public String getScriptExecutionString() {
		StringBuilder sb = new StringBuilder();

		Set<String> ins = getInputVariables();
		for (String in : ins) {
			Object inValue = getInputs().get(in);
			sb.append(in);
			if (isDML()) {
				if (inValue instanceof String) {
					String quotedString = MLContextUtil.quotedString((String) inValue);
					sb.append(" = " + quotedString + ";\n");
				} else if (MLContextUtil.isBasicType(inValue)) {
					sb.append(" = read('', data_type='scalar');\n");
				} else if (MLContextUtil.doesSymbolTableContainFrameObject(symbolTable, in)) {
					sb.append(" = read('', data_type='frame');\n");
				} else {
					sb.append(" = read('');\n");
				}
			} else if (isPYDML()) {
				if (inValue instanceof String) {
					String quotedString = MLContextUtil.quotedString((String) inValue);
					sb.append(" = " + quotedString + "\n");
				} else if (MLContextUtil.isBasicType(inValue)) {
					sb.append(" = load('', data_type='scalar')\n");
				} else if (MLContextUtil.doesSymbolTableContainFrameObject(symbolTable, in)) {
					sb.append(" = load('', data_type='frame')\n");
				} else {
					sb.append(" = load('')\n");
				}
			}

		}

		sb.append(getScriptString());
		if (!getScriptString().endsWith("\n")) {
			sb.append("\n");
		}

		Set<String> outs = getOutputVariables();
		for (String out : outs) {
			if (isDML()) {
				sb.append("write(");
				sb.append(out);
				sb.append(", '');\n");
			} else if (isPYDML()) {
				sb.append("save(");
				sb.append(out);
				sb.append(", '')\n");
			}
		}

		return sb.toString();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		sb.append(MLContextUtil.displayInputs("Inputs", inputs, symbolTable));
		sb.append("\n");
		sb.append(MLContextUtil.displayOutputs("Outputs", outputVariables, symbolTable));
		return sb.toString();
	}

	/**
	 * Display information about the script as a String. This consists of the
	 * script type, inputs, outputs, input parameters, input variables, output
	 * variables, the symbol table, the script string, and the script execution
	 * string.
	 *
	 * @return information about this script as a String
	 */
	public String info() {
		StringBuilder sb = new StringBuilder();

		sb.append("Script Type: ");
		sb.append(scriptType);
		sb.append("\n\n");
		sb.append(MLContextUtil.displayInputs("Inputs", inputs, symbolTable));
		sb.append("\n");
		sb.append(MLContextUtil.displayOutputs("Outputs", outputVariables, symbolTable));
		sb.append("\n");
		sb.append(MLContextUtil.displayMap("Input Parameters", inputParameters));
		sb.append("\n");
		sb.append(MLContextUtil.displaySet("Input Variables", inputVariables));
		sb.append("\n");
		sb.append(MLContextUtil.displaySet("Output Variables", outputVariables));
		sb.append("\n");
		sb.append(MLContextUtil.displaySymbolTable("Symbol Table", symbolTable));
		sb.append("\nScript String:\n");
		sb.append(scriptString);
		sb.append("\nScript Execution String:\n");
		sb.append(getScriptExecutionString());
		sb.append("\n");

		return sb.toString();
	}

	/**
	 * Display the script inputs.
	 *
	 * @return the script inputs
	 */
	public String displayInputs() {
		return MLContextUtil.displayInputs("Inputs", inputs, symbolTable);
	}

	/**
	 * Display the script outputs.
	 *
	 * @return the script outputs as a String
	 */
	public String displayOutputs() {
		return MLContextUtil.displayOutputs("Outputs", outputVariables, symbolTable);
	}

	/**
	 * Display the script input parameters.
	 *
	 * @return the script input parameters as a String
	 */
	public String displayInputParameters() {
		return MLContextUtil.displayMap("Input Parameters", inputParameters);
	}

	/**
	 * Display the script input variables.
	 *
	 * @return the script input variables as a String
	 */
	public String displayInputVariables() {
		return MLContextUtil.displaySet("Input Variables", inputVariables);
	}

	/**
	 * Display the script output variables.
	 *
	 * @return the script output variables as a String
	 */
	public String displayOutputVariables() {
		return MLContextUtil.displaySet("Output Variables", outputVariables);
	}

	/**
	 * Display the script symbol table.
	 *
	 * @return the script symbol table as a String
	 */
	public String displaySymbolTable() {
		return MLContextUtil.displaySymbolTable("Symbol Table", symbolTable);
	}

	/**
	 * Obtain the script name.
	 *
	 * @return the script name
	 */
	public String getName() {
		return name;
	}

	/**
	 * Set the script name.
	 *
	 * @param name
	 *            the script name
	 * @return {@code this} Script object to allow chaining of methods
	 */
	public Script setName(String name) {
		this.name = name;
		return this;
	}

}

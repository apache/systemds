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

package org.apache.sysds.hops;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.ReaderGen;
import org.apache.sysds.runtime.meta.DataCharacteristics;

import java.util.HashMap;
import java.util.Map.Entry;


public class GenerateReaderOp extends Hop {
	private static final Log LOG = LogFactory.getLog(GenerateReaderOp.class.getName());
	private Types.OpOpGenerateReader _op;

	/**
	 * List of "named" input parameters. They are maintained as a hashmap:
	 * parameter names (String) are mapped as indices (Integer) into getInput()
	 * arraylist.
	 * <p>
	 * i.e., getInput().get(_paramIndexMap.get(parameterName)) refers to the Hop
	 * that is associated with parameterName.
	 */
	private HashMap<String, Integer> _paramIndexMap = new HashMap<>();

	private GenerateReaderOp() {
		//default constructor for clone
	}

	@Override
	public void checkArity() {

	}

	@Override
	public boolean allowsAllExecTypes() {
		return false;
	}

	@Override
	protected DataCharacteristics inferOutputCharacteristics(MemoTable memo) {
		return null;
	}

	@Override
	public Lop constructLops() {
		//return already created lops
		if( getLops() != null )
			return getLops();

		Types.ExecType et = Types.ExecType.CP;


		// construct lops for all input parameters
		HashMap<String, Lop> inputLops = new HashMap<>();
		for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
			inputLops.put(cur.getKey(), getInput().get(cur.getValue()).constructLops());
		}

		Lop l = new ReaderGen(getInput().get(0).constructLops(),_dataType, _valueType, et, inputLops);

		setLineNumbers(l);
		setPrivacy(l);
		setLops(l);

		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();

		return getLops();
	}

	@Override
	protected Types.ExecType optFindExecType(boolean transitive) {
		return null;
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += _op.toString();
		s += " "+getName();
		return s;
	}

	@Override
	public boolean isGPUEnabled() {
		return false;
	}

	@Override
	protected double computeOutputMemEstimate(long dim1, long dim2, long nnz) {
		return 0;
	}

	@Override
	protected double computeIntermediateMemEstimate(long dim1, long dim2, long nnz) {
		return 0;
	}

	@Override
	public void refreshSizeInformation() {

	}

	@Override
	public Object clone() throws CloneNotSupportedException {
		return null;
	}

	@Override
	public boolean compare(Hop that) {
		return false;
	}

	/**
	 * Generate Reader operation for Matrix
	 * This constructor supports expression in parameters
	 * @param l ?
	 * @param dt              data type
	 * @param dop             data operator type
	 * @param in              high-level operator
	 * @param inputParameters input parameters
	 */
	public GenerateReaderOp(String l, DataType dt, Types.OpOpGenerateReader dop, Hop in, HashMap<String, Hop> inputParameters) {
		_dataType = dt;
		_op = dop;
		_name = l;
		getInput().add(0, in);
		in.getParent().add(this);

		if(inputParameters != null) {
			int index = 1;
			for(Entry<String, Hop> e : inputParameters.entrySet()) {
				String s = e.getKey();
				Hop input = e.getValue();
				getInput().add(input);
				input.getParent().add(this);

				_paramIndexMap.put(s, index);
				index++;
			}
		}
	}

	public Types.OpOpGenerateReader getOp() {
		return _op;
	}
}

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

package org.apache.sysml.runtime.controlprogram.paramserv;

import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.ListObject;

public abstract class PSWorker {

	protected long _workerID = -1;
	protected String _updFunc;
	protected Statement.PSFrequency _freq;
	protected int _epochs;
	protected long _batchSize;
	protected MatrixObject _features;
	protected MatrixObject _labels;
	protected MatrixObject _valFeatures;
	protected MatrixObject _valLabels;
	protected ListObject _hyperParams;
	protected ExecutionContext _ec;
	protected ParamServer _ps;

	public PSWorker(long _workerID, String _updFunc, Statement.PSFrequency _freq, int _epochs, long _batchSize,
			ListObject _hyperParams, ExecutionContext _ec, ParamServer _ps) {
		this._workerID = _workerID;
		this._updFunc = _updFunc;
		this._freq = _freq;
		this._epochs = _epochs;
		this._batchSize = _batchSize;
		this._hyperParams = _hyperParams;
		this._ec = _ec;
		this._ps = _ps;
	}

	public long getWorkerID() {
		return _workerID;
	}

	public void setWorkerID(long workerID) {
		this._workerID = workerID;
	}

	public String getUpdFunc() {
		return _updFunc;
	}

	public void setUpdFunc(String updFunc) {
		this._updFunc = updFunc;
	}

	public Statement.PSFrequency getFreq() {
		return _freq;
	}

	public void setFreq(Statement.PSFrequency freq) {
		this._freq = freq;
	}

	public int getEpochs() {
		return _epochs;
	}

	public void setEpochs(int epochs) {
		this._epochs = epochs;
	}

	public long getBatchSize() {
		return _batchSize;
	}

	public void setBatchSize(long batchSize) {
		this._batchSize = batchSize;
	}

	public MatrixObject getFeatures() {
		return _features;
	}

	public void setFeatures(MatrixObject features) {
		this._features = features;
	}

	public MatrixObject getLabels() {
		return _labels;
	}

	public void setLabels(MatrixObject labels) {
		this._labels = labels;
	}

	public MatrixObject getValFeatures() {
		return _valFeatures;
	}

	public void setValFeatures(MatrixObject valFeatures) {
		this._valFeatures = valFeatures;
	}

	public MatrixObject getValLabels() {
		return _valLabels;
	}

	public void setValLabels(MatrixObject valLabels) {
		this._valLabels = valLabels;
	}

	public ListObject getHyperParams() {
		return _hyperParams;
	}

	public void setHyperParams(ListObject hyperParams) {
		this._hyperParams = hyperParams;
	}

	public ExecutionContext getEC() {
		return _ec;
	}

	public void setEC(ExecutionContext ec) {
		this._ec = ec;
	}

	public ParamServer getPS() {
		return _ps;
	}

	public void setPS(ParamServer ps) {
		this._ps = ps;
	}
}

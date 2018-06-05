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

import java.util.concurrent.Callable;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.ListObject;

public class LocalPSWorker extends PSWorker implements Callable<Void> {

	protected static final Log LOG = LogFactory.getLog(LocalPSWorker.class.getName());

	public LocalPSWorker(int workerID, String updFunc, Statement.PSFrequency freq, int epochs, long batchSize,
			ExecutionContext ec, ParamServer ps) {
		super(workerID, updFunc, freq, epochs, batchSize, ec, ps);
	}

	@Override
	public Void call() throws Exception {
		try {
			long dataSize = _features.getNumRows();
			int totalIter = (int) Math.ceil(dataSize / _batchSize);

			switch (_freq) {
				case BATCH:
					computeBatch(dataSize, totalIter);
					break;
				case EPOCH:
					computeEpoch(dataSize, totalIter);
					break;
			}

			if (LOG.isDebugEnabled()) {
				LOG.debug(String.format("Local worker_%d: Job finished.", _workerID));
			}
		} catch (Exception e) {
			throw new DMLRuntimeException(String.format("Local worker_%d failed", _workerID), e);
		}
		return null;
	}

	private void computeEpoch(long dataSize, int totalIter) {
		for (int i = 0; i < _epochs; i++) {
			// Pull the global parameters from ps
			ListObject globalParams = pullModel();

			for (int j = 0; j < totalIter; j++) {
				_ec.setVariable(Statement.PS_MODEL, globalParams);

				ListObject gradients = computeGradients(dataSize, totalIter, i, j);

				if (j == totalIter - 1) {
					// Push the gradients to ps
					pushGradients(gradients);
					ParamservUtils.cleanupListObject(_ec, globalParams);
				} else {
					// Update the local model with gradients
					globalParams = _ps.updateModel(gradients, globalParams);
					if (LOG.isDebugEnabled()) {
						LOG.debug(String.format("Local worker_%d: Local global parameter [size:%d kb] updated.",
								_workerID, globalParams.getDataSize()));
					}
				}
			}
			if (LOG.isDebugEnabled()) {
				LOG.debug(String.format("Local worker_%d: Finished %d epoch.", _workerID, i + 1));
			}
		}

	}

	private void computeBatch(long dataSize, int totalIter) {
		for (int i = 0; i < _epochs; i++) {
			for (int j = 0; j < totalIter; j++) {
				ListObject globalParams = pullModel();

				_ec.setVariable(Statement.PS_MODEL, globalParams);
				ListObject gradients = computeGradients(dataSize, totalIter, i, j);

				// Push the gradients to ps
				pushGradients(gradients);

				ParamservUtils.cleanupListObject(_ec, globalParams);
			}
			if (LOG.isDebugEnabled()) {
				LOG.debug(String.format("Local worker_%d: Finished %d epoch.", _workerID, i + 1));
			}
		}
	}

	private ListObject pullModel() {
		// Pull the global parameters from ps
		ListObject globalParams = (ListObject)_ps.pull(_workerID);
		if (LOG.isDebugEnabled()) {
			LOG.debug(String.format("Local worker_%d: Successfully pull the global parameters "
				+ "[size:%d kb] from ps.", _workerID, globalParams.getDataSize() / 1024));
		}
		return globalParams;
	}

	private void pushGradients(ListObject gradients) {
		// Push the gradients to ps
		_ps.push(_workerID, gradients);
		if (LOG.isDebugEnabled()) {
			LOG.debug(String.format("Local worker_%d: Successfully push the gradients "
					+ "[size:%d kb] to ps.", _workerID, gradients.getDataSize() / 1024));
		}
	}

	private ListObject computeGradients(long dataSize, int totalIter, int i, int j) {
		long begin = j * _batchSize + 1;
		long end = Math.min(begin + _batchSize, dataSize);

		// Get batch features and labels
		MatrixObject bFeatures = ParamservUtils.sliceMatrix(_features, begin, end);
		MatrixObject bLabels = ParamservUtils.sliceMatrix(_labels, begin, end);
		_ec.setVariable(Statement.PS_FEATURES, bFeatures);
		_ec.setVariable(Statement.PS_LABELS, bLabels);

		if (LOG.isDebugEnabled()) {
			LOG.debug(String.format("Local worker_%d: Got batch data [size:%d kb] of index from %d to %d. "
				+ "[Epoch:%d  Total epoch:%d  Iteration:%d  Total iteration:%d]", _workerID, bFeatures.getDataSize()
				/ 1024 + bLabels.getDataSize() / 1024, begin, end, i + 1, _epochs, j + 1, totalIter));
		}

		// Invoke the update function
		_inst.processInstruction(_ec);

		// Get the gradients
		ListObject gradients = (ListObject) _ec.getVariable(_output.getName());

		ParamservUtils.cleanupData(bFeatures);
		ParamservUtils.cleanupData(bLabels);
		return gradients;
	}
}

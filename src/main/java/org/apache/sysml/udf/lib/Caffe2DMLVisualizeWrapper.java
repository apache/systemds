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

package org.apache.sysml.udf.lib;

import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Scalar;
import org.apache.sysml.udf.Scalar.ScalarValueType;
import org.apache.sysml.utils.TensorboardLogger;

public class Caffe2DMLVisualizeWrapper extends PackageFunction 
{
	private static final long serialVersionUID = 1L;
	private Scalar _ret;

	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if (pos == 0)
			return _ret;

		throw new RuntimeException(
				"Invalid function output being requested");
	}

	@Override
	public void execute() {
		String layerName = ((Scalar) this.getFunctionInput(0)).getValue();
		String varType = ((Scalar) this.getFunctionInput(1)).getValue();
		String aggFn = ((Scalar) this.getFunctionInput(2)).getValue();
		double x = Double.parseDouble(((Scalar) this.getFunctionInput(3)).getValue());
		double y = Double.parseDouble(((Scalar) this.getFunctionInput(4)).getValue());
		String logDir = ((Scalar) this.getFunctionInput(5)).getValue();

		String key = null;
		if(aggFn.equals("training_loss") || aggFn.equals("validation_loss") ||
				aggFn.equals("training_accuracy") || aggFn.equals("validation_accuracy"))
			key = aggFn;
		else
			key = aggFn + "_" + varType + "_" + layerName;
		TensorboardLogger.writeScalar(logDir, key, (long)x, (float)y);
		_ret = new Scalar(ScalarValueType.Double, String.valueOf(1));
	}

}

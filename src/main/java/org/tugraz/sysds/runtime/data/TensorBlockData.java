/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.data;

import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.runtime.instructions.cp.Data;

/**
 * Temporary implementation of a `Data`-TensorBlock so the Tensor can be tested without implementing CacheableData for it.
 */
public class TensorBlockData extends Data {
	private static final long serialVersionUID = -3858118069498977569L;

	private TensorBlock _tb;

	public TensorBlockData(Types.ValueType vt) {
		super(Types.DataType.TENSOR, vt);
		// TODO handle different parameters
		_tb = new TensorBlock();
	}

	public TensorBlock getTensorBlock() {
		return _tb;
	}

	public void setTensorBlock(TensorBlock tb) {
		_tb = tb;
	}

	@Override
	public String getDebugName() {
		return "TensorBlockData";
	}
}

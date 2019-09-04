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
 *
 */

package org.tugraz.sysds.runtime.io;

import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;

public class TensorReaderFactory {

	public static TensorReader createTensorReader(InputInfo iinfo) {
		TensorReader reader;

		if (iinfo == InputInfo.TextCellInputInfo) {
			reader = new TensorReaderTextCell();
		}
		else if (iinfo == InputInfo.BinaryBlockInputInfo) {
			reader = new TensorReaderBinaryBlock();
		}
		else {
			throw new DMLRuntimeException("Failed to create tensor reader for unknown output info: "
					+ InputInfo.inputInfoToString(iinfo));
		}
		return reader;
	}
}

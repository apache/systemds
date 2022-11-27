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

package org.apache.sysds.runtime.frame.data;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.columns.Array;

public interface FrameUtil {
	@SuppressWarnings({"rawtypes"})
	public static Array[] add(Array[] ar, Array e) {
		if(ar == null)
			return new Array[] {e};
		Array[] ret = new Array[ar.length + 1];
		System.arraycopy(ar, 0, ret, 0, ar.length);
		ret[ar.length] = e;
		return ret;
	}

	public static ValueType isType(String val) {
		val = val.trim().toLowerCase().replaceAll("\"", "");
		if(val.matches("(true|false|t|f|0|1)"))
			return ValueType.BOOLEAN;
		else if(val.matches("[-+]?\\d+")) {
			long maxValue = Long.parseLong(val);
			if((maxValue >= Integer.MIN_VALUE) && (maxValue <= Integer.MAX_VALUE))
				return ValueType.INT32;
			else
				return ValueType.INT64;
		}
		else if(val.matches("[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?")) {
			double maxValue = Double.parseDouble(val);
			if((maxValue >= (-Float.MAX_VALUE)) && (maxValue <= Float.MAX_VALUE))
				return ValueType.FP32;
			else
				return ValueType.FP64;
		}
		else if(val.equals("infinity") || val.equals("-infinity") || val.equals("nan"))
			return ValueType.FP64;
		else
			return ValueType.STRING;
	}
}
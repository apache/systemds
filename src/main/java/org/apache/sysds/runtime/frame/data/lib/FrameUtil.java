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

package org.apache.sysds.runtime.frame.data.lib;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.iterators.IteratorFactory;
import org.apache.sysds.runtime.util.UtilFunctions;

public interface FrameUtil {
	public static final Log LOG = LogFactory.getLog(FrameUtil.class.getName());

	public static Array<?>[] add(Array<?>[] ar, Array<?> e) {
		if(ar == null)
			return new Array[] {e};
		Array<?>[] ret = new Array[ar.length + 1];
		System.arraycopy(ar, 0, ret, 0, ar.length);
		ret[ar.length] = e;
		return ret;
	}

	public static ValueType isType(String val) {
		if (val == null)
			return ValueType.UNKNOWN;
		val = val.trim().toLowerCase().replaceAll("\"", "");
		if(val.matches("(true|false|t|f|0|1|0\\.0+|1\\.0+)"))
			return ValueType.BOOLEAN;
		else if(val.matches("[-+]?\\d+\\.0+")) { // 11.00000000 1313241.0
			long maxValue = Long.parseLong(val.split("\\.")[0]);
			if((maxValue >= Integer.MIN_VALUE) && (maxValue <= Integer.MAX_VALUE))
				return ValueType.INT32;
			else
				return ValueType.INT64;
		}
		else if(val.matches("[-+]?\\d+")) { // 1 3 6 192 14152131
			long maxValue = Long.parseLong(val);
			if((maxValue >= Integer.MIN_VALUE) && (maxValue <= Integer.MAX_VALUE))
				return ValueType.INT32;
			else
				return ValueType.INT64;
		}
		else if(val.matches("[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?")) {
			// parse float, and make back to string if equivalent use float.
			float f = Float.parseFloat(val);
			if(val.equals(Float.toString(f)))
				return ValueType.FP32;
			else
				return ValueType.FP64;
		}
		else if(val.equals("infinity") || val.equals("-infinity") || val.equals("nan"))
			return ValueType.FP64;
		else
			return ValueType.STRING;
	}

	public static ValueType isType(double val){
		if(val == 1.0d || val == 0.0d)
			return ValueType.BOOLEAN;
		else if((long)(val) == val){
			if((int)val == val)
				return ValueType.INT32;
			else
				return ValueType.INT64;
		}
		else if((double)((float) val) == val )
			// Detecting FP32 could use some extra work.
			return ValueType.FP32; 
		
		
		return ValueType.FP64;

	}


	public static FrameBlock mergeSchema(FrameBlock temp1, FrameBlock temp2) {
		String[] rowTemp1 = IteratorFactory.getStringRowIterator(temp1).next();
		String[] rowTemp2 = IteratorFactory.getStringRowIterator(temp2).next();

		if(rowTemp1.length != rowTemp2.length)
			throw new DMLRuntimeException("Schema dimension " + "mismatch: " + rowTemp1.length + " vs " + rowTemp2.length);

		for(int i = 0; i < rowTemp1.length; i++) {
			// modify schema1 if necessary (different schema2)
			if(!rowTemp1[i].equals(rowTemp2[i])) {
				if(rowTemp1[i].equals("STRING") || rowTemp2[i].equals("STRING"))
					rowTemp1[i] = "STRING";
				else if(rowTemp1[i].equals("FP64") || rowTemp2[i].equals("FP64"))
					rowTemp1[i] = "FP64";
				else if(rowTemp1[i].equals("FP32") &&
					new ArrayList<>(Arrays.asList("INT64", "INT32", "CHARACTER")).contains(rowTemp2[i]))
					rowTemp1[i] = "FP32";
				else if(rowTemp1[i].equals("INT64") &&
					new ArrayList<>(Arrays.asList("INT32", "CHARACTER")).contains(rowTemp2[i]))
					rowTemp1[i] = "INT64";
				else if(rowTemp1[i].equals("INT32") || rowTemp2[i].equals("CHARACTER"))
					rowTemp1[i] = "INT32";
			}
		}

		// create output block one row representing the schema as strings
		FrameBlock mergedFrame = new FrameBlock(UtilFunctions.nCopies(temp1.getNumColumns(), ValueType.STRING));
		mergedFrame.appendRow(rowTemp1);
		return mergedFrame;
	}

}

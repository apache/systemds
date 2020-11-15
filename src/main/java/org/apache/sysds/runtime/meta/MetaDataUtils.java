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

package org.apache.sysds.runtime.meta;

import org.apache.sysds.runtime.DMLRuntimeException;

public class MetaDataUtils {

	public static void updateAppendDataCharacteristics(DataCharacteristics mc1,
		DataCharacteristics mc2, DataCharacteristics mcOut, boolean cbind)
	{
		if(!mcOut.dimsKnown()) { 
			if( !mc1.dimsKnown() || !mc2.dimsKnown() )
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from inputs.");
			
			if( cbind )
				mcOut.set(mc1.getRows(), mc1.getCols()+mc2.getCols(), mc1.getBlocksize());
			else //rbind
				mcOut.set(mc1.getRows()+mc2.getRows(), mc1.getCols(), mc1.getBlocksize());
		}
	}
}

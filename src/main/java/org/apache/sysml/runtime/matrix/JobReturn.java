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


package org.apache.sysml.runtime.matrix;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.OutputInfo;


public class JobReturn 
{
	public boolean successful;
	// public MatrixCharacteristics[] stats;
	// public MetaData[] otherMetadata=null;
	public MetaData[] metadata = null;

	public JobReturn() {
		successful = false;
		metadata = null;
	}

	public JobReturn(MatrixCharacteristics[] sts, boolean success) {
		successful = success;
		metadata = new MetaData[sts.length];
		for (int i = 0; i < sts.length; i++) {
			metadata[i] = new MetaData(sts[i]);
		}
	}

	public JobReturn(MatrixCharacteristics[] sts, OutputInfo[] infos, boolean success) {
		successful = success;
		metadata = new MetaDataFormat[sts.length];
		for (int i = 0; i < sts.length; i++) {
			metadata[i] = new MetaDataFormat(sts[i], infos[i], OutputInfo.getMatchingInputInfo(infos[i]));
		}
	}

	public JobReturn(MatrixCharacteristics sts, OutputInfo info, boolean success) {
		successful = success;
		metadata = new MetaDataFormat[1];
		metadata[0] = new MetaDataFormat(sts, info, OutputInfo.getMatchingInputInfo(info));
	}
	
	public JobReturn(MatrixCharacteristics mc, long[] items, int partition0, long number0s, boolean success) {
		successful = success;
		metadata = new MetaDataNumItemsByEachReducer[1];
		metadata[0] = new MetaDataNumItemsByEachReducer(mc, items, partition0, number0s);
	}

	public boolean checkReturnStatus() {
		if( !successful )
			throw new DMLRuntimeException("Error in executing the DML program.");
		return successful;
	}

	public MetaData[] getMetaData() {
		return metadata;
	}

	public MetaData getMetaData(int i) {
		return metadata[i];
	}
}

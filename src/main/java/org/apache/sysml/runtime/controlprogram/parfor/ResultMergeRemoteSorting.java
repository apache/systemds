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

package org.apache.sysml.runtime.controlprogram.parfor;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

public class ResultMergeRemoteSorting extends WritableComparator
{
	
	
	protected ResultMergeRemoteSorting()
	{
		super(ResultMergeTaggedMatrixIndexes.class, true);
	}
	
	@SuppressWarnings("rawtypes")
	@Override
    public int compare(WritableComparable k1, WritableComparable k2) 
	{
		ResultMergeTaggedMatrixIndexes key1 = (ResultMergeTaggedMatrixIndexes)k1;
		ResultMergeTaggedMatrixIndexes key2 = (ResultMergeTaggedMatrixIndexes)k2;

		int ret = key1.getIndexes().compareTo(key2.getIndexes());
		if( ret == 0 ) //same indexes, secondary sort
		{
			ret = ((key1.getTag() == key2.getTag()) ? 0 : 
				   (key1.getTag() < key2.getTag())? -1 : 1);
		}	
			
		return ret; 
		
    }
}

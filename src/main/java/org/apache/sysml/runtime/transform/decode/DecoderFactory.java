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

package org.apache.sysml.runtime.transform.decode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.transform.meta.TfMetaUtils;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONObject;


public class DecoderFactory 
{
	/**
	 * 
	 * @param spec
	 * @param schema
	 * @param meta
	 * @return
	 * @throws DMLRuntimeException
	 */
	@SuppressWarnings("unchecked")
	public static Decoder createDecoder(String spec, List<ValueType> schema, FrameBlock meta) 
		throws DMLRuntimeException 
	{	
		Decoder decoder = null;
		
		try 
		{
			//parse transform specification
			JSONObject jSpec = new JSONObject(spec);
			List<Decoder> ldecoders = new ArrayList<Decoder>();
		
			//create default schema if unspecified
			if( schema == null ) {
				schema = Collections.nCopies(meta.getNumColumns(), ValueType.STRING);
			}
			
			//create decoders 'recode', 'dummy' and 'pass-through'
			List<Integer> rcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, TfUtils.TXMETHOD_RECODE)));
			List<Integer> dcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, TfUtils.TXMETHOD_DUMMYCODE))); 
			rcIDs = new ArrayList<Integer>(CollectionUtils.union(rcIDs, dcIDs));
			List<Integer> ptIDs = new ArrayList<Integer>(CollectionUtils
					.subtract(UtilFunctions.getSequenceList(1, schema.size(), 1), rcIDs)); 
			
			if( !dcIDs.isEmpty() ) {
				ldecoders.add(new DecoderDummycode(schema, 
						ArrayUtils.toPrimitive(dcIDs.toArray(new Integer[0]))));
			}
			if( !rcIDs.isEmpty() ) {
				ldecoders.add(new DecoderRecode(schema, !dcIDs.isEmpty(),
						ArrayUtils.toPrimitive(rcIDs.toArray(new Integer[0]))));
			}
			if( !ptIDs.isEmpty() ) {
				ldecoders.add(new DecoderPassThrough(schema, 
						ArrayUtils.toPrimitive(ptIDs.toArray(new Integer[0])),
						ArrayUtils.toPrimitive(dcIDs.toArray(new Integer[0]))));	
			}
			
			//create composite decoder of all created decoders
			//and initialize with given meta data (recode, dummy, bin)
			decoder = new DecoderComposite(schema, ldecoders);
			if( meta != null )
				decoder.initMetaData(meta);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		return decoder;
	}
}

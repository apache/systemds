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

package org.apache.sysml.runtime.transform.encode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.transform.BinAgent;
import org.apache.sysml.runtime.transform.DummycodeAgent;
import org.apache.sysml.runtime.transform.MVImputeAgent;
import org.apache.sysml.runtime.transform.OmitAgent;
import org.apache.sysml.runtime.transform.RecodeAgent;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.transform.meta.TfMetaUtils;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONObject;

public class EncoderFactory 
{
	/**
	 * 
	 * @param spec
	 * @param clen
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static Encoder createEncoder(String spec, int clen, FrameBlock meta) throws DMLRuntimeException {
		return createEncoder(spec, Collections.nCopies(clen, ValueType.STRING), meta);
	}
	
	/**
	 * 
	 * @param spec
	 * @param schema
	 * @param clen
	 * @param meta
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Encoder createEncoder(String spec, List<ValueType> schema, int clen, FrameBlock meta) throws DMLRuntimeException {
		List<ValueType> lschema = (schema==null) ? Collections.nCopies(clen, ValueType.STRING) : schema;
		return createEncoder(spec, lschema, meta);
	}
	
	
	/**
	 * 
	 * @param spec
	 * @param schema
	 * @return
	 * @throws DMLRuntimeException
	 */
	@SuppressWarnings("unchecked")
	public static Encoder createEncoder(String spec, List<ValueType> schema, FrameBlock meta) 
		throws DMLRuntimeException 
	{	
		Encoder encoder = null;
		int clen = schema.size();
		
		try {
			//parse transform specification
			JSONObject jSpec = new JSONObject(spec);
			List<Encoder> lencoders = new ArrayList<Encoder>();
		
			//prepare basic id lists (recode, dummycode, pass-through)
			//note: any dummycode column requires recode as preparation
			List<Integer> rcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, TfUtils.TXMETHOD_RECODE)));
			List<Integer> dcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, TfUtils.TXMETHOD_DUMMYCODE))); 
			rcIDs = new ArrayList<Integer>(CollectionUtils.union(rcIDs, dcIDs));
			List<Integer> binIDs = TfMetaUtils.parseBinningColIDs(jSpec); 
			List<Integer> ptIDs = new ArrayList<Integer>(CollectionUtils.subtract(
					CollectionUtils.subtract(UtilFunctions.getSequenceList(1, clen, 1), rcIDs), binIDs)); 
			List<Integer> oIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, TfUtils.TXMETHOD_OMIT))); 
			List<Integer> mvIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonObjectIDList(jSpec, TfUtils.TXMETHOD_IMPUTE))); 
			
			//create individual encoders
			if( !rcIDs.isEmpty() ) {
				RecodeAgent ra = new RecodeAgent(jSpec, clen);
				ra.setColList(ArrayUtils.toPrimitive(rcIDs.toArray(new Integer[0])));
				lencoders.add(ra);	
			}
			if( !ptIDs.isEmpty() )
				lencoders.add(new EncoderPassThrough(
						ArrayUtils.toPrimitive(ptIDs.toArray(new Integer[0])), clen));	
			if( !dcIDs.isEmpty() )
				lencoders.add(new DummycodeAgent(jSpec, schema.size()));
			if( !binIDs.isEmpty() )
				lencoders.add(new BinAgent(jSpec, schema.size(), true));
			if( !oIDs.isEmpty() )
				lencoders.add(new OmitAgent(jSpec, schema.size()));
			if( !mvIDs.isEmpty() ) {
				MVImputeAgent ma = new MVImputeAgent(jSpec, schema.size());
				ma.initRecodeIDList(rcIDs);
				lencoders.add(ma);
			}
			
			//create composite decoder of all created encoders
			//and initialize meta data (recode, dummy, bin, mv)
			encoder = new EncoderComposite(lencoders);
			if( meta != null )
				encoder.initMetaData(meta);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		return encoder;
	}
}

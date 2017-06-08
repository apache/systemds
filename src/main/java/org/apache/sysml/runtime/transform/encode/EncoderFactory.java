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
import java.util.List;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.transform.meta.TfMetaUtils;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONObject;

public class EncoderFactory 
{
	public static Encoder createEncoder(String spec, String[] colnames, int clen, FrameBlock meta) throws DMLRuntimeException {
		return createEncoder(spec, colnames, UtilFunctions.nCopies(clen, ValueType.STRING), meta);
	}

	public static Encoder createEncoder(String spec, String[] colnames, ValueType[] schema, int clen, FrameBlock meta) throws DMLRuntimeException {
		ValueType[] lschema = (schema==null) ? UtilFunctions.nCopies(clen, ValueType.STRING) : schema;
		return createEncoder(spec, colnames, lschema, meta);
	}

	@SuppressWarnings("unchecked")
	public static Encoder createEncoder(String spec,  String[] colnames, ValueType[] schema, FrameBlock meta) 
		throws DMLRuntimeException 
	{	
		Encoder encoder = null;
		int clen = schema.length;
		
		try {
			//parse transform specification
			JSONObject jSpec = new JSONObject(spec);
			List<Encoder> lencoders = new ArrayList<Encoder>();
		
			//prepare basic id lists (recode, dummycode, pass-through)
			//note: any dummycode column requires recode as preparation
			List<Integer> rcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TXMETHOD_RECODE)));
			List<Integer> dcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TXMETHOD_DUMMYCODE))); 
			rcIDs = new ArrayList<Integer>(CollectionUtils.union(rcIDs, dcIDs));
			List<Integer> binIDs = TfMetaUtils.parseBinningColIDs(jSpec, colnames); 
			List<Integer> ptIDs = new ArrayList<Integer>(CollectionUtils.subtract(
					CollectionUtils.subtract(UtilFunctions.getSequenceList(1, clen, 1), rcIDs), binIDs)); 
			List<Integer> oIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TXMETHOD_OMIT))); 
			List<Integer> mvIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonObjectIDList(jSpec, colnames, TfUtils.TXMETHOD_IMPUTE))); 
			
			//create individual encoders
			if( !rcIDs.isEmpty() ) {
				EncoderRecode ra = new EncoderRecode(jSpec, colnames, clen);
				ra.setColList(ArrayUtils.toPrimitive(rcIDs.toArray(new Integer[0])));
				lencoders.add(ra);	
			}
			if( !ptIDs.isEmpty() )
				lencoders.add(new EncoderPassThrough(
						ArrayUtils.toPrimitive(ptIDs.toArray(new Integer[0])), clen));	
			if( !dcIDs.isEmpty() )
				lencoders.add(new EncoderDummycode(jSpec, colnames, schema.length));
			if( !binIDs.isEmpty() )
				lencoders.add(new EncoderBin(jSpec, colnames, schema.length, true));
			if( !oIDs.isEmpty() )
				lencoders.add(new EncoderOmit(jSpec, colnames, schema.length));
			if( !mvIDs.isEmpty() ) {
				EncoderMVImpute ma = new EncoderMVImpute(jSpec, colnames, schema.length);
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

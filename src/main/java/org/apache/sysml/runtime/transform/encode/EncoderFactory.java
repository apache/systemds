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
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.transform.RecodeAgent;
import org.apache.sysml.runtime.transform.TfUtils;
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
	public static Encoder createEncoder(String spec, List<ValueType> schema, FrameBlock meta) 
		throws DMLRuntimeException 
	{	
		Encoder encoder = null;
		
		try {
			//parse transform specification
			JSONObject jSpec = new JSONObject(spec);
			List<Encoder> lencoders = new ArrayList<Encoder>();
		
			//create encoders 'recode' and 'pass-through'
			if ( jSpec.containsKey(TfUtils.TXMETHOD_RECODE))  {
				RecodeAgent ra = new RecodeAgent(jSpec);
				if( meta != null )
					ra.initRecodeMaps(meta);
				lencoders.add(ra);
			
				//pass-through decode (non-recode columns)
				int[] rcCols = ra.getColList();
				if( schema.size() > rcCols.length ) {
					int[] ptCols = new int[schema.size()-rcCols.length]; 
					HashSet<Integer> probe = new HashSet<Integer>();
					for( int j=0; j<rcCols.length; j++ )
						probe.add(rcCols[j]-1);
					for( int j=0, pos=0; j<schema.size(); j++ )
						if( !probe.contains(j) )
							ptCols[pos++] = j;
					lencoders.add(new EncoderPassThrough(ptCols));	
				}
			}
			//create full 'pass-through' encoder if necessary
			else {
				int[] ptCols = new int[schema.size()];
				for( int j=0; j<ptCols.length; j++ )
					ptCols[j] = j;
				lencoders.add(new EncoderPassThrough(ptCols));
			}	
			
			//create composite decoder of all created decoders
			encoder = new EncoderComposite(lencoders);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		return encoder;
	}
}

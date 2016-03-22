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
import java.util.Collections;
import java.util.List;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.transform.TransformationAgent;
import org.apache.sysml.runtime.transform.TransformationAgent.TX_METHOD;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONArray;
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
			
			//create decoder 'recode'
			if ( jSpec.containsKey(TX_METHOD.RECODE.toString()))  {
				JSONArray attrs = null;
				if( jSpec.get(TX_METHOD.RECODE.toString()) instanceof JSONObject ) {
					JSONObject obj = (JSONObject) jSpec.get(TX_METHOD.RECODE.toString());
					attrs = (JSONArray) obj.get(TransformationAgent.JSON_ATTRS);
				}
				else
					attrs = (JSONArray)jSpec.get(TX_METHOD.RECODE.toString());
				
				int[] rcCols = new int[attrs.size()];
				for(int j=0; j<rcCols.length; j++) 
					rcCols[j] = UtilFunctions.toInt(attrs.get(j))-1;
				
				ldecoders.add(new DecoderRecode(schema, meta, rcCols));
			}
			
			//create composite decoder of all created decoders
			decoder = new DecoderComposite(schema, ldecoders);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		return decoder;
	}
}

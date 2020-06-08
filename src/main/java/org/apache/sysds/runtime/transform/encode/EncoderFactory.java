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

package org.apache.sysds.runtime.transform.encode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.wink.json4j.JSONObject;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import static org.apache.sysds.runtime.util.CollectionUtils.except;
import static org.apache.sysds.runtime.util.CollectionUtils.unionDistinct;


public class EncoderFactory 
{
	public static Encoder createEncoder(String spec, String[] colnames, int clen, FrameBlock meta) {
		return createEncoder(spec, colnames, UtilFunctions.nCopies(clen, ValueType.STRING), meta);
	}
	
	public static Encoder createEncoder(String spec, String[] colnames, int clen, FrameBlock meta, int minCol,
		int maxCol) {
		return createEncoder(spec, colnames, UtilFunctions.nCopies(clen, ValueType.STRING), meta, minCol, maxCol);
	}

	public static Encoder createEncoder(String spec, String[] colnames, ValueType[] schema, int clen, FrameBlock meta) {
		ValueType[] lschema = (schema==null) ? UtilFunctions.nCopies(clen, ValueType.STRING) : schema;
		return createEncoder(spec, colnames, lschema, meta);
	}
	
	public static Encoder createEncoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta) {
		return createEncoder(spec, colnames, schema, meta, -1, -1);
	}
	
	public static Encoder createEncoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta, int minCol,
		int maxCol) {
		Encoder encoder = null;
		int clen = schema.length;
		
		try {
			//parse transform specification
			JSONObject jSpec = new JSONObject(spec);
			List<Encoder> lencoders = new ArrayList<>();
			
			//prepare basic id lists (recode, feature hash, dummycode, pass-through)
			List<Integer> rcIDs = Arrays.asList(ArrayUtils.toObject(
				TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.RECODE.toString(), minCol, maxCol)));
			List<Integer>haIDs = Arrays.asList(ArrayUtils.toObject(
				TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.HASH.toString(), minCol, maxCol)));
			List<Integer> dcIDs = Arrays.asList(ArrayUtils.toObject(
				TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.DUMMYCODE.toString(), minCol, maxCol)));
			List<Integer> binIDs = TfMetaUtils.parseBinningColIDs(jSpec, colnames, minCol, maxCol);
			//note: any dummycode column requires recode as preparation, unless it follows binning
			rcIDs = except(unionDistinct(rcIDs, except(dcIDs, binIDs)), haIDs);
			List<Integer> ptIDs = except(except(UtilFunctions.getSeqList(1, clen, 1),
				unionDistinct(rcIDs,haIDs)), binIDs);
			List<Integer> oIDs = Arrays.asList(ArrayUtils.toObject(
				TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.OMIT.toString(), minCol, maxCol)));
			List<Integer> mvIDs = Arrays.asList(ArrayUtils.toObject(
				TfMetaUtils.parseJsonObjectIDList(jSpec, colnames, TfMethod.IMPUTE.toString(), minCol, maxCol)));
			
			//create individual encoders
			if( !rcIDs.isEmpty() ) {
				EncoderRecode ra = new EncoderRecode(jSpec, colnames, clen, minCol, maxCol);
				ra.setColList(ArrayUtils.toPrimitive(rcIDs.toArray(new Integer[0])));
				lencoders.add(ra);
			}
			if( !haIDs.isEmpty() ) {
				EncoderFeatureHash ha = new EncoderFeatureHash(jSpec, colnames, clen, minCol, maxCol);
				ha.setColList(ArrayUtils.toPrimitive(haIDs.toArray(new Integer[0])));
				lencoders.add(ha);
			}
			if( !ptIDs.isEmpty() )
				lencoders.add(new EncoderPassThrough(
						ArrayUtils.toPrimitive(ptIDs.toArray(new Integer[0])), clen));
			if( !binIDs.isEmpty() )
				lencoders.add(new EncoderBin(jSpec, colnames, schema.length, minCol, maxCol));
			if( !dcIDs.isEmpty() )
				lencoders.add(new EncoderDummycode(jSpec, colnames, schema.length, minCol, maxCol));
			if( !oIDs.isEmpty() )
				lencoders.add(new EncoderOmit(jSpec, colnames, schema.length, minCol, maxCol));
			if( !mvIDs.isEmpty() ) {
				EncoderMVImpute ma = new EncoderMVImpute(jSpec, colnames, schema.length, minCol, maxCol);
				ma.initRecodeIDList(rcIDs);
				lencoders.add(ma);
			}
			
			//create composite decoder of all created encoders
			encoder = new EncoderComposite(lencoders);
			
			//initialize meta data w/ robustness for superset of cols
			if( meta != null ) {
				String[] colnames2 = meta.getColumnNames();
				if( !TfMetaUtils.isIDSpec(jSpec) && colnames!=null && colnames2!=null
					&& !ArrayUtils.isEquals(colnames, colnames2) )
				{
					HashMap<String, Integer> colPos = getColumnPositions(colnames2);
					//create temporary meta frame block w/ shallow column copy
					FrameBlock meta2 = new FrameBlock(meta.getSchema(), colnames2);
					meta2.setNumRows(meta.getNumRows());
					for( int i=0; i<colnames.length; i++ ) {
						if( !colPos.containsKey(colnames[i]) ) {
							throw new DMLRuntimeException("Column name not found in meta data: "
								+ colnames[i]+" (meta: "+Arrays.toString(colnames2)+")");
						}
						int pos = colPos.get(colnames[i]);
						meta2.setColumn(i, meta.getColumn(pos));
						meta2.setColumnMetadata(i, meta.getColumnMetadata(pos));
					}
					meta = meta2;
				}
				encoder.initMetaData(meta);
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		return encoder;
	}
	
	private static HashMap<String, Integer> getColumnPositions(String[] colnames) {
		HashMap<String, Integer> ret = new HashMap<>();
		for(int i=0; i<colnames.length; i++)
			ret.put(colnames[i], i);
		return ret;
	}
}

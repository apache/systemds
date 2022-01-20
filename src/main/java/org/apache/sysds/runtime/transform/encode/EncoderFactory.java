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

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.encode.ColumnEncoder.EncoderType;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.Statistics;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import static org.apache.sysds.runtime.util.CollectionUtils.except;
import static org.apache.sysds.runtime.util.CollectionUtils.unionDistinct;

public class EncoderFactory {

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, int clen, FrameBlock meta) {
		return createEncoder(spec, colnames, UtilFunctions.nCopies(clen, ValueType.STRING), meta);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, int clen, FrameBlock meta,
		int minCol, int maxCol) {
		return createEncoder(spec, colnames, UtilFunctions.nCopies(clen, ValueType.STRING), meta, minCol, maxCol);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, ValueType[] schema, int clen,
		FrameBlock meta) {
		ValueType[] lschema = (schema == null) ? UtilFunctions.nCopies(clen, ValueType.STRING) : schema;
		return createEncoder(spec, colnames, lschema, meta);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, ValueType[] schema,
		FrameBlock meta) {
		return createEncoder(spec, colnames, schema, meta, -1, -1);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta,
		int minCol, int maxCol) {
		MultiColumnEncoder encoder;
		int clen = schema.length;

		try {
			// parse transform specification
			JSONObject jSpec = new JSONObject(spec);
			List<ColumnEncoderComposite> lencoders = new ArrayList<>();
			HashMap<Integer, List<ColumnEncoder>> colEncoders = new HashMap<>();
			boolean ids = jSpec.containsKey("ids") && jSpec.getBoolean("ids");

			// prepare basic id lists (recode, feature hash, dummycode, pass-through)
			List<Integer> rcIDs = Arrays.asList(ArrayUtils
				.toObject(TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.RECODE.toString(), minCol, maxCol)));
			List<Integer> haIDs = Arrays.asList(ArrayUtils
				.toObject(TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.HASH.toString(), minCol, maxCol)));
			List<Integer> dcIDs = Arrays.asList(ArrayUtils
				.toObject(TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.DUMMYCODE.toString(), minCol, maxCol)));
			List<Integer> binIDs = TfMetaUtils.parseBinningColIDs(jSpec, colnames, minCol, maxCol);
			// note: any dummycode column requires recode as preparation, unless it follows binning
			rcIDs = except(unionDistinct(rcIDs, except(dcIDs, binIDs)), haIDs);
			List<Integer> ptIDs = except(except(UtilFunctions.getSeqList(1, clen, 1), unionDistinct(rcIDs, haIDs)),
				binIDs);
			List<Integer> oIDs = Arrays.asList(ArrayUtils
				.toObject(TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.OMIT.toString(), minCol, maxCol)));
			List<Integer> mvIDs = Arrays.asList(ArrayUtils.toObject(
				TfMetaUtils.parseJsonObjectIDList(jSpec, colnames, TfMethod.IMPUTE.toString(), minCol, maxCol)));

			// create individual encoders
			if(!rcIDs.isEmpty()) {
				for(Integer id : rcIDs) {
					ColumnEncoderRecode ra = new ColumnEncoderRecode(id);
					addEncoderToMap(ra, colEncoders);
				}
			}
			if(!haIDs.isEmpty()) {
				for(Integer id : haIDs) {
					ColumnEncoderFeatureHash ha = new ColumnEncoderFeatureHash(id, TfMetaUtils.getK(jSpec));
					addEncoderToMap(ha, colEncoders);
				}
			}
			if(!ptIDs.isEmpty())
				for(Integer id : ptIDs) {
					ColumnEncoderPassThrough pt = new ColumnEncoderPassThrough(id);
					addEncoderToMap(pt, colEncoders);
				}
			if(!binIDs.isEmpty())
				for(Object o : (JSONArray) jSpec.get(TfMethod.BIN.toString())) {
					JSONObject colspec = (JSONObject) o;
					int numBins = colspec.containsKey("numbins") ? colspec.getInt("numbins") : 1;
					int id = TfMetaUtils.parseJsonObjectID(colspec, colnames, minCol, maxCol, ids);
					if(id <= 0)
						continue;
					String method = colspec.get("method").toString().toUpperCase();
					ColumnEncoderBin.BinMethod binMethod;
					if ("EQUI-WIDTH".equals(method)) {
						binMethod = ColumnEncoderBin.BinMethod.EQUI_WIDTH;
					}
					else if ("EQUI-HEIGHT".equals(method)) {
						binMethod = ColumnEncoderBin.BinMethod.EQUI_HEIGHT;
					}
					else {
						throw new DMLRuntimeException("Unsupported binning method: " + method);
					}
					ColumnEncoderBin bin = new ColumnEncoderBin(id, numBins, binMethod);
					addEncoderToMap(bin, colEncoders);
				}
			if(!dcIDs.isEmpty())
				for(Integer id : dcIDs) {
					ColumnEncoderDummycode dc = new ColumnEncoderDummycode(id);
					addEncoderToMap(dc, colEncoders);
				}
			// create composite decoder of all created encoders
			for(Entry<Integer, List<ColumnEncoder>> listEntry : colEncoders.entrySet()) {
				if(DMLScript.STATISTICS)
					Statistics.incTransformEncoderCount(listEntry.getValue().size());
				lencoders.add(new ColumnEncoderComposite(listEntry.getValue()));
			}
			encoder = new MultiColumnEncoder(lencoders);
			if(!oIDs.isEmpty()) {
				encoder.addReplaceLegacyEncoder(new EncoderOmit(jSpec, colnames, schema.length, minCol, maxCol));
				if(DMLScript.STATISTICS)
					Statistics.incTransformEncoderCount(1);
			}
			if(!mvIDs.isEmpty()) {
				EncoderMVImpute ma = new EncoderMVImpute(jSpec, colnames, schema.length, minCol, maxCol);
				ma.initRecodeIDList(rcIDs);
				encoder.addReplaceLegacyEncoder(ma);
				if(DMLScript.STATISTICS)
					Statistics.incTransformEncoderCount(1);
			}

			// initialize meta data w/ robustness for superset of cols
			if(meta != null) {
				String[] colnames2 = meta.getColumnNames();
				if(!TfMetaUtils.isIDSpec(jSpec) && colnames != null && colnames2 != null &&
					!ArrayUtils.isEquals(colnames, colnames2)) {
					HashMap<String, Integer> colPos = getColumnPositions(colnames2);
					// create temporary meta frame block w/ shallow column copy
					FrameBlock meta2 = new FrameBlock(meta.getSchema(), colnames2);
					meta2.setNumRows(meta.getNumRows());
					for(int i = 0; i < colnames.length; i++) {
						if(!colPos.containsKey(colnames[i])) {
							throw new DMLRuntimeException("Column name not found in meta data: " + colnames[i]
								+ " (meta: " + Arrays.toString(colnames2) + ")");
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

	private static void addEncoderToMap(ColumnEncoder encoder, HashMap<Integer, List<ColumnEncoder>> map) {
		if(!map.containsKey(encoder._colID)) {
			map.put(encoder._colID, new ArrayList<>());
		}
		map.get(encoder._colID).add(encoder);
	}

	public static int getEncoderType(ColumnEncoder columnEncoder) {
		if(columnEncoder instanceof ColumnEncoderBin)
			return EncoderType.Bin.ordinal();
		else if(columnEncoder instanceof ColumnEncoderDummycode)
			return EncoderType.Dummycode.ordinal();
		else if(columnEncoder instanceof ColumnEncoderFeatureHash)
			return EncoderType.FeatureHash.ordinal();
		else if(columnEncoder instanceof ColumnEncoderPassThrough)
			return EncoderType.PassThrough.ordinal();
		else if(columnEncoder instanceof ColumnEncoderRecode)
			return EncoderType.Recode.ordinal();
		throw new DMLRuntimeException("Unsupported encoder type: " + columnEncoder.getClass().getCanonicalName());
	}

	public static ColumnEncoder createInstance(int type) {
		EncoderType etype = EncoderType.values()[type];
		switch(etype) {
			case Bin:
				return new ColumnEncoderBin();
			case Dummycode:
				return new ColumnEncoderDummycode();
			case FeatureHash:
				return new ColumnEncoderFeatureHash();
			case PassThrough:
				return new ColumnEncoderPassThrough();
			case Recode:
				return new ColumnEncoderRecode();
			default:
				throw new DMLRuntimeException("Unsupported encoder type: " + etype);
		}
	}

	private static HashMap<String, Integer> getColumnPositions(String[] colnames) {
		HashMap<String, Integer> ret = new HashMap<>();
		for(int i = 0; i < colnames.length; i++)
			ret.put(colnames[i], i);
		return ret;
	}

}

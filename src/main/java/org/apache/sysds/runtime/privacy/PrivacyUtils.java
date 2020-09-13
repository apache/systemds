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

package org.apache.sysds.runtime.privacy;

import org.apache.sysds.api.DMLException;
import org.apache.sysds.parser.Expression;
import org.apache.sysds.parser.StringIdentifier;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;
import org.apache.wink.json4j.JSON;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONArtifact;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class PrivacyUtils {

	public static void setFineGrainedPrivacy(PrivacyConstraint privacyConstraint, Expression eFineGrainedPrivacy){
		FineGrainedPrivacy fineGrainedPrivacy = privacyConstraint.getFineGrainedPrivacy();
		StringIdentifier fgPrivacyIdentifier = (StringIdentifier) eFineGrainedPrivacy;
		String fgPrivacyValue = fgPrivacyIdentifier.getValue();
		try {
			putFineGrainedConstraintsFromString(fineGrainedPrivacy, fgPrivacyValue);
		} catch (JSONException exception){
			throw new DMLException("JSONException: " + exception);
		}
		privacyConstraint.setFineGrainedPrivacyConstraints(fineGrainedPrivacy);
	}

	private static void putFineGrainedConstraintsFromString(FineGrainedPrivacy fineGrainedPrivacy, String fgPrivacyValue)
		throws JSONException {
		JSONArtifact fgPrivacyJson = JSON.parse(fgPrivacyValue);
		JSONObject fgPrivacyObject = (JSONObject)fgPrivacyJson;
		JSONArray keys = fgPrivacyObject.names();
		for ( int i = 0; i < keys.length(); i++ ){
			String key = keys.getString(i);
			putFineGrainedConstraint(fgPrivacyObject, fineGrainedPrivacy, key);
		}
	}

	private static void putFineGrainedConstraint(JSONObject fgPrivacyObject, FineGrainedPrivacy fineGrainedPrivacy, String key)
		throws JSONException {
		JSONArray privateArray = fgPrivacyObject.getJSONArray(key);
		for (Object range : privateArray.toArray()){
			DataRange dataRange = getDataRangeFromObject(range);
			fineGrainedPrivacy.put(dataRange, PrivacyLevel.valueOf(key));
		}
	}

	private static DataRange getDataRangeFromObject(Object range) throws JSONException {
		JSONArray beginDims = ((JSONArray)range).getJSONArray(0);
		JSONArray endDims = ((JSONArray)range).getJSONArray(1);
		long[] beginDimsLong = new long[beginDims.length()];
		long[] endDimsLong = new long[endDims.length()];
		for ( int dimIndex = 0; dimIndex < beginDims.length(); dimIndex++ ){
			beginDimsLong[dimIndex] = beginDims.getLong(dimIndex);
			endDimsLong[dimIndex] = endDims.getLong(dimIndex);
		}
		return new DataRange(beginDimsLong, endDimsLong);
	}
}

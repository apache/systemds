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

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacyList;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;
import org.apache.wink.json4j.OrderedJSONObject;

/**
 * PrivacyConstraint holds all privacy constraints for data in the system at
 * compile time and runtime.
 */
public class PrivacyConstraint implements Externalizable
{
	public enum PrivacyLevel {
		None,               // No data exchange constraints. Data can be shared with anyone.
		Private,            // Data cannot leave the origin.
		PrivateAggregation  // Only aggregations of the data can leave the origin.
	}

	protected PrivacyLevel privacyLevel = PrivacyLevel.None;
	protected FineGrainedPrivacy fineGrainedPrivacy;
	
	/**
	 * Basic Constructor with a fine-grained collection 
	 * based on a list implementation.
	 */
	public PrivacyConstraint(){
		this(new FineGrainedPrivacyList());
	}

	/**
	 * Constructor with the option to choose between 
	 * different fine-grained collection implementations.
	 * @param fineGrainedPrivacyCollection the instance in which fine-grained constraints are stored
	 */
	public PrivacyConstraint(FineGrainedPrivacy fineGrainedPrivacyCollection){
		setFineGrainedPrivacyConstraints(fineGrainedPrivacyCollection);
	}

	/**
	 * Constructor with default fine-grained collection implementation
	 * where the entire data object is set to the given privacy level.
	 * @param privacyLevel for the entire data object.
	 */
	public PrivacyConstraint(PrivacyLevel privacyLevel) {
		this();
		setPrivacyLevel(privacyLevel);
	}

	public void setPrivacyLevel(PrivacyLevel privacyLevel){
		this.privacyLevel = privacyLevel;
	}

	public PrivacyLevel getPrivacyLevel(){
		return privacyLevel;
	}

	/**
	 * Checks if fine-grained privacy is set for this privacy constraint. 
	 * @return true if the privacy constraint has fine-grained constraints.
	 */
	public boolean hasFineGrainedConstraints(){
		return fineGrainedPrivacy.hasConstraints();
	}

	/**
	 * Sets fine-grained privacy for the privacy constraint. 
	 * Existing fine-grained privacy collection will be overwritten.
	 * @param fineGrainedPrivacy fine-grained privacy instance which is set for the privacy constraint
	 */
	public void setFineGrainedPrivacyConstraints(FineGrainedPrivacy fineGrainedPrivacy){
		this.fineGrainedPrivacy = fineGrainedPrivacy;
	}

	/**
	 * Get fine-grained privacy instance. 
	 * @return fine-grained privacy instance
	 */
	public FineGrainedPrivacy getFineGrainedPrivacy(){
		return fineGrainedPrivacy;
	}

	/**
	 * Return true if any of the elements has privacy level private
	 * @return true if any element has privacy level private
	 */
	public boolean hasPrivateElements(){
		if (privacyLevel == PrivacyLevel.Private) return true;
		if ( hasFineGrainedConstraints() ){
			DataRange[] dataRanges = fineGrainedPrivacy.getDataRangesOfPrivacyLevel(PrivacyLevel.Private);
			return dataRanges != null && dataRanges.length > 0;
		} else return false;
	}

	/**
	 * Return true if any constraints have level Private or PrivateAggregate.
	 * @return true if any constraints have level Private or PrivateAggregate
	 */
	public boolean hasConstraints(){
		if ( privacyLevel != null && 
			(privacyLevel == PrivacyLevel.Private || privacyLevel == PrivacyLevel.PrivateAggregation) )
			return true;
		else if ( hasFineGrainedConstraints() ){
			DataRange[] privateRanges = fineGrainedPrivacy.getDataRangesOfPrivacyLevel(PrivacyLevel.Private);
			DataRange[] aggregateRanges = fineGrainedPrivacy.getDataRangesOfPrivacyLevel(PrivacyLevel.PrivateAggregation);
			return (privateRanges != null && privateRanges.length > 0) 
				|| (aggregateRanges != null && aggregateRanges.length > 0);
		} else return false;
	}

	/**
	 * Get privacy constraints and put them into JSON object. 
	 * @param json JSON object in which the privacy constraints are put
	 * @throws JSONException in case of errors in putting into JSON object
	 */
	public void toJson(JSONObject json) throws JSONException {
		if ( getPrivacyLevel() != null && getPrivacyLevel() != PrivacyLevel.None )
			json.put(DataExpression.PRIVACY, getPrivacyLevel().name());
		if ( hasFineGrainedConstraints() ) {
			DataRange[] privateRanges = getFineGrainedPrivacy().getDataRangesOfPrivacyLevel(PrivacyLevel.Private);
			JSONArray privateRangesJson = getJsonArray(privateRanges);
			
			DataRange[] aggregateRanges = getFineGrainedPrivacy().getDataRangesOfPrivacyLevel(PrivacyLevel.PrivateAggregation);
			JSONArray aggregateRangesJson = getJsonArray(aggregateRanges);
			
			OrderedJSONObject rangesJson = new OrderedJSONObject();
			rangesJson.put(PrivacyLevel.Private.name(), privateRangesJson);
			rangesJson.put(PrivacyLevel.PrivateAggregation.name(), aggregateRangesJson);
			json.put(DataExpression.FINE_GRAINED_PRIVACY, rangesJson);
		}
	}

	private static JSONArray getJsonArray(DataRange[] ranges) throws JSONException {
		JSONArray rangeObjects = new JSONArray();
		for ( DataRange range : ranges ){
			List<Long> rangeBegin = Arrays.stream(range.getBeginDims()).boxed().collect(Collectors.toList());
			List<Long> rangeEnd = Arrays.stream(range.getEndDims()).boxed().collect(Collectors.toList());
			JSONArray beginJson = new JSONArray(rangeBegin);
			JSONArray endJson = new JSONArray(rangeEnd);
			JSONArray rangeObject = new JSONArray();
			rangeObject.put(beginJson);
			rangeObject.put(endJson);
			rangeObjects.add(rangeObject);
		}
		return rangeObjects;
	}

	@Override
	public void readExternal(ObjectInput is) throws IOException {
		this.privacyLevel = PrivacyLevel.values()[is.readInt()];
		int fineGrainedConstraintLength = is.readInt();
		if ( fineGrainedConstraintLength > 0 ){
			for (int i = 0; i < fineGrainedConstraintLength; i++){
				int levelIndex = is.readInt();
				PrivacyLevel rangePrivacy = PrivacyLevel.values()[levelIndex];
				DataRange dataRange = readExternalDataRangeObject(is);
				fineGrainedPrivacy.put(dataRange, rangePrivacy);
			}
		}
	}

	@Override
	public void writeExternal(ObjectOutput objectOutput) throws IOException {
		objectOutput.writeInt(getPrivacyLevel().ordinal());
		
		if (fineGrainedPrivacy != null && fineGrainedPrivacy.hasConstraints()){
			List<Entry<DataRange,PrivacyLevel>> fineGrainedConstraints = fineGrainedPrivacy.getAllConstraintsList();
			objectOutput.writeInt(fineGrainedConstraints.size());
			for ( Entry<DataRange,PrivacyLevel> constraint : fineGrainedConstraints ) {
				objectOutput.writeInt(constraint.getValue().ordinal());
				DataRange dataRange = constraint.getKey();
				objectOutput.writeInt(dataRange.getBeginDims().length);
				writeExternalRangeDim(objectOutput, dataRange.getBeginDims());
				writeExternalRangeDim(objectOutput, dataRange.getEndDims());
			}
		}
		else {
			objectOutput.writeInt(0);
		}
	}

	/**
	 * Reads a DataRange from ObjectInput. 
	 * @param is ObjectInput from which the DataRange is read
	 * @return DataRange from ObjectInput
	 * @throws IOException if an I/O error occurs during read
	 */
	private static DataRange readExternalDataRangeObject(ObjectInput is) throws IOException {
		int dimLength = is.readInt();
		long[] beginDims = readExternalDataRangeDim(is, dimLength);
		long[] endDims = readExternalDataRangeDim(is, dimLength);
		return new DataRange(beginDims, endDims);
	}

	/**
	 * Read a long array of the specified length from object input. 
	 * @param is ObjectInput from which the long array is read
	 * @param dimLength length of input long array
	 * @return the input array as a long array
	 * @throws IOException if an I/O error occurs during read
	 */
	private static long[] readExternalDataRangeDim(ObjectInput is, int dimLength) throws IOException {
		long[] dims = new long[dimLength];
		for(int i = 0; i < dimLength; i++){
			dims[i] = is.readLong();
		}
		return dims;
	}

	/**
	 * Write the long array to ObjectOutput.
	 * @param objectOutput ObjectOutput in which the long array is written.
	 * @param rangeDim long array to write in ObjectOutput. 
	 * @throws IOException if an I/O error occurs during write
	 */
	private static void writeExternalRangeDim(ObjectOutput objectOutput, long[] rangeDim) throws IOException {
		for ( long beginIndex : rangeDim ){
			objectOutput.writeLong(beginIndex);
		}
	}

	@Override
	public boolean equals(Object other){
		if ( other instanceof PrivacyConstraint ){
			PrivacyConstraint otherPrivacyConstraint = (PrivacyConstraint) other;
			return otherPrivacyConstraint.privacyLevel == privacyLevel
				&& otherPrivacyConstraint.getFineGrainedPrivacy().equals(fineGrainedPrivacy);
		} else return false;
	}

	@Override
	public String toString(){
		String constraintString = "General privacy level: " + privacyLevel;
		if ( fineGrainedPrivacy != null && fineGrainedPrivacy.hasConstraints() )
			constraintString = constraintString + System.getProperty("line.separator")
				+ "Fine-grained privacy level: " + fineGrainedPrivacy.toString();
		return constraintString;
	}

}

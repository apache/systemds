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

package org.apache.sysds.runtime.iogen;

import org.apache.sysds.common.Types;

public class MappingProperties {

	public enum RepresentationProperties {
		TYPICAL,
		SYMMETRIC, SKEWSYMMETRIC,
		PATTERN,
		ARRAYCOLWISE, ARRAYROWWISE;
		@Override
		public String toString() {
			return this.name().toUpperCase();
		}
	}

	public enum RecordProperties {
		SINGLELINE, MULTILINE;
		@Override
		public String toString() {
			return this.name().toUpperCase();
		}
	}

	public enum DataProperties {
		FULLEXIST, PARTIALLYEXIST, NOTEXIST;
		@Override
		public String toString() {
			return this.name().toUpperCase();
		}
	}

	private RepresentationProperties representationProperties;
	private RecordProperties recordProperties;
	private DataProperties dataProperties;
	private Object patternValue;
	private Types.ValueType patternValueType;

	public void setSymmetricRepresentation(){
		this.representationProperties = RepresentationProperties.SYMMETRIC;
	}

	public void setSkewSymmetricRepresentation(){
		this.representationProperties = RepresentationProperties.SKEWSYMMETRIC;
	}

	public void setPatternRepresentation(Types.ValueType valueType, Object value){
		this.representationProperties = RepresentationProperties.PATTERN;
		this.patternValueType = valueType;
		this.patternValue = value;
	}

	public void setTypicalRepresentation(){
		this.representationProperties = RepresentationProperties.TYPICAL;
	}

	public void setArrayColWiseRepresentation(){
		this.representationProperties = RepresentationProperties.ARRAYCOLWISE;
	}

	public void setArrayRowWiseRepresentation(){
		this.representationProperties = RepresentationProperties.ARRAYROWWISE;
	}

	public void setDataFullExist(){
		this.dataProperties = DataProperties.FULLEXIST;
	}

	public void setDataNotExist(){
		this.dataProperties = DataProperties.NOTEXIST;
	}

	public void setDataPartiallyExist(){
		this.dataProperties = DataProperties.PARTIALLYEXIST;
	}

	public void setRecordSingleLine(){
		this.recordProperties = RecordProperties.SINGLELINE;
	}

	public void setRecordMultiLine(){
		this.recordProperties = RecordProperties.MULTILINE;
	}

	public boolean isRepresentation(){
		return this.representationProperties != null;
	}

	public RepresentationProperties getRepresentationProperties() {
		return representationProperties;
	}

	public RecordProperties getRecordProperties() {
		return recordProperties;
	}

	public DataProperties getDataProperties() {
		return dataProperties;
	}

	public Object getPatternValue() {
		return patternValue;
	}

	public Types.ValueType getPatternValueType() {
		return patternValueType;
	}
}

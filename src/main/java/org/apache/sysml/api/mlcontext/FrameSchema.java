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

package org.apache.sysml.api.mlcontext;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.apache.sysml.parser.Expression.ValueType;

/**
 * The frame schema, stored as a list of {@code ValueType} values.
 *
 */
public class FrameSchema {

	private List<ValueType> schema = null;

	public FrameSchema() {
	}

	/**
	 * Constructor that specifies the schema as a list of {@code ValueType}
	 * values.
	 *
	 * @param schema
	 *            the frame schema
	 */
	public FrameSchema(List<ValueType> schema) {
		this.schema = schema;
	}

	/**
	 * Constructor that specifies the schema as a comma-separated string.
	 *
	 * @param schema
	 *            the frame schema as a string
	 */
	public FrameSchema(String schema) {
		this.schema = schemaStringToListOfValueTypes(schema);
	}

	/**
	 * Obtain the frame schema
	 *
	 * @return the frame schema as a list of {@code ValueType} values
	 */
	public List<ValueType> getSchema() {
		return schema;
	}

	/**
	 * Set the frame schema
	 *
	 * @param schema
	 *            the frame schema
	 */
	public void setSchema(List<ValueType> schema) {
		this.schema = schema;
	}

	/**
	 * Set the frame schema, specifying the frame schema as a comma-separated
	 * string
	 *
	 * @param schema
	 *            the frame schema as a string
	 */
	public void setSchemaAsString(String schema) {
		this.schema = schemaStringToListOfValueTypes(schema);
	}

	/**
	 * Convert a schema string to a list of {@code ValueType} values
	 *
	 * @param schemaString
	 *            the frame schema as a string
	 * @return the frame schema as a list of {@code ValueType} values
	 */
	private List<ValueType> schemaStringToListOfValueTypes(String schemaString) {
		if (StringUtils.isBlank(schemaString)) {
			return null;
		}
		String[] cols = schemaString.split(",");
		List<ValueType> list = new ArrayList<ValueType>();
		for (String col : cols) {
			list.add(ValueType.valueOf(col.toUpperCase()));
		}
		return list;
	}

	/**
	 * Obtain the schema as a comma-separated string
	 *
	 * @return the frame schema as a string
	 */
	public String getSchemaAsString() {
		if ((schema == null) || (schema.size() == 0)) {
			return null;
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < schema.size(); i++) {
			ValueType vt = schema.get(i);
			sb.append(vt);
			if (i + 1 < schema.size()) {
				sb.append(",");
			}
		}
		return sb.toString();
	}
}

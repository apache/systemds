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

package org.apache.sysds.runtime.controlprogram.federated.monitoring.services;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.Request;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Types;

public class MapperService {
	public static <T extends BaseModel> T getModelFromBody(Request request, Class<T> classType) {
		ObjectMapper mapper = new ObjectMapper();
		mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

		try {
			if (!request.getBody().isEmpty() && !request.getBody().isBlank()) {
				return mapper.readValue(request.getBody(), classType);
			}

			return classType.getDeclaredConstructor().newInstance();
		}
		catch (IOException | InvocationTargetException | IllegalAccessException | InstantiationException |
			   NoSuchMethodException e) {
			throw new RuntimeException(e);
		}
	}

	public static <T extends BaseModel> T mapResultToModel(ResultSet resultSet, Class<T> classType) {
		try {

			var result = classType.getDeclaredConstructor().newInstance();
			var fields = result.getClass().getFields();

			for (int column = 1; column <= resultSet.getMetaData().getColumnCount(); column++) {

				var colName = resultSet.getMetaData().getColumnName(column);

				for (var field: fields) {
					var fieldName = field.getName();
					if (colName.equalsIgnoreCase(fieldName)) {
						if (resultSet.getMetaData().getColumnType(column) == Types.VARCHAR) {
							result.getClass().getField(fieldName).set(result, resultSet.getString(column));
						} else if (resultSet.getMetaData().getColumnType(column) == Types.DOUBLE) {
							result.getClass().getField(fieldName).set(result, resultSet.getDouble(column));
						} else if (resultSet.getMetaData().getColumnType(column) == Types.INTEGER) {
							result.getClass().getField(fieldName).set(result, resultSet.getLong(column));
						} else if (resultSet.getMetaData().getColumnType(column) == Types.TIMESTAMP) {
							result.getClass().getField(fieldName).set(result, resultSet.getTimestamp(column).toLocalDateTime());
						}
					}
				}
			}

			return result;
		} catch (SQLException | NoSuchMethodException | InvocationTargetException | InstantiationException |
				 IllegalAccessException | NoSuchFieldException e) {
			throw new RuntimeException(e);
		}
	}
}

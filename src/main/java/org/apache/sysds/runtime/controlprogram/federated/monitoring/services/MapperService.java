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
import com.fasterxml.jackson.databind.SerializationFeature;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.NodeEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.Request;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatsEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.Constants;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.EntityEnum;

import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Types;

public class MapperService {
	public static BaseEntityModel getModelFromBody(Request request) {
		ObjectMapper mapper = new ObjectMapper();

		mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

		try {
			if (!request.getBody().isBlank() && !request.getBody().isEmpty()) {
				return mapper.readValue(request.getBody(), NodeEntityModel.class);
			}

			return null;
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public static BaseEntityModel mapEntityToModel(ResultSet resultSet, EntityEnum targetModel) {
		try {
			if (targetModel != EntityEnum.WORKER_STATS) {
				NodeEntityModel tmpModel = new NodeEntityModel();

				for (int column = 1; column <= resultSet.getMetaData().getColumnCount(); column++) {
					if (resultSet.getMetaData().getColumnType(column) == Types.INTEGER) {
						tmpModel.setId(resultSet.getLong(column));
					}

					if (resultSet.getMetaData().getColumnType(column) == Types.VARCHAR) {
						if (resultSet.getMetaData().getColumnName(column).equalsIgnoreCase(Constants.ENTITY_NAME_COL)) {
							tmpModel.setName(resultSet.getString(column));
						} else if (resultSet.getMetaData().getColumnName(column).equalsIgnoreCase(Constants.ENTITY_ADDR_COL)) {
							tmpModel.setAddress(resultSet.getString(column));
						}
					}
				}
				return tmpModel;
			} else {
				StatsEntityModel tmpModel = new StatsEntityModel();

				for (int column = 1; column <= resultSet.getMetaData().getColumnCount(); column++) {

					if (resultSet.getMetaData().getColumnType(column) == Types.VARCHAR) {
						if (resultSet.getMetaData().getColumnName(column).equalsIgnoreCase(Constants.ENTITY_TRAFFIC_COL)) {
							tmpModel.setTransferredBytes(resultSet.getString(column));
						} else if (resultSet.getMetaData().getColumnName(column).equalsIgnoreCase(Constants.ENTITY_HEAVY_HITTERS_COL)) {
							tmpModel.setHeavyHitterInstructions(resultSet.getString(column));
						}
					} else if (resultSet.getMetaData().getColumnType(column) == Types.DOUBLE) {
						if (resultSet.getMetaData().getColumnName(column).equalsIgnoreCase(Constants.ENTITY_CPU_COL)) {
							tmpModel.setCPUUsage(resultSet.getDouble(column));
						} else if (resultSet.getMetaData().getColumnName(column).equalsIgnoreCase(Constants.ENTITY_MEM_COL)) {
							tmpModel.setMemoryUsage(resultSet.getDouble(column));
						}
					} else {
						if (resultSet.getMetaData().getColumnName(column).equalsIgnoreCase(Constants.ENTITY_TIMESTAMP_COL)) {
							tmpModel.setTimestamp(resultSet.getTimestamp(column));
						}
					}
				}

				return tmpModel;
			}
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}
	}
}

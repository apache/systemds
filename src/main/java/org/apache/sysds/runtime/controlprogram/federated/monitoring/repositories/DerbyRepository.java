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

package org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories;

import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseEntityModel;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Types;
import java.util.ArrayList;
import java.util.List;

public class DerbyRepository implements IRepository {
	private final static String DB_CONNECTION = "jdbc:derby:memory:derbyDB";
	private final Connection _db;

	private static final String WORKERS_TABLE_NAME= "workers";
	private static final String ENTITY_NAME_COL = "name";
	private static final String ENTITY_ADDR_COL = "address";

	private static final String ENTITY_SCHEMA_CREATE_STMT = "CREATE TABLE %s " +
			"(id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY (START WITH 1, INCREMENT BY 1), " +
			"%s VARCHAR(60), " +
			"%s VARCHAR(120))";
	private static final String ENTITY_INSERT_STMT = "INSERT INTO %s (%s, %s) VALUES (?, ?)";

	private static final String GET_ENTITY_WITH_ID_STMT = "SELECT * FROM %s WHERE id = ?";
	private static final String GET_ALL_ENTITIES_STMT = "SELECT * FROM %s";

	public DerbyRepository() {
		_db = createMonitoringDatabase();
	}

	private Connection createMonitoringDatabase() {
		Connection db = null;
		try {
			// Creates only if DB doesn't exist
			db = DriverManager.getConnection(DB_CONNECTION + ";create=true");
			createMonitoringEntitiesInDB(db);

			return db;
		}
		catch (SQLException e) {
			throw new RuntimeException(e);
		}
	}

	private void createMonitoringEntitiesInDB(Connection db) {
		try {
			var dbMetaData = db.getMetaData();
			var workersExist = dbMetaData.getTables(null, null, WORKERS_TABLE_NAME.toUpperCase(),null);

			// Check if table already exists and create if not
			if(!workersExist.next())
			{
				PreparedStatement st = db.prepareStatement(
						String.format(ENTITY_SCHEMA_CREATE_STMT, WORKERS_TABLE_NAME, ENTITY_NAME_COL, ENTITY_ADDR_COL));
				st.executeUpdate();

			}
		}
		catch (SQLException e) {
			throw new RuntimeException(e);
		}
	}

	public void createEntity(EntityEnum type, BaseEntityModel model) {

		try {
			PreparedStatement st = _db.prepareStatement(
					String.format(ENTITY_INSERT_STMT, WORKERS_TABLE_NAME, ENTITY_NAME_COL, ENTITY_ADDR_COL));

			if (type == EntityEnum.COORDINATOR) {
				// Change statement
			}

			st.setString(1, model.getName());
			st.setString(2, model.getAddress());
			st.executeUpdate();

		} catch (SQLException e) {
			throw new RuntimeException(e);
		}
	}

	public BaseEntityModel getEntity(EntityEnum type, Long id) {
		BaseEntityModel resultModel = null;

		try {
			PreparedStatement st = _db.prepareStatement(
					String.format(GET_ENTITY_WITH_ID_STMT, WORKERS_TABLE_NAME));

			if (type == EntityEnum.COORDINATOR) {
				// Change statement
			}

			st.setLong(1, id);
			var resultSet = st.executeQuery();

			if (resultSet.next()){
				resultModel = mapEntityToModel(resultSet);
			}
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}

		return resultModel;
	}

	public List<BaseEntityModel> getAllEntities(EntityEnum type) {
		List<BaseEntityModel> resultModels = new ArrayList<>();

		try {
			PreparedStatement st = _db.prepareStatement(
					String.format(GET_ALL_ENTITIES_STMT, WORKERS_TABLE_NAME));

			if (type == EntityEnum.COORDINATOR) {
				// Change statement
			}

			var resultSet = st.executeQuery();

			while (resultSet.next()){
				resultModels.add(mapEntityToModel(resultSet));
			}
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}

		return resultModels;
	}

	private BaseEntityModel mapEntityToModel(ResultSet resultSet) throws SQLException {
		BaseEntityModel tmpModel = new BaseEntityModel();

		for (int column = 1; column <= resultSet.getMetaData().getColumnCount(); column++) {
			if (resultSet.getMetaData().getColumnType(column) == Types.INTEGER) {
				tmpModel.setId(resultSet.getLong(column));
			}

			if (resultSet.getMetaData().getColumnType(column) == Types.VARCHAR) {
				if (resultSet.getMetaData().getColumnName(column).equalsIgnoreCase(ENTITY_NAME_COL)) {
					tmpModel.setName(resultSet.getString(column));
				} else if (resultSet.getMetaData().getColumnName(column).equalsIgnoreCase(ENTITY_ADDR_COL)) {
					tmpModel.setAddress(resultSet.getString(column));
				}
			}
		}
		return tmpModel;
	}
}

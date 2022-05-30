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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.NodeEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatsEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.services.MapperService;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

public class DerbyRepository implements IRepository {
	private final static String DB_CONNECTION = "jdbc:derby:memory:derbyDB";
	private final Connection _db;
	private static final String ENTITY_SCHEMA_CREATE_STMT = "CREATE TABLE %s " +
			"(id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY (START WITH 1, INCREMENT BY 1), " +
			"%s VARCHAR(60), " +
			"%s VARCHAR(120))";
	private static final String ENTITY_SCHEMA_CREATE_STATS_STMT = "CREATE TABLE %s " +
			"(id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY (START WITH 1, INCREMENT BY 1), " +
			"%s INTEGER, " +
			"%s DOUBLE, " +
			"%s DOUBLE," +
			"%s VARCHAR(1000)," +
			"%s VARCHAR(1000))";
	private static final String ENTITY_INSERT_STMT = "INSERT INTO %s (%s, %s) VALUES (?, ?)";
	private static final String ENTITY_STATS_INSERT_STMT = "INSERT INTO %s (%s, %s, %s, %s, %s) VALUES (?, ?, ?, ?, ?)";
	private static final String GET_ENTITY_WITH_COL_STMT = "SELECT * FROM %s WHERE %s = ?";
	private static final String DELETE_ENTITY_WITH_COL_STMT = "DELETE FROM %s WHERE %s = ?";
	private static final String UPDATE_ENTITY_WITH_COL_STMT = "UPDATE %s SET %s = ?, %s = ? WHERE %s = ?";
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
			var workersExist = dbMetaData.getTables(null, null, Constants.WORKERS_TABLE_NAME.toUpperCase(),null);
			var statsExist = dbMetaData.getTables(null, null, Constants.STATS_TABLE_NAME.toUpperCase(),null);
			var coordinatorsExist = dbMetaData.getTables(null, null, Constants.COORDINATORS_TABLE_NAME.toUpperCase(),null);

			// Check if table already exists and create if not
			if(!workersExist.next())
			{
				PreparedStatement st = db.prepareStatement(
						String.format(ENTITY_SCHEMA_CREATE_STMT, Constants.WORKERS_TABLE_NAME, Constants.ENTITY_NAME_COL, Constants.ENTITY_ADDR_COL));
				st.executeUpdate();

			}

			if(!statsExist.next())
			{
				PreparedStatement st = db.prepareStatement(
						String.format(ENTITY_SCHEMA_CREATE_STATS_STMT, Constants.STATS_TABLE_NAME,
								Constants.ENTITY_WORKER_ID_COL,
								Constants.ENTITY_CPU_COL,
								Constants.ENTITY_MEM_COL,
								Constants.ENTITY_TRAFFIC_COL,
								Constants.ENTITY_HEAVY_HITTERS_COL));
				st.executeUpdate();

			}

			if(!coordinatorsExist.next())
			{
				PreparedStatement st = db.prepareStatement(
						String.format(ENTITY_SCHEMA_CREATE_STMT, Constants.COORDINATORS_TABLE_NAME, Constants.ENTITY_NAME_COL, Constants.ENTITY_ADDR_COL));
				st.executeUpdate();

			}
		}
		catch (SQLException e) {
			throw new RuntimeException(e);
		}
	}

	public Long createEntity(EntityEnum type, BaseEntityModel model) {

		PreparedStatement st = null;
		long id = -1L;

		try {

			if (type == EntityEnum.WORKER_STATS) {
				st = _db.prepareStatement(
						String.format(ENTITY_STATS_INSERT_STMT, Constants.STATS_TABLE_NAME,
								Constants.ENTITY_WORKER_ID_COL,
								Constants.ENTITY_CPU_COL,
								Constants.ENTITY_MEM_COL,
								Constants.ENTITY_TRAFFIC_COL,
								Constants.ENTITY_HEAVY_HITTERS_COL), PreparedStatement.RETURN_GENERATED_KEYS);

				StatsEntityModel newModel = (StatsEntityModel) model;

				st.setLong(1, newModel.getWorkerId());
				st.setDouble(2, newModel.getCPUUsage());
				st.setDouble(3, newModel.getMemoryUsage());
				st.setString(4, newModel.getTransferredBytes());
				st.setString(5, newModel.getHeavyHitterInstructions());
			} else {
				st = _db.prepareStatement(
						String.format(ENTITY_INSERT_STMT, Constants.WORKERS_TABLE_NAME, Constants.ENTITY_NAME_COL, Constants.ENTITY_ADDR_COL),
						PreparedStatement.RETURN_GENERATED_KEYS);
				NodeEntityModel newModel = (NodeEntityModel) model;

				if (type == EntityEnum.COORDINATOR) {
					st = _db.prepareStatement(
							String.format(ENTITY_INSERT_STMT, Constants.COORDINATORS_TABLE_NAME, Constants.ENTITY_NAME_COL, Constants.ENTITY_ADDR_COL),
							PreparedStatement.RETURN_GENERATED_KEYS);
				}

				st.setString(1, newModel.getName());
				st.setString(2, newModel.getAddress());
			}

			st.executeUpdate();

			ResultSet rs = st.getGeneratedKeys();
			if (rs.next()) {
				id = rs.getLong(1); // this is the auto-generated id key
			}

		} catch (SQLException e) {
			throw new RuntimeException(e);
		}

		return id;
	}

	public BaseEntityModel getEntity(EntityEnum type, Long id) {
		BaseEntityModel resultModel = null;

		try {
			PreparedStatement st = _db.prepareStatement(
					String.format(GET_ENTITY_WITH_COL_STMT, Constants.WORKERS_TABLE_NAME, Constants.ENTITY_ID_COL));

			if (type == EntityEnum.COORDINATOR) {
				st = _db.prepareStatement(
						String.format(GET_ENTITY_WITH_COL_STMT, Constants.COORDINATORS_TABLE_NAME, Constants.ENTITY_ID_COL));
			} else if (type == EntityEnum.WORKER_STATS) {
				st = _db.prepareStatement(
						String.format(GET_ENTITY_WITH_COL_STMT, Constants.STATS_TABLE_NAME, Constants.ENTITY_WORKER_ID_COL));
			}

			st.setLong(1, id);
			var resultSet = st.executeQuery();

			if (resultSet.next()){
				resultModel = MapperService.mapEntityToModel(resultSet, type);
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
					String.format(GET_ALL_ENTITIES_STMT, Constants.WORKERS_TABLE_NAME));

			if (type == EntityEnum.COORDINATOR) {
				st = _db.prepareStatement(
						String.format(GET_ALL_ENTITIES_STMT, Constants.COORDINATORS_TABLE_NAME));
			}

			var resultSet = st.executeQuery();

			while (resultSet.next()){
				resultModels.add(MapperService.mapEntityToModel(resultSet, type));
			}
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}

		return resultModels;
	}

	public List<BaseEntityModel> getAllEntitiesByField(EntityEnum type, Object fieldValue) {
		List<BaseEntityModel> resultModels = new ArrayList<>();
		PreparedStatement st = null;

		try {

			if (type == EntityEnum.WORKER_STATS) {
				st = _db.prepareStatement(
						String.format(GET_ENTITY_WITH_COL_STMT, Constants.STATS_TABLE_NAME, Constants.ENTITY_WORKER_ID_COL));
				st.setLong(1, (Long) fieldValue);
			} else {
				throw new NotImplementedException();
			}

			var resultSet = st.executeQuery();

			while (resultSet.next()){
				resultModels.add(MapperService.mapEntityToModel(resultSet, type));
			}
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}

		return resultModels;
	}

	@Override
	public void updateEntity(EntityEnum type, BaseEntityModel model) {

		try {
			PreparedStatement st = _db.prepareStatement(
					String.format(UPDATE_ENTITY_WITH_COL_STMT, Constants.WORKERS_TABLE_NAME,
							Constants.ENTITY_NAME_COL,
							Constants.ENTITY_ADDR_COL,
							Constants.ENTITY_ID_COL));
			NodeEntityModel editModel = (NodeEntityModel) model;

			if (type == EntityEnum.COORDINATOR) {
				st = _db.prepareStatement(
						String.format(UPDATE_ENTITY_WITH_COL_STMT, Constants.COORDINATORS_TABLE_NAME,
								Constants.ENTITY_NAME_COL,
								Constants.ENTITY_ADDR_COL,
								Constants.ENTITY_ID_COL));
			}

			st.setString(1, editModel.getName());
			st.setString(2, editModel.getAddress());
			st.setLong(3, editModel.getId());

			st.executeUpdate();

		} catch (SQLException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void removeEntity(EntityEnum type, Long id) {
		PreparedStatement st = null;

		try {

			if (type == EntityEnum.WORKER) {
				st = _db.prepareStatement(
						String.format(DELETE_ENTITY_WITH_COL_STMT, Constants.WORKERS_TABLE_NAME, Constants.ENTITY_ID_COL));

				st.setLong(1, id);
			} else {
				st = _db.prepareStatement(
						String.format(DELETE_ENTITY_WITH_COL_STMT, Constants.COORDINATORS_TABLE_NAME, Constants.ENTITY_ID_COL));

				st.setLong(1, id);
			}

			st.executeUpdate();

		} catch (SQLException e) {
			throw new RuntimeException(e);
		}
	}
}

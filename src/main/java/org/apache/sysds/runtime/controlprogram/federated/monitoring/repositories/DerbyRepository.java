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
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class DerbyRepository implements IRepository {
    private final static String DB_CONNECTION = "jdbc:derby:memory:derbyDB";
    private final Connection _db;

    private static final String WORKER_TABLE = "WORKERS";

    public DerbyRepository() {
        _db = createMonitoringDatabase();
    }

    private Connection createMonitoringDatabase() {
        Connection db = null;
        try {
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
            Statement st = db.createStatement();
            st.execute("CREATE TABLE " + WORKER_TABLE +
                    " (id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY (START WITH 1, INCREMENT BY 1)," +
                    "name VARCHAR(60)," +
                    "address VARCHAR(120))");
        }
        catch (SQLException e) {
            String derbyTableCreationError = "X0Y32";
            if(e.getSQLState().equals(derbyTableCreationError)) {
                // Known issue and a solution:
                // https://stackoverflow.com/questions/5866154/how-to-create-table-if-it-doesnt-exist-using-derby-db
                return;
            }
            throw new RuntimeException(e);
        }
    }

    public void createEntity(EntityEnum type, BaseEntityModel model) {
        String query = "";

        if (type == EntityEnum.WORKER) {
            query = "INSERT INTO " + WORKER_TABLE + "(name, address) VALUES " +
                    String.format("('%s', '%s')", model.getName(), model.getAddress());
        }

        executeQuery(type, query);
    }

    public ResultSet getEntity(EntityEnum type, Long id) {
        String query = "";

        if (type == EntityEnum.WORKER) {
            query = "SELECT * FROM " + WORKER_TABLE + " WHERE " +
                    String.format("id = %d", id);
        }

        return executeQuery(type, query);
    }

    public ResultSet getAllEntities(EntityEnum type) {
        String query = "";

        if (type == EntityEnum.WORKER) {
            query = "SELECT * FROM " + WORKER_TABLE;
        }

        return executeQuery(type, query);
    }

    private ResultSet executeQuery(EntityEnum type, String query) {
        try {
            Statement st = _db.createStatement();

            if (type == EntityEnum.WORKER) {
                st.execute(query);
            }

            return st.getResultSet();
        }
        catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }
}

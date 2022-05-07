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

import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.DerbyRepository;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.EntityEnum;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.IRepository;

import java.sql.SQLException;
import java.sql.Types;
import java.util.ArrayList;
import java.util.List;

public class WorkerService {
    private final IRepository _entityRepository = new DerbyRepository();

    public void create(BaseEntityModel model) {
        _entityRepository.createEntity(EntityEnum.WORKER, model);
    }

    public BaseEntityModel get(Long id) {
        return null;
    }

    public List<BaseEntityModel> getAll() {

        var resultSet = _entityRepository.getAllEntities(EntityEnum.WORKER);

        try {
            List<BaseEntityModel> resultModels = new ArrayList<>();

            while(resultSet.next()){
                BaseEntityModel tmpModel = new BaseEntityModel();
                for (int column = 1; column <= resultSet.getMetaData().getColumnCount(); column++) {
                    if (resultSet.getMetaData().getColumnType(column) == Types.INTEGER) {
                        tmpModel.setId(resultSet.getLong(column));
                    }

                    if (resultSet.getMetaData().getColumnType(column) == Types.VARCHAR) {
                        if (resultSet.getMetaData().getColumnName(column).equalsIgnoreCase("name")) {
                            tmpModel.setName(resultSet.getString(column));
                        } else if (resultSet.getMetaData().getColumnName(column).equalsIgnoreCase("address")) {
                            tmpModel.setAddress(resultSet.getString(column));
                        }
                    }
                }
                resultModels.add(tmpModel);
            }

            return resultModels;
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }
}

package org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories;

import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseEntityModel;

import java.sql.ResultSet;

public interface IRepository {
    public void createEntity(EntityEnum type, BaseEntityModel model);

    public ResultSet getEntity(EntityEnum type, Long id);

    public ResultSet getAllEntities(EntityEnum type);
}

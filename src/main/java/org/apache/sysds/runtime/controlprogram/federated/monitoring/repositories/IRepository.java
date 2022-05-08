package org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories;

import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseEntityModel;

import java.util.List;

public interface IRepository {
    void createEntity(EntityEnum type, BaseEntityModel model);

    BaseEntityModel getEntity(EntityEnum type, Long id);

    List<BaseEntityModel> getAllEntities(EntityEnum type);
}

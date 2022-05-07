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

package org.apache.sysds.runtime.controlprogram.federated.monitoring.models;

public class BaseEntityModel {
    private Long _id;
    private String _name;
    private String _address;

    public BaseEntityModel() { }

    public BaseEntityModel(final Long id, final String name, final String address) {
        this._id = id;
        this._name = name;
        this._address = address;
    }

    public Long getId() {
        return _id;
    }

    public void setId(final Long id) {
        this._id = id;
    }

    public String getName() {
        return _name;
    }

    public void setName(final String name) {
        this._name = name;
    }

    public String getAddress() {
        return _address;
    }

    public void setAddress(final String address) {
        this._address = address;
    }

    @Override
    public String toString()
    {
        return String.format("{" +
                "\"id\": %d," +
                "\"name\": \"%s\"," +
                "\"address\": \"%s\"" +
                "}", _id, _name, _address);
    }
}

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

package org.apache.sysds.test.functions.federated.monitoring;

import org.apache.sysds.test.functions.federated.multitenant.MultiTenantTestBase;
import org.junit.After;

public abstract class FederatedMonitoringTestBase extends MultiTenantTestBase {
    protected Process monitoringProcess;

    @Override
    public abstract void setUp();

    // ensure that the processes are killed - even if the test throws an exception
    @After
    public void stopMonitoringProcesses() {
        monitoringProcess.destroyForcibly();
    }

    /**
     * Start federated backend monitoring processes on available port
     *
     * @return int the port of the created federated backend monitoring
     */
    protected int startFedMonitoring(String[] addArgs) {
        int port = getRandomAvailablePort();
        monitoringProcess = startLocalFedMonitoring(port, addArgs);

        return port;
    }
}

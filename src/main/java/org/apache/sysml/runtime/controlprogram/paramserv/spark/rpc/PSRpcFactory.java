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

package org.apache.sysml.runtime.controlprogram.paramserv.spark.rpc;

import java.io.IOException;
import java.util.Collections;

import org.apache.spark.network.TransportContext;
import org.apache.spark.network.server.TransportServer;
import org.apache.spark.network.util.SystemPropertyConfigProvider;
import org.apache.spark.network.util.TransportConf;
import org.apache.sysml.runtime.controlprogram.paramserv.LocalParamServer;
import org.apache.sysml.runtime.controlprogram.paramserv.spark.SparkPSProxy;

public class PSRpcFactory {

	private static final String MODULE_NAME = "ps";

	private static TransportContext createTransportContext(LocalParamServer ps) {
		TransportConf conf = new TransportConf(MODULE_NAME, new SystemPropertyConfigProvider());
		PSRpcHandler handler = new PSRpcHandler(ps);
		return new TransportContext(conf, handler);
	}

	/**
	 * Create and start the server
	 * @return server
	 */
	public static TransportServer createServer(LocalParamServer ps, String host) {
		TransportContext context = createTransportContext(ps);
		return context.createServer(host, 0, Collections.emptyList());	// bind rpc to an ephemeral port
	}

	public static SparkPSProxy createSparkPSProxy(String host, int port, long rpcTimeout) throws IOException {
		TransportContext context = createTransportContext(new LocalParamServer());
		return new SparkPSProxy(context.createClientFactory().createClient(host, port), rpcTimeout);
	}
}

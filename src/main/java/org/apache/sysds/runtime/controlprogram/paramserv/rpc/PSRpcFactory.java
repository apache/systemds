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

package org.apache.sysds.runtime.controlprogram.paramserv.rpc;

import java.io.IOException;
import java.util.Collections;

import org.apache.spark.SparkConf;
import org.apache.spark.network.TransportContext;
import org.apache.spark.network.netty.SparkTransportConf;
import org.apache.spark.network.server.TransportServer;
import org.apache.spark.network.util.TransportConf;
import org.apache.spark.util.LongAccumulator;
import org.apache.sysds.runtime.controlprogram.paramserv.LocalParamServer;
import org.apache.sysds.runtime.controlprogram.paramserv.SparkPSProxy;

public class PSRpcFactory {

	private static final String MODULE_NAME = "ps";

	private static TransportContext createTransportContext(SparkConf conf, LocalParamServer ps) {
		TransportConf tc = SparkTransportConf.fromSparkConf(conf, MODULE_NAME, 0);
		PSRpcHandler handler = new PSRpcHandler(ps);
		return new TransportContext(tc, handler);
	}

	/**
	 * Create and start the server
	 * @param conf spark config
	 * @param ps LocalParamServer object
	 * @param host hostname
	 * @return server
	 */
	public static TransportServer createServer(SparkConf conf, LocalParamServer ps, String host) {
		TransportContext context = createTransportContext(conf, ps);
		return context.createServer(host, 0, Collections.emptyList());	// bind rpc to an ephemeral port
	}

	public static SparkPSProxy createSparkPSProxy(SparkConf conf, int port, LongAccumulator aRPC) throws IOException {
		long rpcTimeout = conf.contains("spark.rpc.askTimeout") ?
			conf.getTimeAsMs("spark.rpc.askTimeout") :
			conf.getTimeAsMs("spark.network.timeout", "120s");
		String host = conf.get("spark.driver.host");
		TransportContext context = createTransportContext(conf, new LocalParamServer());
		return new SparkPSProxy(context.createClientFactory().createClient(host, port), rpcTimeout, aRPC);
	}
}

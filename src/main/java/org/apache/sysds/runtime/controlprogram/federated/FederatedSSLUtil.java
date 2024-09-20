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

package org.apache.sysds.runtime.controlprogram.federated;

import java.net.InetSocketAddress;

import javax.net.ssl.SSLException;

import org.apache.sysds.runtime.DMLRuntimeException;

import io.netty.channel.socket.SocketChannel;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.SslHandler;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;

public class FederatedSSLUtil {

	private FederatedSSLUtil(){
		// private constructor.
	}

	/** A Singleton constructed SSL context, that only is assigned if ssl is enabled. */
	private static SslContextMan sslInstance = null;

	protected static SslContextMan SslConstructor() {
		if(sslInstance == null)
			return new SslContextMan();
		else
			return sslInstance;
	}

	protected static SslHandler createSSLHandler(SocketChannel ch, InetSocketAddress address) {
		return SslConstructor().context.newHandler(ch.alloc(), address.getAddress().getHostAddress(), address.getPort());
	}


	private static class SslContextMan {
		protected final SslContext context;

		private SslContextMan() {
			try {
				context = SslContextBuilder.forClient().trustManager(InsecureTrustManagerFactory.INSTANCE).build();
			}
			catch(SSLException e) {
				throw new DMLRuntimeException("Static SSL setup failed for client side", e);
			}
		}
	}
}

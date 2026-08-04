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

import java.io.File;
import java.net.InetSocketAddress;

import javax.net.ssl.SSLEngine;
import javax.net.ssl.SSLException;
import javax.net.ssl.SSLParameters;

import org.apache.log4j.Logger;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;

import io.netty.channel.socket.SocketChannel;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.SslHandler;

public class FederatedSSLUtil {
	private static final Logger LOG = Logger.getLogger(FederatedSSLUtil.class);

	// The password of an encrypted worker private key is read from the environment instead of the configuration,
	// so that it is not leaked when the configuration file is shared or published.
	public static final String SSL_KEY_PASSWORD_ENV = "SYSTEMDS_FEDERATED_SSL_KEY_PASSWORD";

	private FederatedSSLUtil(){
		// private constructor.
	}

	/** A Singleton constructed SSL context, that only is assigned if ssl is enabled. */
	private static SslContextMan sslInstance = null;

	protected synchronized static SslContextMan SslConstructor() {
		if(sslInstance == null)
			sslInstance = new SslContextMan();
		return sslInstance;
	}

	// Drop the cached client side SSL context, so that the next connection is built from the current configuration.
	// Only relevant if the configuration changes while the JVM is running, as it does in tests.
	public synchronized static void resetClientContext() {
		sslInstance = null;
	}

	protected static SslHandler createSSLHandler(SocketChannel ch, InetSocketAddress address) {
		final SslContextMan man = SslConstructor();
		// prefer the configured host name over the resolved address, since certificates are issued for host names.
		final String host = (address.getHostString() != null) ? address.getHostString() : address.getAddress()
			.getHostAddress();
		final SslHandler handler = man.context.newHandler(ch.alloc(), host, address.getPort());

		// the certificate of a worker has to be issued for the host it is contacted on, otherwise any worker
		// with a trusted certificate could impersonate any other worker.
		final SSLEngine engine = handler.engine();
		final SSLParameters params = engine.getSSLParameters();
		params.setEndpointIdentificationAlgorithm("HTTPS");
		engine.setSSLParameters(params);

		return handler;
	}

	/**
	 * Construct the SSL context of a federated worker, based on the certificate and private key configured via
	 * {@link DMLConfig#FEDERATED_SSL_CERT} and {@link DMLConfig#FEDERATED_SSL_KEY}. Both are required, a worker that
	 * cannot be authenticated by the coordinator is not supported. If the private key is encrypted, its password is
	 * read from the {@link #SSL_KEY_PASSWORD_ENV} environment variable.
	 *
	 * @return The server side SSL context of the federated worker
	 */
	public static SslContext createServerContext() {
		final DMLConfig conf = ConfigurationManager.getDMLConfig();
		final String certPath = conf.getTextValue(DMLConfig.FEDERATED_SSL_CERT);
		final String keyPath = conf.getTextValue(DMLConfig.FEDERATED_SSL_KEY);
		final String keyPassword = System.getenv(SSL_KEY_PASSWORD_ENV);

		if(!isSet(certPath) || !isSet(keyPath))
			throw new DMLRuntimeException("Federated SSL requires a signed certificate, configure the certificate "
				+ "chain in " + DMLConfig.FEDERATED_SSL_CERT + " and the matching private key in "
				+ DMLConfig.FEDERATED_SSL_KEY + ".");

		try {
			LOG.info("Federated worker SSL using certificate: " + certPath);
			return SslContextBuilder
				.forServer(readableFile(certPath, DMLConfig.FEDERATED_SSL_CERT),
					readableFile(keyPath, DMLConfig.FEDERATED_SSL_KEY), isSet(keyPassword) ? keyPassword : null)
				.build();
		}
		catch(SSLException e) {
			throw new DMLRuntimeException("Static SSL setup failed for worker side", e);
		}
	}

	private static boolean isSet(String value) {
		return value != null && !value.trim().isEmpty();
	}

	private static File readableFile(String path, String configName) {
		final File f = new File(path.trim());
		if(!f.canRead())
			throw new DMLRuntimeException(
				"Federated SSL file configured in " + configName + " is not a readable file: " + path);
		return f;
	}

	private static class SslContextMan {
		protected final SslContext context;

		private SslContextMan() {
			final DMLConfig conf = ConfigurationManager.getDMLConfig();
			final String trustPath = conf.getTextValue(DMLConfig.FEDERATED_SSL_TRUST);

			if(!isSet(trustPath))
				throw new DMLRuntimeException("Federated SSL requires the certificates that are trusted to sign "
					+ "worker certificates, configure them in " + DMLConfig.FEDERATED_SSL_TRUST + ".");

			try {
				LOG.debug("Federated SSL trusting certificates in: " + trustPath);
				context = SslContextBuilder.forClient()
					.trustManager(readableFile(trustPath, DMLConfig.FEDERATED_SSL_TRUST)).build();
			}
			catch(SSLException e) {
				throw new DMLRuntimeException("Static SSL setup failed for client side", e);
			}
		}
	}
}

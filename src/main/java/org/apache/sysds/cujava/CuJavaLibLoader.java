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

package org.apache.sysds.cujava;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.*;
public class CuJavaLibLoader {

	private static final String LIB_BASE = "cujava";  // -> libcujava.so / cujava.dll / libcujava.dylib
	private static volatile boolean loaded = false;   // fast-path guard

	/** Public entry – call from static blocks in binding classes. */
	public static synchronized void load() {
		if (loaded)                                   // already done
			return;

		// 1) Standard lookup (java.library.path or OS default locations)
		try {
			System.loadLibrary(LIB_BASE);
			loaded = true;
			return;
		}
		catch (UnsatisfiedLinkError ignored) {
			// Fall through to JAR extraction
		}

		// 2) Extract the library from the JAR (/lib/...) to a temp file
		String fileName = System.mapLibraryName(LIB_BASE);   // platform-specific
		String resource = "/lib/" + fileName;                // matches <targetPath>lib in the POM

		try (InputStream in = CuJavaLibLoader.class.getResourceAsStream(resource)) {
			if (in == null)
				throw new UnsatisfiedLinkError(
					"Native library not found inside JAR at " + resource);

			Path tmp = Files.createTempFile("cujava_", fileName);
			tmp.toFile().deleteOnExit();
			Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);

			System.load(tmp.toAbsolutePath().toString());
			loaded = true;
		}
		catch (IOException | UnsatisfiedLinkError e) {
			throw (UnsatisfiedLinkError)
				new UnsatisfiedLinkError("Failed to load native CUDA bridge: " + e).initCause(e);
		}
	}

	private CuJavaLibLoader() { /* no instances */ }

}

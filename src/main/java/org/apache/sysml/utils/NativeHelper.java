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

package org.apache.sysml.utils;

import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.HashMap;
import java.util.Vector;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.File;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.SystemUtils;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;

/**
 * This class helps in loading native library.
 * By default, it first tries to load Intel MKL, else tries to load OpenBLAS.
 */
public class NativeHelper {
	private static boolean isSystemMLLoaded = false;
	private static final Log LOG = LogFactory.getLog(NativeHelper.class.getName());
	private static HashMap<String, String> supportedArchitectures = new HashMap<String, String>();
	public static String blasType;
	private static int maxNumThreads = -1;
	private static boolean setMaxNumThreads = false;
	static {
		// Note: we only support 64 bit Java on x86, PowerPC and AMD machine
		supportedArchitectures.put("x86_64", "x86_64");
		supportedArchitectures.put("amd64", "x86_64");
		supportedArchitectures.put("powerpc", "ppc64le");
		supportedArchitectures.put("ppc64le", "ppc64le");
	}

	private static boolean attemptedLoading = false;

	private static String hintOnFailures = "";

	// Performing loading in a method instead of a static block will throw a detailed stack trace in case of fatal errors
	private static void init() {
		// Only Linux and Windows supported for BLAS
		if(!(SystemUtils.IS_OS_LINUX || SystemUtils.IS_OS_WINDOWS))
			return;

		// attemptedLoading variable ensures that we don't try to load SystemML and other dependencies 
		// again and again especially in the parfor (hence the double-checking with synchronized).
		if(!attemptedLoading) {
			DMLConfig dmlConfig = ConfigurationManager.getDMLConfig();
			// -------------------------------------------------------------------------------------
			// We allow BLAS to be enabled or disabled or explicitly selected in one of the two ways:
			// 1. DML Configuration: native.blas (boolean flag)
			// 2. Environment variable: SYSTEMML_BLAS (can be set to mkl, openblas or none)
			// The option 1 will be removed in later SystemML versions.
			// The option 2 is useful for two reasons:
			// - Developer testing of different BLAS 
			// - Provides fine-grained control. Certain machines could use mkl while others use openblas, etc.
			String userSpecifiedBLAS = (dmlConfig == null) ? "auto" : dmlConfig.getTextValue(DMLConfig.NATIVE_BLAS).trim().toLowerCase();

			if(userSpecifiedBLAS.equals("auto") || userSpecifiedBLAS.equals("mkl") || userSpecifiedBLAS.equals("openblas")) {
				long start = System.nanoTime();
				if(!supportedArchitectures.containsKey(SystemUtils.OS_ARCH)) {
					LOG.info("Unsupported architecture for native BLAS:" + SystemUtils.OS_ARCH);
					return;
				}
				synchronized(NativeHelper.class) {
					if(!attemptedLoading) {
						// -----------------------------------------------------------------------------
						// =============================================================================
						// By default, we will native.blas=true and we will attempt to load MKL first.
						// If MKL is not enabled then we try to load OpenBLAS.
						// If both MKL and OpenBLAS are not available we fall back to Java BLAS.
						if(userSpecifiedBLAS.equals("auto")) {
							blasType = isMKLAvailable() ? "mkl" : isOpenBLASAvailable() ? "openblas" : null;
							if(blasType == null)
								LOG.info("Unable to load either MKL or OpenBLAS due to " + hintOnFailures);
						}
						else if(userSpecifiedBLAS.equals("mkl")) {
							blasType = isMKLAvailable() ? "mkl" : null;
							if(blasType == null)
								LOG.info("Unable to load MKL due to " + hintOnFailures);
						}
						else if(userSpecifiedBLAS.equals("openblas")) {
							blasType = isOpenBLASAvailable() ? "openblas" : null;
							if(blasType == null)
								LOG.info("Unable to load OpenBLAS due to " + hintOnFailures);
						}
						else {
							// Only thrown at development time.
							throw new RuntimeException("Unsupported BLAS:" + userSpecifiedBLAS);
						}
						// =============================================================================
						if(blasType != null && loadLibraryHelper("systemml_" + blasType)) {
							String blasPathAndHint = "";
							// ------------------------------------------------------------
							// This logic gets the list of native libraries that are loaded
							if(LOG.isDebugEnabled()) {
								// Only perform the checking of library paths when DEBUG is enabled to avoid runtime overhead. 
								try {
									java.lang.reflect.Field loadedLibraryNamesField = ClassLoader.class.getDeclaredField("loadedLibraryNames");
									loadedLibraryNamesField.setAccessible(true);
									@SuppressWarnings("unchecked")
									Vector<String> libraries = (Vector<String>) loadedLibraryNamesField.get(ClassLoader.getSystemClassLoader());
									LOG.debug("List of native libraries loaded:" + libraries);
									for(String library : libraries) {
										if(library.contains("libmkl_rt") || library.contains("libopenblas")) {
											blasPathAndHint = " from the path " + library;
											break;
										}
									}
								} catch (NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException e) {
									LOG.debug("Error while finding list of native libraries:" + e.getMessage());
								}
							}
							// ------------------------------------------------------------

							LOG.info("Using native blas: " + blasType + blasPathAndHint);
							isSystemMLLoaded = true;
						}
					}
				}
				double timeToLoadInMilliseconds = (System.nanoTime()-start)*1e-6;
				if(timeToLoadInMilliseconds > 1000) 
					LOG.warn("Time to load native blas: " + timeToLoadInMilliseconds + " milliseconds.");
			}
			else {
				LOG.debug("Using internal Java BLAS as native BLAS support the configuration 'native.blas'=" + userSpecifiedBLAS + ".");
			}
			attemptedLoading = true;
		}
	}

	public static boolean isNativeLibraryLoaded() {
		init();
		if(maxNumThreads == -1)
			maxNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(isSystemMLLoaded && !setMaxNumThreads && maxNumThreads != -1) {
			// This method helps us decide whether to use GetPrimitiveArrayCritical or GetDoubleArrayElements in JNI as each has different tradeoffs.
			// In current implementation, we always use GetPrimitiveArrayCritical as it has proven to be fastest. 
			// We can revisit this decision later and hence I would not recommend removing this method. 
			setMaxNumThreads(maxNumThreads);
			setMaxNumThreads = true;
		}
		return isSystemMLLoaded;
	}

	public static int getMaxNumThreads() {
		if(maxNumThreads == -1)
			maxNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		return maxNumThreads;
	}


	private static boolean isMKLAvailable() {
		return loadBLAS("mkl_rt", null);
	}

	private static boolean isOpenBLASAvailable() {
		if(!loadBLAS("gomp", "gomp required for loading OpenBLAS-enabled SystemML library")) 
			return false;
		return loadBLAS("openblas", null);
	}

	private static boolean loadBLAS(String blas, String optionalMsg) {
		try {
			System.loadLibrary(blas);
			return true;
		}
		catch (UnsatisfiedLinkError e) {
			if(!hintOnFailures.contains(blas))
				hintOnFailures = hintOnFailures + blas + " ";
			if(optionalMsg != null)
				LOG.debug("Unable to load " + blas + "(" + optionalMsg + "):" + e.getMessage());
			else
				LOG.debug("Unable to load " + blas + ":" + e.getMessage());
			return false;
		}
	}
	
	/**
	 * Get file path for a given library name 
	 * @param libName can be systemml_mkl or systemml_openblas
	 * @return file path
	 */
	private static String getPath(String libName) {
		String prefix = "";
		String suffix = "";
		String os = "";
		if (SystemUtils.IS_OS_MAC_OSX) {
			prefix = "lib";
			suffix = "dylib";
			os = "Darwin";
		} else if (SystemUtils.IS_OS_LINUX) {
			prefix = "lib";
			suffix = "so";
			os = "Linux";
		} else if (SystemUtils.IS_OS_WINDOWS) {
			prefix = "";
			suffix = "dll";
			os = "Windows";
		} else {
			LOG.info("Unsupported OS:" + SystemUtils.OS_NAME);
			throw new RuntimeException("Unsupported OS");
		}
		
		String arch = supportedArchitectures.get(SystemUtils.OS_ARCH);
		if(arch == null) {
			LOG.info("Unsupported architecture:" + SystemUtils.OS_ARCH);
			throw new RuntimeException("Unsupported architecture:" + SystemUtils.OS_ARCH);
		}
		return prefix + libName + "-" + os + "-" + arch + "." + suffix;
	}

	/**
	 * Loads the specified library.
	 * 
	 * @param libName can be systemml_mkl or systemml_openblas
	 * @return true if the library is loaded successfully
	 */
	private static boolean loadLibraryHelper(String libName)  {
		String path = getPath(libName);
		InputStream in = null; OutputStream out = null;
		try {
			// This logic is added because Java doesnot allow to load library from a resource file.
			in = NativeHelper.class.getResourceAsStream("/lib/"+path);
			if(in != null) {
				File temp = File.createTempFile(path, "");
				temp.deleteOnExit();
				out = FileUtils.openOutputStream(temp);
				IOUtils.copy(in, out);
				in.close(); in = null;
				out.close(); out = null;
				System.load(temp.getAbsolutePath());
				return true;
			}
			else
				LOG.warn("No lib available in the jar:" + path);
		} catch(IOException e) {
			LOG.warn("Unable to load library " + path + " from resource:" + e.getMessage());
		} finally {
			if(out != null)
				try {
					out.close();
				} catch (IOException e) {}
			if(in != null)
				try {
					in.close();
				} catch (IOException e) {}
		}
		return false;
	}

	// TODO: Add pmm, wsloss, mmchain, etc.
	public static native boolean matrixMultDenseDense(double [] m1, double [] m2, double [] ret, int m1rlen, int m1clen, int m2clen, int numThreads);
	private static native boolean tsmm(double [] m1, double [] ret, int m1rlen, int m1clen, boolean isLeftTranspose, int numThreads);

	// ----------------------------------------------------------------------------------------------------------------
	// LibMatrixDNN operations:
	// N = number of images, C = number of channels, H = image height, W = image width
	// K = number of filters, R = filter height, S = filter width
	// TODO: case not handled: sparse filters (which will only be executed in Java). Since filters are relatively smaller, this is a low priority.

	// Returns -1 if failures or returns number of nonzeros
	// Called by ConvolutionCPInstruction if both input and filter are dense
	public static native int conv2dDense(double [] input, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	public static native int conv2dBiasAddDense(double [] input, double [] bias, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	// Called by ConvolutionCPInstruction if both input and filter are dense
	public static native int conv2dBackwardFilterDense(double [] input, double [] dout, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	// If both filter and dout are dense, then called by ConvolutionCPInstruction
	// Else, called by LibMatrixDNN's thread if filter is dense. dout[n] is converted to dense if sparse.
	public static native int conv2dBackwardDataDense(double [] filter, double [] dout, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);

	// Currently only supported with numThreads = 1 and sparse input
	// Called by LibMatrixDNN's thread if input is sparse. dout[n] is converted to dense if sparse.
	public static native boolean conv2dBackwardFilterSparseDense(int apos, int alen, int[] aix, double[] avals, double [] rotatedDoutPtr, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	// Called by LibMatrixDNN's thread if input is sparse and filter is dense
	public static native boolean conv2dSparse(int apos, int alen, int[] aix, double[] avals, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	// ----------------------------------------------------------------------------------------------------------------

	// This method helps us decide whether to use GetPrimitiveArrayCritical or GetDoubleArrayElements in JNI as each has different tradeoffs.
	// In current implementation, we always use GetPrimitiveArrayCritical as it has proven to be fastest. 
	// We can revisit this decision later and hence I would not recommend removing this method. 
	private static native void setMaxNumThreads(int numThreads);
}

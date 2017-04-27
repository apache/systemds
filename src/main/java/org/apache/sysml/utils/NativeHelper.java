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
		// Note: we only support 64 bit Java on x86 and AMD machine
    supportedArchitectures.put("x86_64", "x86_64");
    supportedArchitectures.put("amd64", "x86_64");
	}
	
	private static boolean attemptedLoading = false;
	// This will throw a detailed stack trace in case of fatal errors
	private static void init() {
		if(!attemptedLoading) {
			DMLConfig dmlConfig = ConfigurationManager.getDMLConfig();
			if(dmlConfig != null && dmlConfig.getBooleanValue(DMLConfig.NATIVE_BLAS)) {
		    if(supportedArchitectures.containsKey(SystemUtils.OS_ARCH)) {
		    	synchronized(NativeHelper.class) {
		    		if(!attemptedLoading) {
		    			String userSpecifiedBLAS = System.getenv("SYSTEMML_BLAS");
		    			if(userSpecifiedBLAS != null) {
		    				LOG.info("SYSTEMML_BLAS is set to " + userSpecifiedBLAS);
		    				if(userSpecifiedBLAS.trim().equals("")) {
			    				// Check if MKL is enabled, if not then check if openblas is enabled
							    blasType = isMKLAvailable() ? "mkl" : (isOpenBLASAvailable() ? "openblas" : null);
			    			}
			    			else if(userSpecifiedBLAS.equalsIgnoreCase("mkl")) {
			    				blasType = isMKLAvailable() ? "mkl" : null;
			    			}
			    			else if(userSpecifiedBLAS.equalsIgnoreCase("openblas")) {
			    				blasType = isOpenBLASAvailable() ? "openblas" : null;
			    			}
			    			else {
			    				LOG.warn("Unsupported BLAS:" + userSpecifiedBLAS);
			    			}
		    			}
		    			else {
		    				// Default behavior:
		    				// Check if MKL is enabled, if not then check if openblas is enabled
						    blasType = isMKLAvailable() ? "mkl" : (isOpenBLASAvailable() ? "openblas" : null);
		    			}
		    			
					    
							if(blasType != null) {
								try {
									loadLibrary("systemml", "_" + blasType);
									LOG.info("Using native blas: " + blasType);
									isSystemMLLoaded = true;
								} catch (IOException e) {
									LOG.warn("Using Java-based BLAS as unable to load native BLAS");
								}
							}
		    		}
		    	}
		    }
		    else {
		    	LOG.warn("Unsupported architecture for native BLAS:" + SystemUtils.OS_ARCH);
		    }
			}
			attemptedLoading = true;
		}
	}
	
	public static boolean isNativeLibraryLoaded() {
		init();
		if(maxNumThreads == -1)
			maxNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(isSystemMLLoaded && !setMaxNumThreads && maxNumThreads != -1) {
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
		// ------------------------------------------------------------
		// Set environment variable MKL_THREADING_LAYER to GNU on Linux for performance
		if(SystemUtils.IS_OS_LINUX || SystemUtils.IS_OS_MAC_OSX) {
			try {
				loadLibrary("preload_systemml", "");
			} catch (IOException e) {
				LOG.warn("Unable to load preload_systemml (required for loading MKL-enabled SystemML library):" + e.getMessage());
				return false;
			}
			// The most reliable way in my investigation to ensure that MKL runs smoothly with OpenMP (used by conv2d*)
			// is setting the environment variable MKL_THREADING_LAYER to GNU
			EnvironmentHelper.setEnv("MKL_THREADING_LAYER", "GNU");
			String gompLibrary = SystemUtils.IS_OS_LINUX ? "gomp" : "iomp5";
			if(!loadBLAS(gompLibrary, gompLibrary + " required for loading MKL-enabled SystemML library")) 
				return false;
		}
		// ------------------------------------------------------------
		return loadBLAS("mkl_rt");
	}
	
	private static boolean isOpenBLASAvailable() {
		// OpenBLAS is not supported on MacOSX
		// Default installation of openblas on MacOSX throws "Incompatible library versions" error.
		// Until that is fixed, SystemML will not support OpenBLAS on Mac OSX. This is documented in the troubleshooting guide.
		if(SystemUtils.IS_OS_MAC_OSX)
			return false;
		else if(SystemUtils.IS_OS_WINDOWS) 
			return loadBLAS("openblas") || loadBLAS("libopenblas"); 
		else 
			return loadBLAS("openblas");
	}
	
	private static boolean loadBLAS(String blas) {
		return loadBLAS(blas, null);
	}
	
	private static boolean loadBLAS(String blas, String optionalMsg) {
		try {
			 System.loadLibrary(blas);
			 return true;
		}
		catch (UnsatisfiedLinkError e) {
			if(optionalMsg != null)
				LOG.warn("Unable to load " + blas + "(" + optionalMsg + "):" + e.getMessage());
			else
				LOG.warn("Unable to load " + blas + ":" + e.getMessage());
			return false;
		}
	}
	
	private static void loadLibrary(String libName, String suffix1) throws IOException {
		String prefix = SystemUtils.IS_OS_WINDOWS ? "" : "lib";
		String suffix2 = "";
		String os = "";
		if (SystemUtils.IS_OS_MAC_OSX) {
			suffix2 = "dylib";
			os = "Darwin";
		} else if (SystemUtils.IS_OS_LINUX) {
			suffix2 = "so";
			os = "Linux";
		} else if (SystemUtils.IS_OS_WINDOWS) {
			suffix2 = "dll";
			os = "Windows";
		} else {
			LOG.warn("Unsupported OS for native BLAS:" + SystemUtils.OS_NAME);
			throw new IOException("Unsupported OS");
		}
		
		String arch = supportedArchitectures.get(SystemUtils.OS_ARCH);
		if(arch == null) {
			LOG.warn("Unsupported architecture for native BLAS:" + SystemUtils.OS_ARCH);
			throw new IOException("Unsupported architecture:" + SystemUtils.OS_ARCH);
		}
		loadLibraryHelper(prefix + libName + suffix1 + "-" + os + "-" + arch + "." + suffix2);
	}

	private static void loadLibraryHelper(String path) throws IOException {
		InputStream in = null; OutputStream out = null;
		try {
			in = NativeHelper.class.getResourceAsStream("/lib/"+path);
			if(in != null) {
				File temp = File.createTempFile(path, "");
				temp.deleteOnExit();
				out = FileUtils.openOutputStream(temp);
        IOUtils.copy(in, out);
        in.close(); in = null;
        out.close(); out = null;
				System.load(temp.getAbsolutePath());
			}
			else
				throw new IOException("No lib available in the jar:" + path);
			
		} catch(IOException e) {
			LOG.info("Unable to load library " + path + " from resource:" + e.getMessage());
			throw e;
		} finally {
			if(out != null)
				out.close();
			if(in != null)
				in.close();
		}
		
	}
	
	// TODO: Add pmm, wsloss, mmchain, etc.
	public static native boolean matrixMultDenseDense(double [] m1, double [] m2, double [] ret, int m1rlen, int m1clen, int m2clen, int numThreads);
	private static native boolean tsmm(double [] m1, double [] ret, int m1rlen, int m1clen, boolean isLeftTranspose, int numThreads);
	
	// LibMatrixDNN operations:
	// N = number of images, C = number of channels, H = image height, W = image width
	// K = number of filters, R = filter height, S = filter width
	public static native boolean conv2dDense(double [] input, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	public static native boolean conv2dBiasAddDense(double [] input, double [] bias, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	public static native boolean conv2dBackwardDataDense(double [] filter, double [] dout, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	public static native boolean conv2dBackwardFilterDense(double [] input, double [] dout, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	
	private static native void setMaxNumThreads(int numThreads);
}

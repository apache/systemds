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
		// Note: we only support 64 bit Java on x86 and AMD machine
    supportedArchitectures.put("x86_64", "x86_64");
    supportedArchitectures.put("amd64", "x86_64");
	}
	
	private static boolean attemptedLoading = false;
	
	// Performing loading in a method instead of a static block will throw a detailed stack trace in case of fatal errors
	private static void init() {
		// Only Linux supported for BLAS
		if(!SystemUtils.IS_OS_LINUX)
			return;
		
		// attemptedLoading variable ensures that we don't try to load SystemML and other dependencies 
		// again and again especially in the parfor (hence the double-checking with synchronized).
		if(!attemptedLoading) {
			DMLConfig dmlConfig = ConfigurationManager.getDMLConfig();
			String userSpecifiedBLAS = System.getenv("SYSTEMML_BLAS");
			userSpecifiedBLAS = (userSpecifiedBLAS == null) ? "" :  userSpecifiedBLAS.trim().toLowerCase();
			// -------------------------------------------------------------------------------------
			// We allow BLAS to be enabled or disabled or explicitly selected in one of the two ways:
			// 1. DML Configuration: native.blas (boolean flag)
			// 2. Environment variable: SYSTEMML_BLAS (can be set to mkl, openblas or none)
			// The option 1 will be removed in later SystemML versions.
			// The option 2 is useful for two reasons:
			// - Developer testing of different BLAS 
			// - Provides fine-grained control. Certain machines could use mkl while others use openblas, etc.
			boolean enabledViaConfig = (dmlConfig == null) ? true : dmlConfig.getBooleanValue(DMLConfig.NATIVE_BLAS);
			boolean enabledViaEnvironmentVariable = userSpecifiedBLAS.equals("") || userSpecifiedBLAS.equals("mkl") || userSpecifiedBLAS.equals("openblas");
			
			if(enabledViaConfig && enabledViaEnvironmentVariable) {
				long start = System.nanoTime();
				if(!supportedArchitectures.containsKey(SystemUtils.OS_ARCH)) {
					LOG.warn("Unsupported architecture for native BLAS:" + SystemUtils.OS_ARCH);
					return;
				}
	    	synchronized(NativeHelper.class) {
	    		if(!attemptedLoading) {
	    			// -----------------------------------------------------------------------------
	    			// =============================================================================
	    			// By default, we will native.blas=true and we will attempt to load MKL first.
    				// If MKL is not enabled then we try to load OpenBLAS.
    				// If both MKL and OpenBLAS are not available we fall back to Java BLAS.
	    			if(userSpecifiedBLAS.equalsIgnoreCase("")) {
	    				blasType = isMKLAvailable() ? "mkl" : isOpenBLASAvailable() ? "openblas" : null;
	    				if(blasType == null)
	    					LOG.info("Unable to load either MKL or OpenBLAS. Please set ");
	    			}
	    			else if(userSpecifiedBLAS.equalsIgnoreCase("mkl")) {
	    				blasType = isMKLAvailable() ? "mkl" : null;
	    				if(blasType == null)
	    					LOG.info("Unable to load MKL");
	    			}
	    			else if(userSpecifiedBLAS.equalsIgnoreCase("openblas")) {
	    				blasType = isOpenBLASAvailable() ? "openblas" : null;
	    				if(blasType == null)
	    					LOG.info("Unable to load OpenBLAS");
	    			}
	    			else {
	    				LOG.info("Unsupported BLAS:" + userSpecifiedBLAS);
	    			}
	    			// =============================================================================
				    if(blasType != null && loadLibraryHelper("libsystemml_" + blasType + "-Linux-x86_64.so")) {
				    	String blasPathAndHint = "";
				    	// ------------------------------------------------------------
				    	// This logic gets the list of native libraries that are loaded
				    	try {
				    		java.lang.reflect.Field loadedLibraryNamesField = ClassLoader.class.getDeclaredField("loadedLibraryNames");
								loadedLibraryNamesField.setAccessible(true);
								@SuppressWarnings("unchecked")
								Vector<String> libraries = (Vector<String>) loadedLibraryNamesField.get(ClassLoader.getSystemClassLoader());
								LOG.debug("List of native libraries loaded:" + libraries);
								for(String library : libraries) {
									if(library.endsWith("libmkl_rt.so"))
										blasPathAndHint = " from the path " + library;
									else if(library.endsWith("libopenblas.so")) {
										blasPathAndHint = " from the path " + library + ". Hint: Please make sure that the libopenblas.so is built with GNU OpenMP threading (ldd " + library + " | grep libgomp).";
									}
								}
							} catch (NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException e) {
								LOG.debug("Error while finding list of native libraries:" + e.getMessage());
							}
				    	// ------------------------------------------------------------
				    	
							LOG.info("Using native blas: " + blasType + blasPathAndHint);
							isSystemMLLoaded = true;
						}
	    		}
	    	}
	    	double timeToLoadInMilliseconds = (System.nanoTime()-start)*1e-6;
	    	if(timeToLoadInMilliseconds > 100) 
	    		LOG.warn("Time to load native blas: " + timeToLoadInMilliseconds + " milliseconds.");
			}
			else {
				if(enabledViaConfig)
					LOG.warn("Using internal Java BLAS as native BLAS support is disabled by the configuration 'native.blas'.");
				else
					LOG.warn("Using internal Java BLAS as native BLAS support is disabled by the environment variable 'SYSTEMML_BLAS=" + userSpecifiedBLAS + "'.");
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
		// ------------------------------------------------------------
		// Set environment variable MKL_THREADING_LAYER to GNU on Linux for performance
		if(!loadLibraryHelper("libpreload_systemml-Linux-x86_64.so")) {
			LOG.debug("Unable to load preload_systemml (required for loading MKL-enabled SystemML library)");
			return false;
		}
		// The most reliable way in my investigation to ensure that MKL runs smoothly with OpenMP (used by conv2d*)
		// is setting the environment variable MKL_THREADING_LAYER to GNU
		EnvironmentHelper.setEnv("MKL_THREADING_LAYER", "GNU");
		if(!loadBLAS("gomp", "gomp required for loading MKL-enabled SystemML library")) 
			return false;
		
		// ------------------------------------------------------------
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
			if(optionalMsg != null)
				LOG.debug("Unable to load " + blas + "(" + optionalMsg + "):" + e.getMessage());
			else
				LOG.debug("Unable to load " + blas + ":" + e.getMessage());
			return false;
		}
	}

	private static boolean loadLibraryHelper(String path)  {
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
	
	// Called by ConvolutionCPInstruction if both input and filter are dense
	public static native boolean conv2dDense(double [] input, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	public static native boolean conv2dBiasAddDense(double [] input, double [] bias, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	// Called by ConvolutionCPInstruction if both input and filter are dense
	public static native boolean conv2dBackwardFilterDense(double [] input, double [] dout, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	// If both filter and dout are dense, then called by ConvolutionCPInstruction
	// Else, called by LibMatrixDNN's thread if filter is dense. dout[n] is converted to dense if sparse.
	public static native boolean conv2dBackwardDataDense(double [] filter, double [] dout, double [] ret, int N, int C, int H, int W, 
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

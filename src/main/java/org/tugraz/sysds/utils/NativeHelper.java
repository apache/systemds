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

package org.tugraz.sysds.utils;

import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.conf.DMLConfig;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;

import java.util.Vector;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.io.File;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.SystemUtils;

/**
 * This class helps in loading native library.
 * By default, it first tries to load Intel MKL, else tries to load OpenBLAS.
 */
public class NativeHelper {
	
	public static enum NativeBlasState {
		NOT_ATTEMPTED_LOADING_NATIVE_BLAS,
		SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_IN_USE,
		SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_NOT_IN_USE,
		ATTEMPTED_LOADING_NATIVE_BLAS_UNSUCCESSFULLY
	}
	
	public static NativeBlasState CURRENT_NATIVE_BLAS_STATE = NativeBlasState.NOT_ATTEMPTED_LOADING_NATIVE_BLAS;
	private static String blasType;
	private static final Log LOG = LogFactory.getLog(NativeHelper.class.getName());
	
	// Useful for deciding whether to use native BLAS in parfor environment.
	private static int maxNumThreads = -1;
	private static boolean setMaxNumThreads = false;
	
	/**
	 * Called by Statistics to print the loaded BLAS.
	 * 
	 * @return empty string or the BLAS that is loaded
	 */
	public static String getCurrentBLAS() {
		return blasType != null ? blasType : "";
	}
	
	/**
	 * Called by runtime to check if the BLAS is available for exploitation
	 * 
	 * @return true if CURRENT_NATIVE_BLAS_STATE is SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_NOT_IN_USE else false
	 */
	public static boolean isNativeLibraryLoaded() {
		if(!isBLASLoaded()) {
			DMLConfig dmlConfig = ConfigurationManager.getDMLConfig();
			String userSpecifiedBLAS = (dmlConfig == null) ? "auto" : dmlConfig.getTextValue(DMLConfig.NATIVE_BLAS).trim().toLowerCase();
			String customLibPath = (dmlConfig == null) ? "none" : dmlConfig.getTextValue(DMLConfig.NATIVE_BLAS_DIR).trim();
			performLoading(customLibPath, userSpecifiedBLAS);
		}
		if(maxNumThreads == -1)
			maxNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(CURRENT_NATIVE_BLAS_STATE == NativeBlasState.SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_IN_USE && !setMaxNumThreads && maxNumThreads != -1) {
			// This method helps us decide whether to use GetPrimitiveArrayCritical or GetDoubleArrayElements in JNI as each has different tradeoffs.
			// In current implementation, we always use GetPrimitiveArrayCritical as it has proven to be fastest. 
			// We can revisit this decision later and hence I would not recommend removing this method. 
			setMaxNumThreads(maxNumThreads);
			setMaxNumThreads = true;
		}
		return CURRENT_NATIVE_BLAS_STATE == NativeBlasState.SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_IN_USE;
	}
	
	/**
	 * Initialize the native library before executing the DML program 
	 * 
	 * @param customLibPath specified by sysds.native.blas.directory
	 * @param userSpecifiedBLAS specified by sysds.native.blas
	 */
	public static void initialize(String customLibPath, String userSpecifiedBLAS) {
		if(isBLASLoaded() && isSupportedBLAS(userSpecifiedBLAS) && !blasType.equalsIgnoreCase(userSpecifiedBLAS)) {
			throw new DMLRuntimeException("Cannot replace previously loaded blas \"" + blasType + "\" with \"" + userSpecifiedBLAS + "\".");
		}
		else if(isBLASLoaded() && userSpecifiedBLAS.equalsIgnoreCase("none")) {
			CURRENT_NATIVE_BLAS_STATE = NativeBlasState.SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_NOT_IN_USE;
		}
		else if(isBLASLoaded() && userSpecifiedBLAS.equalsIgnoreCase(blasType)) {
			CURRENT_NATIVE_BLAS_STATE = NativeBlasState.SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_IN_USE;
		}
		else if(!isBLASLoaded() && isSupportedBLAS(userSpecifiedBLAS)) {
			performLoading(customLibPath, userSpecifiedBLAS);
		}
	}
	
	/**
	 * Return true if the given BLAS type is supported.
	 * 
	 * @param userSpecifiedBLAS BLAS type specified via sysds.native.blas property
	 * @return true if the userSpecifiedBLAS is auto | mkl | openblas, else false
	 */
	private static boolean isSupportedBLAS(String userSpecifiedBLAS) {
		return userSpecifiedBLAS.equalsIgnoreCase("auto") || 
				userSpecifiedBLAS.equalsIgnoreCase("mkl") || 
				userSpecifiedBLAS.equalsIgnoreCase("openblas");
	}
	
	/**
	 * Note: we only support 64 bit Java on x86 and AMD machine
	 * 
	 * @return true if the hardware architecture is supported
	 */
	private static boolean isSupportedArchitecture() {
		if(SystemUtils.OS_ARCH.equals("x86_64") || SystemUtils.OS_ARCH.equals("amd64")) {
			return true;
		}
		LOG.info("Unsupported architecture for native BLAS:" + SystemUtils.OS_ARCH);
		return false;
	}
	
	/**
	 * Check if native BLAS libraries have been successfully loaded
	 * @return true if CURRENT_NATIVE_BLAS_STATE is SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_IN_USE or SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_NOT_IN_USE
	 */
	private static boolean isBLASLoaded() {
		return CURRENT_NATIVE_BLAS_STATE == NativeBlasState.SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_IN_USE || 
				CURRENT_NATIVE_BLAS_STATE == NativeBlasState.SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_NOT_IN_USE;
	}
	
	/**
	 * Check if we should attempt to perform loading.
	 * If custom library path is provided, we should attempt to load again if not already loaded.
	 * 
	 * @param customLibPath custom library path 
	 * @return true if we should attempt to load blas again
	 */
	private static boolean shouldReload(String customLibPath) {
		boolean isValidBLASDirectory = customLibPath != null && !customLibPath.equalsIgnoreCase("none");
		return CURRENT_NATIVE_BLAS_STATE == NativeBlasState.NOT_ATTEMPTED_LOADING_NATIVE_BLAS ||
			   (isValidBLASDirectory && !isBLASLoaded());
	}

	// Performing loading in a method instead of a static block will throw a detailed stack trace in case of fatal errors
	private static void performLoading(String customLibPath, String userSpecifiedBLAS) {
		// Only Linux supported for BLAS
//		if(!SystemUtils.IS_OS_LINUX)
//			return;

		// attemptedLoading variable ensures that we don't try to load SystemDS and other dependencies
		// again and again especially in the parfor (hence the double-checking with synchronized).
		if(shouldReload(customLibPath) && isSupportedBLAS(userSpecifiedBLAS) && isSupportedArchitecture()) {
			long start = System.nanoTime();
			synchronized(NativeHelper.class) {
				if(shouldReload(customLibPath)) {
					// Set attempted loading unsuccessful in case of exception
					CURRENT_NATIVE_BLAS_STATE = NativeBlasState.ATTEMPTED_LOADING_NATIVE_BLAS_UNSUCCESSFULLY;
					String [] blas = new String[] { userSpecifiedBLAS }; 
					if(userSpecifiedBLAS.equalsIgnoreCase("auto")) {
						blas = new String[] { "mkl", "openblas" };
					}


					if(SystemUtils.IS_OS_WINDOWS) {
						if (checkAndLoadBLAS(customLibPath + "\\lib", blas) &&
//								loadLibraryHelper(customLibPath + "\\bin\\systemds_" + blasType + "-Windows-AMD64.dll")) {
//								loadLibraryHelper("systemds_" + blasType + "-Windows-AMD64.lib")) {
//																loadLibraryHelper(customLibPath + "\\systemds_" + blasType + "-Windows-AMD64.dll")) {
								loadBLAS(customLibPath, "systemds_" + blasType + "-Windows-AMD64", ""))
						{
							LOG.info("Using native blas: " + blasType + getNativeBLASPath());
							CURRENT_NATIVE_BLAS_STATE = NativeBlasState.SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_IN_USE;
						}
					}
					else {
						if (checkAndLoadBLAS(customLibPath, blas) && loadLibraryHelper("libsystemds_" + blasType + "-Linux-x86_64.so")) {
							LOG.info("Using native blas: " + blasType + getNativeBLASPath());
							CURRENT_NATIVE_BLAS_STATE = NativeBlasState.SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_IN_USE;
						}
					}
				}
			}
			double timeToLoadInMilliseconds = (System.nanoTime()-start)*1e-6;
			if(timeToLoadInMilliseconds > 1000) 
				LOG.warn("Time to load native blas: " + timeToLoadInMilliseconds + " milliseconds.");
		}
		else if(LOG.isDebugEnabled() && !isSupportedBLAS(userSpecifiedBLAS)) {
			LOG.debug("Using internal Java BLAS as native BLAS support the configuration 'sysds.native.blas'=" + userSpecifiedBLAS + ".");
		}
	}
	
	private static boolean checkAndLoadBLAS(String customLibPath, String [] listBLAS) {
		if(customLibPath != null && customLibPath.equalsIgnoreCase("none"))
			customLibPath = null;
		
		boolean isLoaded = false;
		for(int i = 0; i < listBLAS.length; i++) {
			String blas = listBLAS[i];
			if(blas.equalsIgnoreCase("mkl")) {
				if(SystemUtils.IS_OS_WINDOWS)
					isLoaded = true;
//					isLoaded = loadBLAS(customLibPath, "mkl_rt.dll", null);
				else
					isLoaded = loadBLAS(customLibPath, "mkl_rt", null);
			}
			else if(blas.equalsIgnoreCase("openblas")) {
				boolean isGompLoaded = loadBLAS(customLibPath, "gomp", "gomp required for loading OpenBLAS-enabled SystemDS library");
				if(isGompLoaded) {
					isLoaded = loadBLAS(customLibPath, "openblas", null);
				}
			}
			if(isLoaded) {
				blasType = blas;
				break;
			}
		}
		return isLoaded;
	}
	
	/**
	 * Useful method for debugging.
	 * 
	 * @return empty string (if !LOG.isDebugEnabled()) or the path from where openblas or mkl is loaded.
	 */
	private static String getNativeBLASPath() {
		String blasPathAndHint = "";
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
		return blasPathAndHint;
	}

	public static int getMaxNumThreads() {
		if(maxNumThreads == -1)
			maxNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		return maxNumThreads;
	}

	/**
	 * Attempts to load native BLAS
	 * 
	 * @param customLibPath can be null (if we want to only want to use LD_LIBRARY_PATH), else the 
	 * @param blas can be gomp, openblas or mkl_rt
	 * @param optionalMsg message for debugging
	 * @return true if successfully loaded BLAS
	 */
	private static boolean loadBLAS(String customLibPath, String blas, String optionalMsg) {
		// First attempt to load from custom library path
		if(customLibPath != null) {
			String libPath = customLibPath + File.separator + System.mapLibraryName(blas);
			try {
				System.load(libPath);
				// Print to stdout as this feature is intended for cloud environment
				System.out.println("Loaded the library:" + libPath);
				return true;
			}
			catch (UnsatisfiedLinkError e1) { 
				// Print to stdout as this feature is intended for cloud environment
				System.out.println("Unable to load " + libPath + ":" + e1.getMessage());
			}
		}

		// Then try loading using loadLibrary
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
		OutputStream out = null;
		try(InputStream in = NativeHelper.class.getResourceAsStream("/lib/"+path))
		{
			// This logic is added because Java does not allow to load library from a resource file.
			if(in != null)
			{
				File temp = File.createTempFile(path, "");
				temp.deleteOnExit();
				out = FileUtils.openOutputStream(temp);
				IOUtils.copy(in, out);
				System.load(temp.getAbsolutePath());
				return true;
			}
			else
				LOG.warn("No lib available in the jar:" + path);
		}
		catch(IOException e) {
			LOG.warn("Unable to load library " + path + " from resource:" + e.getMessage());
		}
		finally {
			IOUtilFunctions.closeSilently(out);
		}
		return false;
	}

	// TODO: Add pmm, wsloss, mmchain, etc.
	
	//double-precision matrix multiply dense-dense
	public static native boolean dmmdd(double [] m1, double [] m2, double [] ret, int m1rlen, int m1clen, int m2clen, int numThreads);
	//single-precision matrix multiply dense-dense
	public static native boolean smmdd(FloatBuffer m1, FloatBuffer m2, FloatBuffer ret, int m1rlen, int m1clen, int m2clen, int numThreads);
	//transpose-self matrix multiply
	public static native boolean tsmm(double[] m1, double[] ret, int m1rlen, int m1clen, boolean leftTrans, int numThreads);

	// ----------------------------------------------------------------------------------------------------------------
	// LibMatrixDNN operations:
	// N = number of images, C = number of channels, H = image height, W = image width
	// K = number of filters, R = filter height, S = filter width
	// TODO: case not handled: sparse filters (which will only be executed in Java). Since filters are relatively smaller, this is a low priority.

	// Returns -1 if failures or returns number of nonzeros
	// Called by DnnCPInstruction if both input and filter are dense
	public static native int conv2dDense(double [] input, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	public static native int dconv2dBiasAddDense(double [] input, double [] bias, double [] filter, double [] ret, int N,
		int C, int H, int W, int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	public static native int sconv2dBiasAddDense(FloatBuffer input, FloatBuffer bias, FloatBuffer filter, FloatBuffer ret,
		int N, int C, int H, int W, int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	// Called by DnnCPInstruction if both input and filter are dense
	public static native int conv2dBackwardFilterDense(double [] input, double [] dout, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	// If both filter and dout are dense, then called by DnnCPInstruction
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

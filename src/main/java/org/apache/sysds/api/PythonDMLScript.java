/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.	See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.	 See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.api;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.api.jmlc.Connection;
import org.apache.sysds.common.Types.ValueType;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UnixPipeUtils;
import py4j.DefaultGatewayServerListener;
import py4j.GatewayServer;
import py4j.Py4JNetworkException;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;


public class PythonDMLScript {

	private static final Log LOG = LogFactory.getLog(PythonDMLScript.class.getName());
	final private Connection _connection;
	public static GatewayServer GwS;

	private static String fromPythonBase = "py2java";
	private static String toPythonBase = "java2py";
	public HashMap<Integer, BufferedInputStream> fromPython = null;
	public HashMap<Integer, BufferedOutputStream> toPython = null;
	public String baseDir;
	private static int BATCH_SIZE = 32*1024;

	/**
	 * Entry point for Python API.
	 * 
	 * @param args Command line arguments.
	 * @throws Exception Throws exceptions if there is issues in startup or while running.
	 */
	public static void main(String[] args) throws Exception {
		final DMLOptions dmlOptions = DMLOptions.parseCLArguments(args);
		DMLScript.loadConfiguration(dmlOptions.configFile);
		GwS = new GatewayServer(new PythonDMLScript(), dmlOptions.pythonPort);
		GwS.addListener(new DMLGateWayListener());
		try {
			GwS.start();
		}
		catch(Py4JNetworkException p4e) {
			/**
			 * This sometimes happens when the startup is using a port already in use. In this case we handle it in python
			 * therefore use logging framework. and terminate program.
			 */
			LOG.info("failed startup", p4e);
			exitHandler.exit(-1);
		}
		catch(Exception e) {
			throw new DMLException("Failed startup and maintaining Python gateway", e);
		}
	}

	private PythonDMLScript() {
		// we enable multi-threaded I/O and operations for a single JMLC
		// connection because the calling Python process is unlikely to run
		// multi-threaded streams of operations on the same shared context
		_connection = new Connection();
	}

	public static void setDMLGateWayListenerLoggerLevel(Level l){
		Logger.getLogger(DMLGateWayListener.class).setLevel(l);
	}

	public Connection getConnection() {
		return _connection;
	}


	public void openPipes(String path, int num) throws IOException {
		fromPython = new HashMap<>(num * 2);
		toPython = new HashMap<>(num * 2);
		baseDir = path;
		for (int i = 0; i < num; i++) {
			BufferedInputStream pipe_in = UnixPipeUtils.openInput(path + "/" + fromPythonBase + "-" + i, i);
			LOG.debug("PY2JAVA pipe "+i+" is ready!");
			fromPython.put(i, pipe_in);

			BufferedOutputStream pipe_out = UnixPipeUtils.openOutput(path + "/" +  toPythonBase + "-" + i, i);
			toPython.put(i, pipe_out);
		}
	}

	public MatrixBlock startReadingMbFromPipe(int id, int rlen, int clen, ValueType type) throws IOException {
		long limit = (long) rlen * clen;
		LOG.debug("trying to read matrix from "+id+" with "+rlen+" rows and "+clen+" columns. Total size: "+limit);
		if(limit > Integer.MAX_VALUE)
			throw new DMLRuntimeException("Dense NumPy array of size " + limit +
					" cannot be converted to MatrixBlock");
		MatrixBlock mb;
		if(fromPython != null){
			BufferedInputStream pipe = fromPython.get(id);
			double[] denseBlock = new double[(int) limit];
			long nnz = UnixPipeUtils.readNumpyArrayInBatches(pipe, id, BATCH_SIZE, (int) limit, type, denseBlock, 0);
			mb = new MatrixBlock(rlen, clen, denseBlock);
			mb.setNonZeros(nnz);
		} else {
			throw new DMLRuntimeException("FIFO Pipes are not initialized.");
		}
		LOG.debug("Reading from Python finished");
		mb.examSparsity();
		return mb;
	}

	public MatrixBlock startReadingMbFromPipes(int[] blockSizes, int rlen, int clen, ValueType type) throws ExecutionException, InterruptedException {
		long limit = (long) rlen * clen;
		if(limit > Integer.MAX_VALUE)
			throw new DMLRuntimeException("Dense NumPy array of size " + limit +
					" cannot be converted to MatrixBlock");
		MatrixBlock mb = new MatrixBlock(rlen, clen, false, rlen*clen);
		if(fromPython != null){
			ExecutorService pool = CommonThreadPool.get();
			double[] denseBlock = new double[(int) limit];
			int offsetOut = 0;
			List<Future<Long>> futures = new ArrayList<>();
			for (int i = 0; i < blockSizes.length; i++) {
				BufferedInputStream pipe = fromPython.get(i);
				int id = i, blockSize = blockSizes[i], _offsetOut = offsetOut;
				Callable<Long> task = () -> {
					return UnixPipeUtils.readNumpyArrayInBatches(pipe, id, BATCH_SIZE, blockSize, type, denseBlock, _offsetOut);
				};

				futures.add(pool.submit(task));
				offsetOut += blockSize;
			}
			// Wait for all tasks and propagate exceptions, sum up nonzeros
			long nnz = 0;
			for (Future<Long> f : futures) {
				nnz += f.get();
			}

			mb = new MatrixBlock(rlen, clen, denseBlock);
			mb.setNonZeros(nnz);
		} else {
			throw new DMLRuntimeException("FIFO Pipes are not initialized.");
		}
		mb.examSparsity();
		return mb;
	}

	public void startWritingMbToPipe(int id, MatrixBlock mb) throws IOException {
		if (toPython != null) {
			int rlen = mb.getNumRows();
			int clen = mb.getNumColumns();
			int numElem = rlen * clen;
			LOG.debug("Trying to write matrix ["+baseDir + "-"+ id+"] with "+rlen+" rows and "+clen+" columns. Total size: "+numElem*8);

			BufferedOutputStream out = toPython.get(id);
			long bytes = UnixPipeUtils.writeNumpyArrayInBatches(out, id, BATCH_SIZE, numElem, ValueType.FP64, mb);

			LOG.debug("Writing of " + bytes +" Bytes to Python ["+baseDir + "-"+ id+"] finished");
		} else {
			throw new DMLRuntimeException("FIFO Pipes are not initialized.");
		}
	}

	public void startReadingColFromPipe(int id, FrameBlock fb, int rows, int totalBytes, int col, ValueType type, boolean any) throws IOException {
		if (fromPython == null) {
			throw new DMLRuntimeException("FIFO Pipes are not initialized.");
		}

		BufferedInputStream pipe = fromPython.get(id);
		LOG.debug("Start reading FrameBlock column from pipe #" + id + " with type " + type);

		// Delegate to UnixPipeUtils
		Array<?> arr = UnixPipeUtils.readFrameColumnFromPipe(pipe, id, rows, totalBytes, BATCH_SIZE, type);
		// Set column into FrameBlock
		fb.setColumn(col, arr);
		ValueType[] schema = fb.getSchema();
		// inplace update the schema for cases: int8 -> int32
		schema[col] = arr.getValueType();

		LOG.debug("Finished reading FrameBlock column from pipe #" + id);
	}

	public void startWritingColToPipe(int id, FrameBlock fb, int col) throws IOException {
		if (toPython == null) {
			throw new DMLRuntimeException("FIFO Pipes are not initialized.");
		}

		BufferedOutputStream pipe = toPython.get(id);
		ValueType type = fb.getSchema()[col];
		int rows = fb.getNumRows();
		Array<?> array = fb.getColumn(col);
		
		LOG.debug("Start writing FrameBlock column #" + col + " to pipe #" + id + " with type " + type + " and " + rows + " rows");

		// Delegate to UnixPipeUtils
		long bytes = UnixPipeUtils.writeFrameColumnToPipe(pipe, id, BATCH_SIZE, array, type);

		LOG.debug("Finished writing FrameBlock column #" + col + " to pipe #" + id + ". Total bytes: " + bytes);
	}

	public void closePipes() throws IOException {
		LOG.debug("Closing all pipes in Java");
		for (BufferedInputStream pipe : fromPython.values())
			pipe.close();
		for (BufferedOutputStream pipe : toPython.values())
			pipe.close();
		LOG.debug("Closed all pipes in Java");
	}

	@FunctionalInterface
	public interface ExitHandler {
		void exit(int status);
	}

	private static volatile ExitHandler exitHandler = System::exit;

	public static void setExitHandler(ExitHandler handler) {
		exitHandler = handler == null ? System::exit : handler;
	}

	public static void resetExitHandler() {
		exitHandler = System::exit;
	}
	protected static class DMLGateWayListener extends DefaultGatewayServerListener {
		private static final Log LOG = LogFactory.getLog(DMLGateWayListener.class.getName());

		@Override
		public void serverPostShutdown() {
			LOG.info("Shutdown done");
		}

		@Override
		public void serverPreShutdown() {
			LOG.info("Starting JVM shutdown");
		}

		@Override
		public void serverStarted() {
			LOG.info("GatewayServer started");
		}

		@Override
		public void serverStopped() {
			LOG.info("GatewayServer stopped");
		}
	}

}

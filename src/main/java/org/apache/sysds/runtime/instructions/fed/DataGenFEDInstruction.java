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

package org.apache.sysds.runtime.instructions.fed;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.lops.DataGen;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;

public class DataGenFEDInstruction extends UnaryFEDInstruction {
	private static final Log LOG = LogFactory.getLog(DataGenFEDInstruction.class.getName());
	private OpOpDG method;

	private final CPOperand rows, cols, dims;
	private final int blocksize;
	private boolean minMaxAreDoubles;
	private final String minValueStr, maxValueStr;
	private final double minValue, maxValue, sparsity;
	private final String pdf, pdfParams, frame_data, schema;
	private final long seed;
	private Long runtimeSeed;

	// sequence specific attributes
	private final CPOperand seq_from, seq_to, seq_incr;

	// sample specific attributes
	private final boolean replace;
	private final int numThreads;

	// seed positions
	private static final int SEED_POSITION_RAND = 8;
	private static final int SEED_POSITION_SAMPLE = 4;

	private DataGenFEDInstruction(Operator op, OpOpDG mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, String minValue, String maxValue, double sparsity, long seed,
		String probabilityDensityFunction, String pdfParams, int k, CPOperand seqFrom, CPOperand seqTo,
		CPOperand seqIncr, boolean replace, String data, String schema, String opcode, String istr) {
		super(FEDType.Rand, op, in, out, opcode, istr);
		this.method = mthd;
		this.rows = rows;
		this.cols = cols;
		this.dims = dims;
		this.blocksize = blen;
		this.minValueStr = minValue;
		this.maxValueStr = maxValue;
		double minDouble, maxDouble;
		try {
			minDouble = !minValue.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ? Double.valueOf(minValue) : -1;
			maxDouble = !maxValue.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ? Double.valueOf(maxValue) : -1;
			minMaxAreDoubles = true;
		}
		catch(NumberFormatException e) {
			// Non double values
			if(!minValueStr.equals(maxValueStr)) {
				throw new DMLRuntimeException(
					"Rand instruction does not support " + "non numeric Datatypes for range initializations.");
			}
			minDouble = -1;
			maxDouble = -1;
			minMaxAreDoubles = false;
		}
		this.minValue = minDouble;
		this.maxValue = maxDouble;
		this.sparsity = sparsity;
		this.seed = seed;
		this.pdf = probabilityDensityFunction;
		this.pdfParams = pdfParams;
		this.numThreads = k;
		this.seq_from = seqFrom;
		this.seq_to = seqTo;
		this.seq_incr = seqIncr;
		this.replace = replace;
		this.frame_data = data;
		this.schema = schema;
	}

	private DataGenFEDInstruction(Operator op, OpOpDG mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, String minValue, String maxValue, double sparsity, long seed,
		String probabilityDensityFunction, String pdfParams, int k, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, blen, minValue, maxValue, sparsity, seed, probabilityDensityFunction,
			pdfParams, k, null, null, null, false, null, null, opcode, istr);
	}

	private DataGenFEDInstruction(Operator op, OpOpDG mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, String maxValue, boolean replace, long seed, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, blen, "0", maxValue, 1.0, seed, null, null, 1, null, null, null,
			replace, null, null, opcode, istr);
	}

	private DataGenFEDInstruction(Operator op, OpOpDG mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, CPOperand seqFrom, CPOperand seqTo, CPOperand seqIncr, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, blen, "0", "1", 1.0, -1, null, null, 1, seqFrom, seqTo, seqIncr,
			false, null, null, opcode, istr);
	}

	private DataGenFEDInstruction(Operator op, OpOpDG mthd, CPOperand out, String opcode, String istr) {
		this(op, mthd, null, out, null, null, null, 0, "0", "0", 0, 0, null, null, 1, null, null, null, false, null,
			null, opcode, istr);
	}

	public DataGenFEDInstruction(Operator op, OpOpDG method, CPOperand out, CPOperand rows, CPOperand cols, String data,
		String schema, String opcode, String str) {
		this(op, method, null, out, rows, cols, null, 0, "0", "0", 0, 0, null, null, 1, null, null, null, false, data,
			schema, opcode, str);
	}

	public long getRows() {
		return rows.isLiteral() ? UtilFunctions.parseToLong(rows.getName()) : -1;
	}

	public long getCols() {
		return cols.isLiteral() ? UtilFunctions.parseToLong(cols.getName()) : -1;
	}

	public String getDims() {
		return dims.getName();
	}

	public int getBlocksize() {
		return blocksize;
	}

	public double getSparsity() {
		return sparsity;
	}

	public String getPdf() {
		return pdf;
	}

	public long getSeed() {
		return seed;
	}

	public long getFrom() {
		return seq_from.isLiteral() ? UtilFunctions.parseToLong(seq_from.getName()) : -1;
	}

	public long getTo() {
		return seq_to.isLiteral() ? UtilFunctions.parseToLong(seq_to.getName()) : -1;
	}

	public long getIncr() {
		return seq_incr.isLiteral() ? UtilFunctions.parseToLong(seq_incr.getName()) : -1;
	}

	public static DataGenFEDInstruction parseInstruction(String str) {
		OpOpDG method = null;
		String[] s = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = s[0];

		if(opcode.equalsIgnoreCase(DataGen.RAND_OPCODE)) {
			method = OpOpDG.RAND;
			InstructionUtils.checkNumFields(s, 10, 11);
		}
		else
			throw new DMLRuntimeException("DataGenFEDInstruction: only matrix rand(..) is supported.");

		CPOperand out = new CPOperand(s[s.length - 1]);
		Operator op = null;

		if(method == OpOpDG.RAND) {
			int missing; // number of missing params (row & cols or dims)
			CPOperand rows = null, cols = null, dims = null;
			if(s.length == 12) {
				missing = 1;
				rows = new CPOperand(s[1]);
				cols = new CPOperand(s[2]);
			}
			else {
				missing = 2;
				dims = new CPOperand(s[1]);
			}
			int blen = Integer.parseInt(s[4 - missing]);
			double sparsity = !s[7 - missing].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ? Double
				.parseDouble(s[7 - missing]) : -1;
			long seed = !s[SEED_POSITION_RAND - missing].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ? Long
				.parseLong(s[SEED_POSITION_RAND - missing]) : -1;
			String pdf = s[9 - missing];
			String pdfParams = !s[10 - missing].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ? s[10 - missing] : null;
			int k = Integer.parseInt(s[11 - missing]);

			return new DataGenFEDInstruction(op, method, null, out, rows, cols, dims, blen, s[5 - missing],
				s[6 - missing], sparsity, seed, pdf, pdfParams, k, opcode, str);
		}
		else
			throw new DMLRuntimeException("Unrecognized data generation method: " + method);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// process specific datagen operator
		if(method == OpOpDG.RAND)
			processRandInstruction(ec);
	}

	private MatrixObject processRandInstruction(ExecutionContext ec) {
		MatrixObject out;

		// derive number of workers to use
		int nworkers = DMLScript.FED_WORKER_PORTS.size();
		if(getRows() % nworkers != 0) {
			if(Math.round(getRows() / (double) nworkers) == Math.floor(getRows() / (double) nworkers))
				nworkers--;
		}

		// generate seeds and rows
		long[] lSeeds = randomSeedsGenerator(nworkers);

		// new sizes
		int size = (int) Math.round(getRows() / (double) nworkers);
		int addedRows = size * (nworkers -1);
		long[] rowsPerWorker = new long[nworkers];
		Arrays.fill(rowsPerWorker, 0, nworkers-1, size);
		rowsPerWorker[nworkers-1] = getRows() - addedRows;

		out = processRandInstructionMatrix(ec, rowsPerWorker, lSeeds);

		// reset runtime seed (e.g., when executed in loop)
		runtimeSeed = null;
		return out;
	}

	private MatrixObject processRandInstructionMatrix(ExecutionContext ec, long[] rowsPerWorker, long[] lSeeds) {
		long lrows = ec.getScalarInput(rows).getLongValue();
		long lcols = ec.getScalarInput(cols).getLongValue();
		checkValidDimensions(lrows, lcols);

		String[] instStrings = new String[rowsPerWorker.length];
		Arrays.fill(instStrings, 0, rowsPerWorker.length, instString);

		InetAddress addr = null;
		try {
			addr = InetAddress.getLocalHost();
		}
		catch(UnknownHostException e) {
			e.printStackTrace();
		}
		String host = addr.getHostName();

		int rangeBegin = 0;
		List<Pair<FederatedRange, FederatedData>> d = new ArrayList<>();
		for(int i = 0; i < rowsPerWorker.length; i++) {
			// new inst strings
			instStrings[i] = InstructionUtils.replaceOperand(instString, 2, InstructionUtils.createLiteralOperand(String.valueOf(rowsPerWorker[i]), ValueType.INT64));
			instStrings[i] = InstructionUtils.replaceOperand(instStrings[i], 8, String.valueOf(lSeeds[i]));

			FederatedRange X1r = new FederatedRange(new long[]{rangeBegin, 0}, new long[] {rangeBegin += rowsPerWorker[i], lcols});
			FederatedData X1d = null;
			try {

				InetSocketAddress inetSocketAddress = new InetSocketAddress(InetAddress.getByName(host), DMLScript.FED_WORKER_PORTS.get(i));
				X1d = new FederatedData(DataType.MATRIX, inetSocketAddress, null);
			}
			catch(UnknownHostException e) {
				e.printStackTrace();
			}
			d.add(new ImmutablePair<>(X1r, X1d));
		}

		FederationMap fedMap = new FederationMap(d);
		fedMap.setType(FederationMap.FType.ROW);

		long id = FederationUtils.getNextFedDataID();

		FederatedRequest[] fr1 = FederationUtils.callInstruction(instStrings, output, id, Types.ExecType.CP);
		boolean currFlagStatus = FEDInstructionUtils.fedDataGen;
		FEDInstructionUtils.fedDataGen = false;
		fedMap.execute(getTID(), true, fr1, new FederatedRequest[0]);
		FEDInstructionUtils.fedDataGen = currFlagStatus;

		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(new MatrixCharacteristics(lrows, lcols, blocksize, lrows * lcols));
		out.setFedMapping(fedMap.copyWithNewID(fr1[0].getID()));

		return out;
	}

	private long generateSeed() {
		// generate pseudo-random seed (because not specified)
		long lSeed = seed; // seed per invocation
		if(lSeed == DataGenOp.UNSPECIFIED_SEED) {
			if(runtimeSeed == null)
				runtimeSeed = DataGenOp.generateRandomSeed();
			lSeed = runtimeSeed;
		}

		if(LOG.isTraceEnabled())
			LOG.trace("Process DataGenCPInstruction rand with seed = " + lSeed + ".");

		return lSeed;
	}

	private long[] randomSeedsGenerator(int n) {
		long curSeed = generateSeed();
		long[] seeds = new long[n];

		seeds[0] = curSeed;

		IntStream.range(1, n).forEach(i -> {
			if(runtimeSeed != null && curSeed == runtimeSeed) {
				runtimeSeed = null;
				seeds[i] = generateSeed();
			} else {
				Random generator = new Random(curSeed);
				seeds[i] = generator.nextLong();
			}
		});
		return seeds;
	}

	private static void checkValidDimensions(long rows, long cols) {
		// check valid for integer dimensions (we cannot even represent empty blocks with larger dimensions)
		if(rows > Integer.MAX_VALUE || cols > Integer.MAX_VALUE)
			throw new DMLRuntimeException("DataGenFEDInstruction does not "
				+ "support dimensions larger than integer: rows=" + rows + ", cols=" + cols + ".");
	}

}

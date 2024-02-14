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

package org.apache.sysds.runtime.matrix.data;

import org.apache.sysds.runtime.io.ReaderWavFile;

import java.net.URL;

import java.io.InputStream;
import java.io.FileInputStream;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class LibMatrixKeywordSpotting {

	List<double[]> samples = new ArrayList<>();
	List<String> labels = new ArrayList<>();

	public LibMatrixKeywordSpotting() {

		// load all data
		// data: http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip
		// zip contains command folders which contain corresponding .wav files
		// maybe change label to int?
		loadAllData();

		// convert waveforms to magnitudes of spectrogram
		// uses stft
		for (int i = 0; i < samples.size(); i++){
			double[] wave = samples.get(i);
			double[] magnitudes = convertWaveToMagnitudesSpectrogram(wave);
			samples.set(i, magnitudes);
		}

		// TODO:
		// train model
		// use gaussianClassifier???
		// [prior, means, covs, det] = gaussianClassifier(D=X, C=y, varSmoothing=$2);
		// use global variables for classifier
	}

	private double[] convertWaveToMagnitudesSpectrogram(double[] wave){

		// length=255, overlap=128
		// TODO: adjust stft
		double[][] spectrogram = LibMatrixSTFT.one_dim_stft(wave, 255, 128);

		int cols = spectrogram[0].length;
		double[] magnitudes = new double[cols];
		for (int i = 0; i < cols; i++){
			magnitudes[i] = Math.sqrt(Math.pow(spectrogram[0][i], 2) + Math.pow(spectrogram[0][i], 2));
		}

		return magnitudes;
	}

	public String predictCommandForFile(String filePath){

		// read wave file
		double[] wave = ReaderWavFile.readMonoAudioFromWavFile(filePath);

		// convert waveforms to spectrogram
		double[] magnitudes = convertWaveToMagnitudesSpectrogram(wave);

		// use global variables for prediction
		// TODO

		return null;
	}

	private void loadAllData(){

		String url = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip";

		try {
			// get zip data
			byte[] zipData = getZipData(new URL(url));

			// get folder names
			Set<String> dirs = getDirectories(zipData);

			for (String dir : dirs) {
				readWavFilesDirectory(zipData, dir);
			}

		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	private Set<String> getDirectories(byte[] zipData) throws IOException {

		Set<String> dirs = new HashSet<>();
		ZipInputStream stream = new ZipInputStream(new ByteArrayInputStream(zipData));

		// exclude main directory
		ZipEntry entry = stream.getNextEntry();
		int mainDirLength = entry.getName().length();

		while ((entry = stream.getNextEntry()) != null) {
			if (entry.isDirectory()) {
				String dir = entry.getName();
				// remove "/" at the end
				dirs.add(dir.substring(mainDirLength, dir.length() - 1));
			}
		}

		return dirs;
	}

	private void readWavFilesDirectory(byte[] zipData, String dir) throws IOException {

		ZipInputStream stream = new ZipInputStream(new ByteArrayInputStream(zipData));
		ZipEntry entry;

		while ((entry = stream.getNextEntry()) != null) {
			if (entry.getName().startsWith(dir) && entry.isDirectory()) {
				readWavFilesDirectory(stream, dir);
				// dont read next dir
				break;
			}
		}

	}

	private void readWavFilesDirectory(ZipInputStream stream, String dir) throws IOException {

		ZipEntry entry;
		while ((entry = stream.getNextEntry()) != null && !entry.isDirectory() && entry.getName().endsWith(".wav")) {
			readWavFile(entry, dir);
		}

	}

	private void readWavFile(ZipEntry entry, String dir) {

		InputStream stream = new ByteArrayInputStream(entry.getExtra());
		double[] data = ReaderWavFile.readMonoAudioFromWavFile(stream);
		samples.add(data);
		labels.add(dir);

	}

	private byte[] getZipData(URL url) throws IOException {
		InputStream in = url.openConnection().getInputStream();

		ByteArrayOutputStream out = new ByteArrayOutputStream();
		byte[] dataBuffer = new byte[1024];

		int bytesRead;
		while ((bytesRead = in.read(dataBuffer, 0, 1024)) != -1) {
			out.write(dataBuffer, 0, bytesRead);
		}

		return out.toByteArray();
	}

}

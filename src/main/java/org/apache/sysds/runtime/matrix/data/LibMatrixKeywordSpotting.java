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
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

import java.util.List;
import java.util.ArrayList;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class LibMatrixKeywordSpotting {

	public static void main(String[] args) {

		// load all data
		String url = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip";

		// zip contains command folders which contain corresponding .wav files
		List<double[]> waves = new ArrayList<>();
		List<String> labels = new ArrayList<>();
		loadAllData(url, waves, labels);

		// convert waveforms to magnitudes of spectrogram
		// uses stft
		List<double[]> spectrograms = new ArrayList<>();
		for(double[] wave : waves) {
			double[] magnitudes = convertWaveToMagnitudesSpectrogram(wave);
			spectrograms.add(magnitudes);
		}

		// TODO:
		// csv for spectrograms
		// csv for labels - use index?
		// if we use index we also need a csv to translate index back to command

	}

	private static void loadAllData(String url, List<double[]> waves, List<String> labels) {

		try {
			// get zip data
			byte[] zipData = getBytesZipFile(new URL(url));

			// get folder names
			List<String> dirs = getDirectories(zipData);

			readWaveFiles(zipData, dirs, waves, labels);

		}
		catch(IOException e) {
			e.printStackTrace();
		}

	}

	private static byte[] getBytesZipFile(URL url) throws IOException {

		InputStream in = url.openConnection().getInputStream();

		ByteArrayOutputStream out = new ByteArrayOutputStream();
		byte[] dataBuffer = new byte[1024];

		int bytesRead;
		while((bytesRead = in.read(dataBuffer, 0, 1024)) != -1) {
			out.write(dataBuffer, 0, bytesRead);
		}

		return out.toByteArray();

	}

	private static List<String> getDirectories(byte[] zipData) throws IOException {

		List<String> dirs = new ArrayList<>();
		ZipInputStream stream = new ZipInputStream(new ByteArrayInputStream(zipData));

		// exclude main directory
		ZipEntry entry = stream.getNextEntry();
		int mainDirLength = entry.getName().length();

		while((entry = stream.getNextEntry()) != null) {
			if(entry.isDirectory()) {
				String dir = entry.getName();
				// remove "/" at the end
				dirs.add(dir.substring(mainDirLength, dir.length() - 1));
			}
		}

		return dirs;
	}

	private static void readWaveFiles(byte[] zipData, List<String> dirs, List<double[]> waves, List<String> labels)
		throws IOException {

		ZipInputStream stream = new ZipInputStream(new ByteArrayInputStream(zipData));
		ZipEntry entry;
		String dir = dirs.get(0);

		while((entry = stream.getNextEntry()) != null) {
			if(!entry.isDirectory() && entry.getName().startsWith(dir) && entry.getName().endsWith(".wav")) {

				// read file
				double[] data = ReaderWavFile.readMonoAudioFromWavFile(new ByteArrayInputStream(entry.getExtra()));
				waves.add(data);
				labels.add(dir);
			}
			else {
				dir = dirs.get(dirs.indexOf(dir) + 1);
			}
		}

	}

	private static double[] convertWaveToMagnitudesSpectrogram(double[] wave) {

		// length=255, overlap=128
		// TODO: adjust stft
		double[][] spectrogram = LibMatrixSTFT.one_dim_stft(wave, 255, 128);

		int cols = spectrogram[0].length;
		double[] magnitudes = new double[cols];
		for(int i = 0; i < cols; i++) {
			magnitudes[i] = Math.sqrt(Math.pow(spectrogram[0][i], 2) + Math.pow(spectrogram[0][i], 2));
		}

		return magnitudes;
	}

}

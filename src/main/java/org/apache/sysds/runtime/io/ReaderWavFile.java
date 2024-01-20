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

package org.apache.sysds.runtime.io;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.UnsupportedAudioFileException;

public class ReaderWavFile {

	public static double[] readMonoAudioFromWavFile(String filePath) {

		try {
			// open audio file
			AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(new File(filePath));
			double[] audioValues = readMonoAudioFromWavFile(audioInputStream);
			audioInputStream.close();
			return audioValues;

		} catch (UnsupportedAudioFileException | IOException e) {
			e.printStackTrace();
			return null;
		}

	}

	public static double[] readMonoAudioFromWavFile(InputStream inputStream) {

		try {
			// open audio file
			AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(inputStream);

			// collapse channels to mono channel
			int channels = 1;
			AudioFormat monoAudioFormat = new AudioFormat(
					audioInputStream.getFormat().getSampleRate(),
					audioInputStream.getFormat().getSampleSizeInBits(),
					channels,
					true,
					false
			);
			AudioInputStream monoAudioInputStream = AudioSystem.getAudioInputStream(monoAudioFormat, audioInputStream);

			// curation of audio
			int numFrames = (int) monoAudioInputStream.getFrameLength();
			// size of one frame in bytes
			int frameSize = monoAudioInputStream.getFormat().getFrameSize();

			// read audio into buffer
			byte[] audioData = new byte[numFrames * frameSize];
			int bytesRead = audioInputStream.read(audioData);

			// read operation failed
			if (bytesRead == -1) {
				return null;
			}

			// convert byte array to double array
			double[] audioValues = new double[numFrames];
			for (int i = 0, frameIndex = 0; i < bytesRead; i += frameSize, frameIndex++) {
				// 16-bit PCM encoding
				// combine two bytes into a 16-bit integer (short)
				short sampleValue = (short) ((audioData[i + 1] << 8) | (audioData[i] & 0xFF));
				// audio ranges from -32768 to 32767, normalize to range -1 to 1
				audioValues[frameIndex] = sampleValue / 32768.0;
			}

			// close audio streams
			monoAudioInputStream.close();
			audioInputStream.close();
			return audioValues;

		} catch (UnsupportedAudioFileException | IOException e) {
			e.printStackTrace();
			return null;
		}

	}

}

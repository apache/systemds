package org.apache.sysds.runtime.io;

import javax.sound.sampled.*;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class ReaderWavFile {

	public static double[] readMonoFromWavFile(String filePath) {
		try {
			// open audio file
			AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(new File(filePath));
			double[] audioValues = readMonoFromWavFile(audioInputStream);
			audioInputStream.close();
			return audioValues;

		} catch (UnsupportedAudioFileException | IOException e) {
			e.printStackTrace();
			return null;
		}
	}

	public static double[] readMonoFromWavFile(InputStream inputStream) {

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

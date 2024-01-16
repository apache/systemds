package org.apache.sysds.runtime.matrix.data;

import org.apache.sysds.runtime.io.ReaderWavFile;

import java.io.*;
import java.net.URL;
import java.util.*;
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
        // use gaussianClassifier/CNN???

        // [prior, means, covs, det] = gaussianClassifier(D=X, C=y, varSmoothing=$2);
        // use global variables for classifier
    }


    private double[] convertWaveToMagnitudesSpectrogram(double[] wave){
        // length=255, overlap=128
        double[][] spectrogram = LibMatrixSTFT.one_dim_stft(wave, 255, 128);

        int cols = spectrogram[0].length;
        double[] magnitudes = new double[cols];
        for (int i = 0; i < cols; i++){
            magnitudes[i] = Math.sqrt(Math.pow(spectrogram[0][i], 2) + Math.pow(spectrogram[0][i], 2));
        }
        return magnitudes;
    }

    public String getCommandForFile(String filePath){

        // read wave file
        double[] wave = ReaderWavFile.readMonoFromWavFile(filePath);

        // convert waveforms to spectrogram
        double[] magnitudes = convertWaveToMagnitudesSpectrogram(wave);

        // use global variables for classifier
        // TODO

        return null;
    }

    private void loadAllData(){

        // doesn't work for url
        // String url = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip";
        // Set<String> dirs = Set.of("yes", "no");

        String zipFilePath = "/Users/jessica/desktop/mini_speech_commands.zip";

        try {
            // get zip data
            byte[] zipData = getZipData(new FileInputStream(zipFilePath));

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
                // remove / at the end
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
        double[] data = ReaderWavFile.readMonoFromWavFile(stream);
        samples.add(data);
        labels.add(dir);
    }

    private byte[] getZipData(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] dataBuffer = new byte[1024];
        int bytesRead;
        while ((bytesRead = in.read(dataBuffer, 0, 1024)) != -1) {
            out.write(dataBuffer, 0, bytesRead);
        }
        return out.toByteArray();
    }

    private byte[] getZipData(URL url) throws IOException {
        InputStream in = new BufferedInputStream(url.openStream());
        return getZipData(in);
    }
}

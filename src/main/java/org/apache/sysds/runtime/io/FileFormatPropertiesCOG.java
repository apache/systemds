package org.apache.sysds.runtime.io;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.Serializable;

public class FileFormatPropertiesCOG extends FileFormatProperties implements Serializable {
    protected static final Log LOG = LogFactory.getLog(FileFormatPropertiesCOG.class.getName());
    private static final long serialVersionUID = 1038419221722594985L;

    private String datasetName;

    public FileFormatPropertiesCOG() {
        this.datasetName = "systemdscog";
    }

    public FileFormatPropertiesCOG(String datasetName) {
        this.datasetName = datasetName;
    }

    public String getDatasetName() {
        return datasetName;
    }

    @Override public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(" datasetName " + datasetName);
        return sb.toString();
    }
}

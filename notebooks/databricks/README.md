#### Setup Apache SystemDS on Databricks platform

1. Create a new account at [databricks cloud](https://community.cloud.databricks.com/)
2. In left-side navbar select **Clusters** > **`+ Create Cluster`** > Name the cluster! > **`Create Cluster`**
3. Navigate to the created cluster configuration.
    1. Select **Libraries**
    2. Select **Install New** > **Library Source [`Upload`]** and **Library Type [`Jar`]**
    3. Upload the `SystemDS.jar` file! > **`Install`**
4. Attach a notebook to the cluster above.

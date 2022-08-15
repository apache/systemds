
# Backend for monitoring tool of federated infrastructure

A backend application, used to collect, store, aggregate and return metrics data from coordinators and workers in the cluster


## Install & Run

The backend process can be started in a similar manner with how a worker is started:

```bash
  cd systemds
  mvn package
  ./bin/systemds [-r] FEDMONITOR [SystemDS.jar] <portnumber> [arguments]
```

Or with the specified **-fedMonitor 8080** flag indicating the start of the backend process on the specified port, in our case **8080**.

## Main components

### Architecture
The following diagram illustrates the processes running in the backend.


![Backend Architecture](./Backend-architecture.svg)

#### Controller
Serves as the main integration point between the frontend and backend.

#### Service
Holds the business logic of the backend application.

#### Repository
serves as the main integration point between the backend and the chosen persistent storage. It can be extended to persist data in the file system, by extending the **IRepository** class and changing the instance in the service classes.

### Database schema
The following diagram illustrates the current state of the database schema.


![Database Schema](./DB-diagram.svg)

**Important to note**
- There is no foreign key constraint between the worker and statistics tables.

### Processes
The following diagram illustrates the processes running in the backend.


![Backend Processes](./Backend-processes.svg)

#### Statistics collection thread
There is a dedicated thread for the communication between the backend and the workers and statistics are gathered periodically (every 3 seconds by default).

#### Request processing
The main logic of the application listens for REST requests coming from the frontend.  
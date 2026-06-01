# Immigration Checkpoint Pedestrian Simulation

## Overview

This project implements a simple 2D immigration checkpoint simulation in AnyLogic. The model represents passengers arriving in batches from buses or planes, moving through an immigration checkpoint, and being routed through different operational checks.

The simulation is designed to help explore queue build-up, traveller flow, and the operational impact of additional screening or interrogation stages. The visual layout is intentionally simple and focuses on system behaviour rather than graphical polish.

## Engine Version Used

- Simulation engine: AnyLogic
- AnyLogic version: AnyLogic Personal Learning Edition 8.9.6
- Operating system: Windows 11

## Libraries Used

This model uses:

- AnyLogic Pedestrian Library

## Simulation Flow

The current simulation flow is:

```text
Vehicle arrival
    ↓
Batch of pedestrians enters checkpoint
    ↓
Pedestrians walk to immigration area
    ↓
Some pedestrians are selected for bag checks
    ↓
Some bag-check pedestrians may require further interrogation
    ↓
Interrogated pedestrians are either:
        - released back into the normal pedestrian flow, or
        - escorted out of the checkpoint
    ↓
Remaining pedestrians proceed through immigration
    ↓
Some pedestrians may require additional immigration checks
    ↓
Additional-check pedestrians are either:
        - released back into the normal pedestrian flow, or
        - escorted out of the checkpoint
    ↓
Cleared pedestrians exit the checkpoint
```

In simple terms, the model simulates a checkpoint where most pedestrians follow the normal flow, while a smaller proportion are diverted for additional checks.

## Arrival Logic

The model uses batch arrivals to represent buses or planes arriving at intervals.

Instead of generating pedestrians continuously at a fixed rate, the model creates groups of pedestrians when a vehicle arrives. This better represents real checkpoint operations where a large number of travellers may enter the checkpoint shortly after a bus or flight arrival.

Example arrival logic:

```text
Vehicle arrives every N minutes
Vehicle carries a random number of pedestrians
Pedestrians are released into the checkpoint flow
```

The exact interval and group size can be adjusted through model parameters.

## Simulation Parameters

The model parameters can be modified inside the `Main` agent in AnyLogic.

Common parameters include:

| Parameter | Description |
|---|---|
| `numImmigrationCounters` | Number of available immigration counters |
| `numSecurityCounters` | Number of available bag-check stations |
| `numInterrogationCounters` | Number of available interrogation stations |
| `busIntervalMinutes` | Time interval between bus/plane arrivals |
| `minBusPassengers` | Minimum number of pedestrians per vehicle arrival |
| `maxBusPassengers` | Maximum number of pedestrians per vehicle arrival |
| `securityCheckProbability` | Probability that a pedestrian is selected for bag check |
| `extraSecurityCheckProbability` | Probability that a pedestrian that is selected for bag check needs further check|
| `extraSecurityCheckFailProbability` | Probability that a pedestrian fails the further check and is escorted out |
| `extraImmigrationCheckProbability` | Probability that a pedestrian is sent for further interrogation after immigration check|
| `extraImmigrationCheckFailProbability` | Probability that an interrogated pedestrian is escorted out |

Processing-time parameters can also be configured, such as:

| Parameter | Description |
|---|---|
| `immigrationProcessTime` | Time taken for normal immigration processing |
| `bagCheckProcessTime` | Time taken for bag checking |
| `extraSecurityCheckProcessTime` | Time taken for interrogation |
| `extraImmigrationCheckTime` | Time taken for additional immigration checks |

## Dashboard Metrics

The simulation dashboard displays key operational metrics, including:

- Completed travellers
- Number of pedestrians currently in the checkpoint
- Immigration queue length
- Bag-check queue length
- Interrogation queue length
- Average immigration waiting time
- Average bag-check waiting time
- Average interrogation waiting time
- Average total time in system

These metrics are used to understand where congestion forms and how additional screening stages affect overall checkpoint performance.

## Setup Instructions

1. Install AnyLogic.
2. Open AnyLogic.
3. Clone or download this repository.
4. Open the provided `.alp` model file in AnyLogic.
5. Ensure that the AnyLogic Pedestrian Library is available in the installed AnyLogic version.
6. Open the `Simulation` experiment.

## Run Instructions

To run the simulation:

1. Open the `.alp` file in AnyLogic.
2. Open the `Simulation` experiment.
3. Click **Run**.
4. Observe the 2D pedestrian movement and dashboard metrics.
5. Adjust parameters in the `Main` agent if different operating conditions need to be tested.

## Build Instructions

No standalone build is required.

The model should be run directly inside AnyLogic using the provided `.alp` file.

## How to Modify Simulation Parameters

To modify simulation parameters:

1. Open the model in AnyLogic.
2. In the Projects panel, open the `Main` agent.
3. Locate the parameter objects.
4. Edit the default values of the relevant parameters.
5. Re-run the `Simulation` experiment.

Examples of parameters that can be changed:

```text
Number of immigration counters
Number of bag-check counters
Vehicle arrival interval
Passenger group size
Probability of bag checks
Probability of failing bag checks
Probability of interrogation
Probability of additional immigration checks
Processing-time distributions
```

Changing these parameters allows the user to test different checkpoint operating conditions.

## Assumptions

The simulation uses the following assumptions:

- Travellers arrive in batches to represent buses or planes.
- Pedestrians are processed individually even though they arrive in groups.
- The 2D layout is simplified and generic.
- Each counter or check station serves one pedestrian at a time.
- Bag checks, interrogation, escort-out decisions, and additional immigration checks are probability-based.
- Processing times are assumed distributions and can be adjusted.

# CPU Scheduling Simulator with DART Algorithm

## Project Overview

This project is a **CPU Scheduling Simulator** developed for an Operating Systems course.

It implements several classical CPU scheduling algorithms and extends an existing simulator by adding a custom scheduling method called **DART (Dynamic Adaptive Response Time)**.

The simulator allows users to:

- Add processes with **Arrival Time (AT)** and **CPU Burst Time (CBT)**
- Configure **context switch time** and **time quantum**
- Run different scheduling algorithms
- Compare performance metrics
- Visualize execution using a **Gantt chart**

## Implemented Algorithms

The simulator includes:

- **FCFS** – First Come First Served  
- **SPN (SJF)** – Shortest Process Next  
- **SRTF** – Shortest Remaining Time First  
- **HRRN** – Highest Response Ratio Next  
- **Round Robin (RR)**  
- **DART** – Dynamic Adaptive Response Time (Project Extension)

## About DART

DART is a custom scheduling algorithm designed to:

- Improve responsiveness for interactive processes  
- Prevent starvation  
- Dynamically adapt to system behavior  

It calculates a priority score based on:

- Waiting time  
- Remaining burst time  
- Previous CPU burst behavior  
- Penalty factor for CPU-heavy processes  

This makes it more adaptive compared to traditional static algorithms.


## Technologies Used

- **Python 3**
- **Tkinter** (GUI)
- **Matplotlib** (Gantt chart visualization)


### Main Components

**1. CPUScheduler Class**
- Implements all scheduling algorithms
- Calculates Waiting Time (WT)
- Calculates Turnaround Time (TT)
- Handles context switching
- Generates Gantt chart data

**2. SchedulerApp Class**
- Handles GUI interface
- Takes user input
- Runs selected algorithm
- Displays results and charts


##  How to Run

1. Install Python 3  
2. Install required library:

```bash
pip install matplotlib

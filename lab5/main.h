#ifndef MAIN_H
#define MAIN_H

#include <mpi.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INITIAL_TASK_COUNT 200
#define TOTAL_WEIGHT 2e9

#define TAG_REQUEST   1
#define TAG_TASK_INFO 2
#define TAG_TASK_DATA 3
#define TAG_TERMINATE 4

typedef struct {
    int repeatNum;
} Task;

typedef struct {
    Task        *taskArray;
    int          taskCount;
    int          inProgress;
    double       globalRes;
    int          rank;
    int          size;
    int          provided;
    MPI_Comm     recv_comm;
    pthread_mutex_t taskMutex;
} Context;

void addTasks(Context *ctx, const Task *tasks, int count);
bool fetchTask(Context *ctx, int *rep);
void *receiverThread(void *arg);

void defaultDistribution(Context* ctx, Task *initial);
void increasingDistribution(Context *ctx, Task *initial);
void decreasingDistribution(Context *ctx, Task *initial);
void sinDistribution(Context *ctx, Task *initial);
void firstAllDistribution(Context *ctx, Task *initial);
void firstHalfAllDistribution(Context *ctx, Task *initial);

void normalizeGlobal(Task *initial, int N);
void printTotalWeightAllTasks(Context *ctx, Task *initial);

#endif

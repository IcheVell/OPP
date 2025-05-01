#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
#include <mpi.h>

#define ITER_MAX         10
#define TASKS_PER_ITER  100

enum {
    TAG_REQUEST   = 1,
    TAG_RESPONSE  = 2,
    TAG_TERMINATE = 3
};

typedef struct {
    int  *data;
    int   size;
    int   capacity;
    pthread_mutex_t mutex;
} TaskQueue;

typedef struct {
    TaskQueue *queue;
    int        rank, size;
    MPI_Comm   comm;
    double    *globalRes;
} ThreadArgs;

int pop_task(TaskQueue *q, int *task);
double heavy_compute(int repeats);
int split_tasks(TaskQueue *q, int **out);
void *listener_thread(void *v);
void *worker_thread(void *v);

#endif

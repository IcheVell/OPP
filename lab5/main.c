#include "main.h"

int main(int argc, char **argv) {
    Context ctx;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ctx.provided);
    if (ctx.provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI_THREAD_MULTIPLE not supported\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);

    MPI_Comm_dup(MPI_COMM_WORLD, &ctx.recv_comm);

    ctx.taskCount    = 0;
    ctx.inProgress   = 0;
    pthread_mutex_init(&ctx.taskMutex, NULL);
    ctx.taskArray = malloc(INITIAL_TASK_COUNT * sizeof(Task));
    if (!ctx.taskArray) {
        perror("Can't allocate memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    pthread_t recvThr;
    pthread_create(&recvThr, NULL, receiverThread, &ctx);

    Task *initial = calloc(INITIAL_TASK_COUNT, sizeof(Task));

    //firstHalfAllDistribution(&ctx, initial);
    //firstAllDistribution(&ctx, initial);
    //sinDistribution(&ctx, initial);
    //decreasingDistribution(&ctx, initial);
    //increasingDistribution(&ctx, initial);
    defaultDistribution(&ctx, initial);

    normalizeGlobal(initial, INITIAL_TASK_COUNT);

    printTotalWeightAllTasks(&ctx, initial);

    addTasks(&ctx, initial, INITIAL_TASK_COUNT);
    free(initial);

    double timeStart = MPI_Wtime();
    long long solvedWeight = 0;
    while (1) {
        int rep;
        if (fetchTask(&ctx, &rep)) {
            solvedWeight += rep;
            for (int i = 0; i < rep; ++i) {
                ctx.globalRes += sqrt(i) * sin(i) * cos(i);
            }
            pthread_mutex_lock(&ctx.taskMutex);
            ctx.inProgress--;
            pthread_mutex_unlock(&ctx.taskMutex);
            continue;
        }

        int workFound = 0;
        for (int d = 1; d < ctx.size && !workFound; ++d) {
            int tgt = (ctx.rank + d) % ctx.size;

            MPI_Send(NULL, 0, MPI_CHAR, tgt, TAG_REQUEST, ctx.recv_comm);

            int shareCount;
            MPI_Recv(&shareCount, 1, MPI_INT, tgt, TAG_TASK_INFO, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (shareCount > 0) {
                Task *buf = malloc(shareCount * sizeof(Task));
                MPI_Recv(buf, shareCount * sizeof(Task), MPI_BYTE, tgt, TAG_TASK_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                addTasks(&ctx, buf, shareCount);
                free(buf);
                workFound = 1;
                break;
            }
        }
        if (workFound) {
            continue;
        }

        int localLeft = ctx.taskCount + ctx.inProgress;
        int globalLeft;
        MPI_Allreduce(&localLeft, &globalLeft, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (globalLeft == 0) {
            break;
        }
    }

    double timeEnd = MPI_Wtime();
    double totalTime = timeEnd - timeStart;

    printf("Rank %d solved total weight = %lld\n", ctx.rank, solvedWeight);
    fflush(stdout);

    if (ctx.rank == 0) {
        printf("Total loops time: %f s\n", totalTime);
        fflush(stdout);
    }

    for (int p = 0; p < ctx.size; ++p) {
        if (p == ctx.rank) continue;
        MPI_Send(NULL, 0, MPI_CHAR, p, TAG_TERMINATE, ctx.recv_comm);
    }
    pthread_join(recvThr, NULL);

    free(ctx.taskArray);
    pthread_mutex_destroy(&ctx.taskMutex);
    MPI_Comm_free(&ctx.recv_comm);
    MPI_Finalize();

    if (ctx.rank == 0) {
        fprintf(stderr, "Job succeeded\n");
    }
    return 0;
}

void addTasks(Context *ctx, const Task *tasks, int count) {
    pthread_mutex_lock(&ctx->taskMutex);
    for (int i = 0; i < count; ++i) {
        ctx->taskArray[ctx->taskCount++] = tasks[i];
    }
    pthread_mutex_unlock(&ctx->taskMutex);
}

bool fetchTask(Context *ctx, int *rep) {
    pthread_mutex_lock(&ctx->taskMutex);
    if (ctx->taskCount > 0) {
        *rep = ctx->taskArray[--ctx->taskCount].repeatNum;
        ctx->inProgress++;
        pthread_mutex_unlock(&ctx->taskMutex);
        return true;
    }
    pthread_mutex_unlock(&ctx->taskMutex);
    return false;
}

void *receiverThread(void *arg) {
    Context *ctx = (Context*)arg;
    MPI_Status status;

    while (1) {
        MPI_Recv(NULL, 0, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, ctx->recv_comm, &status);

        if (status.MPI_TAG == TAG_REQUEST) {
            int src = status.MPI_SOURCE;
            pthread_mutex_lock(&ctx->taskMutex);
            int shareCount = ctx->taskCount / 2;
            Task *shareBuf = malloc(shareCount * sizeof(Task));
            for (int i = 0; i < shareCount; ++i) {
                shareBuf[i] = ctx->taskArray[--ctx->taskCount];
            }
            pthread_mutex_unlock(&ctx->taskMutex);

            MPI_Send(&shareCount, 1, MPI_INT, src, TAG_TASK_INFO, MPI_COMM_WORLD);

            if (shareCount > 0) {
                MPI_Send(shareBuf, shareCount * sizeof(Task), MPI_BYTE, src, TAG_TASK_DATA, MPI_COMM_WORLD);
            }
            free(shareBuf);
        }
        else if (status.MPI_TAG == TAG_TERMINATE) {
            break;
        }
    }
    return NULL;
}

void defaultDistribution(Context *ctx, Task *initial) {
    for (int i = 0; i < INITIAL_TASK_COUNT; i++) {
        initial[i].repeatNum = 1;
    }
}

void increasingDistribution(Context *ctx, Task *initial) {
    int totalTasks = ctx->size * INITIAL_TASK_COUNT;
    for (int i = 0; i < INITIAL_TASK_COUNT; i++) {
        double currNumTask = (INITIAL_TASK_COUNT * (ctx->rank) + i);
        double factor = currNumTask / totalTasks;
        initial[i].repeatNum = factor * 100000;
    }
}

void decreasingDistribution(Context *ctx, Task *initial) {
    int totalTasks = ctx->size * INITIAL_TASK_COUNT;
    for (int i = 0; i < INITIAL_TASK_COUNT; i++) {
        double currNumTask = (INITIAL_TASK_COUNT * (ctx->rank) + i);
        double factor = (totalTasks - currNumTask) / totalTasks;
        initial[i].repeatNum = factor * 100000;
    }
}


void sinDistribution(Context *ctx, Task *initial) {
    int totalTasks = ctx->size * INITIAL_TASK_COUNT;
    double step = M_PI / totalTasks;
    for (int i = 0; i < INITIAL_TASK_COUNT; i++) {
        double currNumTask = (INITIAL_TASK_COUNT * (ctx->rank) + i);
        double radian = step * currNumTask;
        double s = sin(radian);
        initial[i].repeatNum = s * 100000;
    }
}


void firstAllDistribution(Context *ctx, Task *initial) {
    if (ctx->rank == 0) {
        defaultDistribution(ctx, initial);
    }
}

void firstHalfAllDistribution(Context *ctx, Task *initial) {
    if (ctx->rank < ctx->size / 2) {
        defaultDistribution(ctx, initial);
    }
}

void normalizeGlobal(Task *initial, int N) {
    double localSum = 0.0;
    for (int i = 0; i < N; ++i) {
        localSum += initial[i].repeatNum;
    }

    double globalSum = 0.0;
    MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (globalSum > 0.0) {
        double factor = TOTAL_WEIGHT / globalSum;
        for (int i = 0; i < N; ++i) {
            initial[i].repeatNum = (int)(initial[i].repeatNum * factor);
        }
    }
}

void printTotalWeightAllTasks(Context *ctx, Task *initial) {
    double localSum = 0.0;
    for (int i = 0; i < INITIAL_TASK_COUNT; ++i) {
        localSum += initial[i].repeatNum;
    }

    printf("Rank %d: inited %d\n", ctx->rank, (int)localSum);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);

    double globalSum = 0.0;
    MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (ctx->rank == 0) {
        printf("total normalized complexity = %.0f\n", globalSum);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}
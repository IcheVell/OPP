#include "main.h"

int main(int argc, char **argv) {
    MPI_Init_thread(&argc, &argv,
                    MPI_THREAD_MULTIPLE, &(int){0});

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int iter = 0; iter < ITER_MAX; iter++) {
        TaskQueue queue;
        queue.capacity = TASKS_PER_ITER * size;
        queue.size     = TASKS_PER_ITER;
        queue.data     = malloc(queue.capacity * sizeof(int));
        pthread_mutex_init(&queue.mutex, NULL);

        int weight = abs(rank - (iter % size)) + 1;
        for (int i = 0; i < TASKS_PER_ITER; i++)
            queue.data[i] = weight;

        double globalRes = 0.0;
        ThreadArgs args = {
            .queue     = &queue,
            .rank      = rank,
            .size      = size,
            .comm      = MPI_COMM_WORLD,
            .globalRes = &globalRes
        };

        pthread_t listener, worker;
        pthread_create(&listener, NULL,
                       listener_thread, &args);
        pthread_create(&worker, NULL,
                       worker_thread, &args);

        pthread_join(worker, NULL);
        pthread_join(listener, NULL);

        pthread_mutex_destroy(&queue.mutex);
        free(queue.data);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Iteration %d: result = %f\n",
                   iter, globalRes);
        }
    }

    MPI_Finalize();
    return 0;
}

int pop_task(TaskQueue *q, int *task) {
    pthread_mutex_lock(&q->mutex);
    if (q->size > 0) {
        *task = q->data[--q->size];
        pthread_mutex_unlock(&q->mutex);
        return 1;
    }
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

int split_tasks(TaskQueue *q, int **out) {
    pthread_mutex_lock(&q->mutex);
    int count = q->size / 2;
    if (count > 0) {
        *out = malloc(count * sizeof(int));
        for (int i = 0; i < count; i++)
            (*out)[i] = q->data[q->size - count + i];
        q->size -= count;
    }
    pthread_mutex_unlock(&q->mutex);
    return count;
}

void *listener_thread(void *v) {
    ThreadArgs *a = v;
    MPI_Status  st;
    int         buf;

    while (1) {
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_TERMINATE, a->comm, &flag, &st);
        if (flag) {
            MPI_Recv(&buf, 1, MPI_INT, st.MPI_SOURCE,
                     TAG_TERMINATE, a->comm, &st);
            break;
        }

        MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE,
                 TAG_REQUEST, a->comm, &st);
        int src = st.MPI_SOURCE;

        int *tasks_to_send = NULL;
        int  cnt = split_tasks(a->queue, &tasks_to_send);

        MPI_Send(&cnt, 1, MPI_INT, src, TAG_RESPONSE, a->comm);
        if (cnt > 0) {
            MPI_Send(tasks_to_send, cnt, MPI_INT,
                     src, TAG_RESPONSE, a->comm);
            free(tasks_to_send);
        }
    }

    return NULL;
}

double heavy_compute(int repeats) {
    double acc = 0.0;
    for (int i = 0; i < repeats; i++) {
        acc += sqrt(i);
    }
    return acc;
}

void *worker_thread(void *v) {
    ThreadArgs *a = v;
    MPI_Status  st;
    int         dummy;
    int         idle_rounds = 0;

    while (1) {
        int task;
        if (pop_task(a->queue, &task)) {
            *a->globalRes += heavy_compute(task);
            idle_rounds = 0;
        } else {
            int next = (a->rank + 1) % a->size;
            MPI_Send(&dummy, 1, MPI_INT, next,
                     TAG_REQUEST, a->comm);

            int cnt;
            MPI_Recv(&cnt, 1, MPI_INT, next,
                     TAG_RESPONSE, a->comm, &st);

            if (cnt > 0) {
                int *buf = malloc(cnt * sizeof(int));
                MPI_Recv(buf, cnt, MPI_INT, next,
                         TAG_RESPONSE, a->comm, &st);
                pthread_mutex_lock(&a->queue->mutex);
                for (int i = 0; i < cnt; i++)
                    a->queue->data[a->queue->size++] = buf[i];
                pthread_mutex_unlock(&a->queue->mutex);
                free(buf);
                idle_rounds = 0;
            } else {
                if (++idle_rounds >= a->size) {
                    MPI_Send(&dummy, 1, MPI_INT, a->rank,
                             TAG_TERMINATE, a->comm);
                    break;
                }
            }
        }
    }
    return NULL;
}

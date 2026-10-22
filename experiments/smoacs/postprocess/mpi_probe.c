#define _GNU_SOURCE
#include <mpi.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    if (sched_getaffinity(0, sizeof(cpus), &cpus) != 0) MPI_Abort(MPI_COMM_WORLD, 2);
    unsigned long long mask = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &cpus)) {
            if (cpu >= 64) MPI_Abort(MPI_COMM_WORLD, 3);
            mask |= 1ULL << cpu;
        }
    }
    unsigned long long *masks = calloc((size_t)size, sizeof(*masks));
    MPI_Allgather(&mask, 1, MPI_UNSIGNED_LONG_LONG, masks, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    int ok = size == atoi(getenv("SLURM_NTASKS")) && mask != 0;
    for (int i = 0; i < size; ++i)
        for (int j = i + 1; j < size; ++j)
            if (masks[i] & masks[j]) ok = 0;
    if (rank == 0) {
        printf("MPI_WORLD_SIZE=%d MPI_AFFINITY_OK=%d MASKS=", size, ok);
        for (int i = 0; i < size; ++i) printf("%llx ", masks[i]);
        printf("\n");
        fflush(stdout);
    }
    free(masks);
    if (!ok) MPI_Abort(MPI_COMM_WORLD, 4);
    MPI_Finalize();
    return 0;
}

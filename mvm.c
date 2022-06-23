// include MPI library
#ifdef USE_MPI
#include <mpi.h>
#endif

// include OpenMP library
#ifdef USE_OMP
#include <omp.h>
#endif

// include standard C math library
#include <math.h>

// include standard C IO library
#include <stdio.h>

// include standard C library
#include <stdlib.h>

// include standard C malloc library
#include <malloc.h>

// library for time measurements
#include <time.h>


int main(int argc, char **argv)
{
    // initialization of MPI
    int processor_rank = 0;
    int processors_num = 1;

#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processors_num);
#endif

    // initialization of OpenMP
    int threads_num = 1;
#ifdef USE_OMP
    threads_num = omp_get_max_threads();
#endif

    // parse matrix size and number of repetition from terminal
    int matrix_size;
    int repetitions_num;

    if (argc != 3)
    {
        if (processor_rank == 0)
        {
            printf("Two arguments should be given: matrix size and number of repetitions!");
#ifdef USE_MPI
            MPI_Finalize();
#endif
        }
        return 1;
    }
    else
    {
        matrix_size = strtol(argv[1], NULL, 10);
        repetitions_num = strtol(argv[2], NULL, 10);
    }

    // compute local matrix (vector) size
    int local_size = matrix_size/processors_num;
    
    // allocate local matrix
    double* local_matrix = (double*) malloc(local_size*matrix_size*sizeof(double));

    // allocate local lhs vector
    double* local_lhs = (double*) malloc(local_size*sizeof(double));

    // allocate local rhs (result) vector
    double* local_rhs = (double*) malloc(local_size*sizeof(double));

    // allocate global lhs vector
    double* global_lhs = (double*) malloc(matrix_size*sizeof(double));

    // pointer to auxilary array (matrix row)
    double* aux_row;

     // time counter
    double time;

#ifndef USE_MPI
    clock_t seq_clock;
#endif

    // initialization of local matrix
    for (int i = 0; i < local_size; i++) 
    {
        for (int j = 0; j < matrix_size; j++)
        {
            local_matrix[i*matrix_size + j] = 1.0/(1.0 + i + processor_rank*local_size + j);
        }
    }

    // initialization of local lhs vector (and global lhs for 1 processor)
    for (int i = 0; i < local_size; i++)
    {
        local_lhs[i] = processor_rank*local_size + i;
        if (processors_num == 1)
        {
            global_lhs[i] = local_lhs[i];
        }
    }

    // initialization of local rhs (result) vector as zero vector
    for (int i=0; i < local_size; i++) 
    {
        local_rhs[i] = 0.0;
    }

    // launch timer
#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime();
#elif defined(USE_OMP)
    time = omp_get_wtime();
#else
    seq_clock = clock();
#endif
    
    // perform repetition for matrix vector multiplication
    for (int k = 0; k < repetitions_num; k++) 
    {
#ifdef USE_MPI
        // Gather global lhs from all processors
        MPI_Allgather(local_lhs, local_size, MPI_DOUBLE, global_lhs, local_size, MPI_DOUBLE, MPI_COMM_WORLD);
#endif

        // perform local matrix-vector multiplication
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int i = 0; i < local_size; i++) 
        {
            // address of i-th matrix row
            aux_row = local_matrix + i*matrix_size;

            for (int j = 0; j < matrix_size; j++)
            {
                local_rhs[i] += aux_row[j]*global_lhs[j];
            }
        }
    }

    // stop timer
#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - time;
#elif defined (USE_OMP)
    time = omp_get_wtime() - time;
#else
    seq_clock = clock() - seq_clock;
    time = (double)seq_clock / CLOCKS_PER_SEC;
#endif

    // compute performance 
    double performance = 1e-6*matrix_size*matrix_size*repetitions_num/time;

    // compute norm of rhs vector for testing
    double sum = 0.0;

#ifdef USE_OMP
#pragma omp parallel for reduction(+:sum)
#endif
    for (int i=0; i < local_size; i++)
         sum += local_rhs[i]*local_rhs[i];

    // collect sum on every process
    double sum_all = sum;
#ifdef USE_MPI
    MPI_Allreduce(&sum, &sum_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    sum = sqrt(sum_all)/repetitions_num;

    if (processor_rank == 0)
       printf("MPI-1p mvm: N=%d np=%d th=%d norm=%lf time=%lf perf=%.2lf MFLOPS\n", matrix_size, processors_num, threads_num, sum, time, performance);

    // free allocated memory
    free(local_matrix);
    free(local_lhs);
    free(local_rhs);
    free(global_lhs);

#ifdef USE_MPI
    // finalize MPI activity
    MPI_Finalize();
#endif
    return 0;
}
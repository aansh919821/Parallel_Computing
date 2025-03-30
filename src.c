#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define INDEX(x, y, z, nx, ny) ((z) * (nx) * (ny) + (y) * (nx) + (x))

// Function to read the input file (only rank 0 does this)
void read_data_rank0(const char *file1, int nx, int ny, int nz, int nc, float ****data) {
    // Allocate 3D array dynamically
    *data = (float ***)malloc(nx * sizeof(float **));
    for (int x = 0; x < nx; x++) {
        (*data)[x] = (float **)malloc(ny * sizeof(float *));
        for (int y = 0; y < ny; y++) {
            (*data)[x][y] = (float *)malloc(nz * nc * sizeof(float));
        }
    }


    FILE *fp = fopen(file1, "r");
    if (!fp) {
        perror("File open error");
        exit(1);
    }

    // Read data: XYZ order
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                for (int t = 0; t < nc; t++) {
                    if (fscanf(fp, "%f", &((data)[x][y][z][t])) != 1) {
                        perror("File read error");
                        fclose(fp);
                        exit(1);
                    }
                }
            }
        }
    }

    fclose(fp);
}

void distribute_data(int rank, int size, int nx, int ny, int nz, int nc, float ****data, float ****local_data, int px, int py, int pz) {
    int nx_local = nx / px;
    int ny_local = ny / py;
    int nz_local = nz / pz;
    int tag = 0;

    if (rank == 0) {
        for (int k = 0; k < nz; k += nz_local) {
            for (int j = 0; j < ny; j += ny_local) {
                for (int i = 0; i < nx; i += nx_local) {
                    int dest_rank = (k / nz_local) * (px * py) + (j / ny_local) * px + (i / nx_local);

                    if (dest_rank == 0) {
                        // Keep the portion assigned to rank 0
                        for (int x = 0; x < nx_local; x++) {
                            for (int y = 0; y < ny_local; y++) {
                                for (int z = 0; z < nz_local; z++) {
                                    for (int t = 0; t < nc; t++) {
                                        (local_data)[x][y][z][t] = (data)[i + x][j + y][k + z][t];
                                    }
                                }
                            }
                        }
                    } else {
                        // Send data to corresponding process
                        MPI_Send(&((data)[i][j][k][0]), nx_local * ny_local * nz_local * nc, MPI_FLOAT, dest_rank, tag, MPI_COMM_WORLD);
                    }
                }
            }
        }
    } else {
        // Receive the assigned portion for this rank
        MPI_Recv(&((local_data)[0][0][0][0]), nx_local * ny_local * nz_local * nc, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 10) {  
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <data_file> <PX> <PY> <PZ> <NX> <NY> <NZ> <NC> <output_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Read command-line arguments
    const char *file1 = argv[1];
    int px = atoi(argv[2]);
    int py = atoi(argv[3]);
    int pz = atoi(argv[4]);
    int nx = atoi(argv[5]);
    int ny = atoi(argv[6]);
    int nz = atoi(argv[7]);
    int nc = atoi(argv[8]);
    const char *output_file = argv[9];

    if (size != px * py * pz) {
        if (rank == 0) {
            fprintf(stderr, "Error: Expected %d processes, but got %d.\n", px * py * pz, size);
        }
        MPI_Finalize();
        return 1;
    }

    float ***data = NULL;  // Full dataset (only on rank 0)
    float ***local_data = NULL; // Local dataset per process

    if (rank == 0) {
        read_data_rank0(file1, nx, ny, nz, nc, &data);
        distribute_data(rank, size, nx, ny, nz, nc, &data, &local_data, px, py, pz);
    }

    MPI_Finalize();
    return 0;
}

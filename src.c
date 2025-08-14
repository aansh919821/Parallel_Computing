#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include <stdbool.h>
#include "mpi.h"

/* Utility functions for finding max/min of two floats */
float max(float a, float b) {
    return (a > b) ? a : b;
}

float min(float a, float b) {
    return (a < b) ? a : b;
}

/* Calculates optimal batch size for I/O operations based on memory constraints */
long long get_time(long long nx, long long ny, long long nz, long long nc) {
    unsigned long long x = 1024*1024*1024;  // 1GB
    x *= 10;  // Using 10GB as memory budget
    x /= (nx*ny*nz);  // Divide by total grid points per timestep
    
    // Return minimum of calculated batch size or total timesteps
    return (x > (unsigned long long)nc) ? nc : (long long)x;
}

int main(int argc, char *argv[]) {
    /* MPI Initialization */
    int myrank, size;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Command line argument validation */
    if (argc != 10) {
        if (myrank == 0) {
            fprintf(stderr, "Usage: %s <data_file> <PX> <PY> <PZ> <NX> <NY> <NZ> <NC> <output_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    /* Parse command line arguments */
    const char *file1 = argv[1];  // Input data file
    long long px = atoi(argv[2]); // Process grid X dimension
    long long py = atoi(argv[3]); // Process grid Y dimension
    long long pz = atoi(argv[4]); // Process grid Z dimension
    long long nx = atoi(argv[5]); // Global data X dimension
    long long ny = atoi(argv[6]); // Global data Y dimension
    long long nz = atoi(argv[7]); // Global data Z dimension
    long long nc = atoi(argv[8]); // Number of timesteps/channels
    const char *output_file = argv[9]; // Output file path

    /* Validate process count matches decomposition */
    if (size != (px * py * pz)) {
        if (myrank == 0) {
            fprintf(stderr, "Error: Expected %d processes, but got %d.\n", px * py * pz, size);
        }
        MPI_Finalize();
        return 1;
    }

    /* Ensure domain is evenly divisible by process grid */
    assert(nx % px == 0);
    assert(ny % py == 0);
    assert(nz % pz == 0);

    /* Calculate local subdomain dimensions by dividing global dimensions by process grid */
    long long lx = nx / px;  // Local size in x-dimension
    long long ly = ny / py;  // Local size in y-dimension 
    long long lz = nz / pz;  // Local size in z-dimension

    /* Store original number of timesteps and calculate optimal batch size */
    long long nc_ = nc;      // Preserve original timestep count
    nc = get_time(lx, ly, lz, nc); // Adjust batch size based on memory constraints

    /* Allocate arrays for tracking extrema counts and values */
    long long int *local_minima_count = malloc(sizeof(long long int) * nc_);  // Count of local minima per timestep
    long long int *local_maxima_count = malloc(sizeof(long long int) * nc_);  // Count of local maxima per timestep
    float *sub_global_minima = malloc(sizeof(float) * nc_);  // Process-local minima values
    float *sub_global_maxima = malloc(sizeof(float) * nc_);  // Process-local maxima values

    /* Define MPI datatype parameters for halo exchange */
    int array_of_sizes[4] = {lx + 2, ly + 2, lz + 2, nc};  // Full local dimensions including ghost cells
    int array_of_subsizes_XY[4] = {lx, ly, 1, nc};         // XY-plane for Z-direction communication  
    int array_of_subsizes_XZ[4] = {lx, 1, lz, nc};         // XZ-plane for Y-direction communication
    int array_of_subsizes_YZ[4] = {1, ly, lz, nc};         // YZ-plane for X-direction communication
    int array_of_starts_XY[4] = {0, 0, 0, 0};                 // Starting indices for all dimensions
    int array_of_starts_XZ[4] = {0, 0, 0, 0}; 
    int array_of_starts_YZ[4] = {0, 0, 0, 0}; 
    /* Create derived MPI datatypes for boundary plane communication */
    MPI_Datatype plane_XY, plane_XZ, plane_YZ;
    // YZ-plane datatype for X-direction communication
    MPI_Type_create_subarray(4, array_of_sizes, array_of_subsizes_YZ, array_of_starts_YZ, 
                            MPI_ORDER_C, MPI_FLOAT, &plane_YZ);
    MPI_Type_commit(&plane_YZ);
    // XZ-plane datatype for Y-direction communication  
    MPI_Type_create_subarray(4, array_of_sizes, array_of_subsizes_XZ, array_of_starts_XZ,
                            MPI_ORDER_C, MPI_FLOAT, &plane_XZ);
    MPI_Type_commit(&plane_XZ);
    // XY-plane datatype for Z-direction communication
    MPI_Type_create_subarray(4, array_of_sizes, array_of_subsizes_XY, array_of_starts_XY,
                            MPI_ORDER_C, MPI_FLOAT, &plane_XY);
    MPI_Type_commit(&plane_XY);

    /* Calculate 3D process coordinates in the decomposition grid */
    int process_x_coordinate = myrank / (py * pz);  // X-coordinate in process grid
    int process_y_coordinate = (myrank / pz) % py;  // Y-coordinate in process grid  
    int process_z_coordinate = (myrank % pz);       // Z-coordinate in process grid

    /* Determine neighbor ranks for halo exchange */
    int x_low = (process_x_coordinate > 0) ? myrank - (py * pz) : MPI_PROC_NULL;  // Left neighbor
    int x_high = (process_x_coordinate < px - 1) ? myrank + (py * pz) : MPI_PROC_NULL; // Right neighbor
    int y_low = (process_y_coordinate > 0) ? myrank - pz : MPI_PROC_NULL;         // Front neighbor
    int y_high = (process_y_coordinate < py - 1) ? myrank + pz : MPI_PROC_NULL;   // Back neighbor
    int z_low = (process_z_coordinate > 0) ? myrank - 1 : MPI_PROC_NULL;          // Bottom neighbor  
    int z_high = (process_z_coordinate < pz - 1) ? myrank + 1 : MPI_PROC_NULL;    // Top neighbor

    /* Allocate contiguous memory buffer for local data including ghost cells */
    float *lin_data2 = malloc(sizeof(float) * (lx + 2) * (ly + 2) * (lz + 2) * nc);

    /* Create 4D array structure mapping to the linear buffer */
    float ****local_data = (float ****)malloc((lx + 2) * sizeof(float ***));
    for (long long x = 0; x < lx + 2; x++) {
        local_data[x] = (float ***)malloc((ly + 2) * sizeof(float **));
        for (long long y = 0; y < ly + 2; y++) {
            local_data[x][y] = (float **)malloc((lz + 2) * sizeof(float *));
            for (long long z = 0; z < lz + 2; z++) {
                /* Calculate linear index with proper striding:
                * x-stride: (ly+2)*(lz+2)*nc 
                * y-stride: (lz+2)*nc
                * z-stride: nc */
                long long index = x * (ly + 2) * (lz + 2) * nc  // X offset
                            + y * (lz + 2) * nc              // Y offset
                            + z * nc;                        // Z offset
                local_data[x][y][z] = &lin_data2[index];  // Point to location in linear buffer
            }
        }
    }

    /* Alternative process coordinate calculation */
    int px_idx = myrank / (py * pz);      // X index in process grid
    int remainder = myrank % (py * pz);   // Remaining after X division
    int py_idx = remainder / pz;          // Y index in process grid  
    int pz_idx = remainder % pz;          // Z index in process grid

    /* Calculate global starting indices for this subdomain */
    long long global_x_start = px_idx * lx;  // Global X start index
    long long global_y_start = py_idx * ly;  // Global Y start index
    long long global_z_start = pz_idx * lz;  // Global Z start index
    /* Allocate temporary buffer for reading a batch of timesteps */
    float *local_inner = malloc(sizeof(float) * lx * ly * lz * nc);

    /* Open the input file in parallel using MPI-IO */
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, file1, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    /* Initialize timing variables */
    double t_main=0;    // Computation time (excludes I/O)
    double t_read=0;    // I/O read time
    double t_total=0;   // Total processing time

    /* Process data in batches to optimize memory usage */
    for(long long time=0; time<nc_; time+=nc) {
        double time1=MPI_Wtime();  // Start timer for this batch
        
        /* Calculate remaining timesteps in this batch */
        long long k=nc_-time;
        if(k>nc) k=nc;  // Don't exceed maximum batch size
        
        /* Parallel file reading - each process reads its subdomain */
        for (long long local_z = 0; local_z < lz; local_z++) {
            long long global_z = global_z_start + local_z;  // Global Z index
            
            for (long long local_y = 0; local_y < ly; local_y++) {
                long long global_y = global_y_start + local_y;  // Global Y index
                MPI_Offset offset;
                long long row_index = (local_z * ly + local_y);  // Linear index in local_inner
                
                if(k==nc_) {
                    /* Contiguous read for full batch */
                    offset = (MPI_Offset)(((global_z * ny + global_y) * nx + global_x_start) * nc * sizeof(float));
                    MPI_File_read_at(fh, offset, local_inner + row_index * lx * nc, lx * nc, MPI_FLOAT, &status);
                }
                else {
                    /* Non-contiguous read for partial batch */
                    for(long long local_x=0; local_x<lx; local_x++) {
                        long long global_x = global_x_start + local_x;  // Global X index
                        offset = (MPI_Offset)((((global_z * ny + global_y) * nx + global_x) * nc_ + time)* sizeof(float));
                        MPI_File_read_at(fh, offset, local_inner + row_index * lx * k + local_x*k, k, MPI_FLOAT, &status);
                    }
                }
            }
        }
        
        /* Copy read data from linear buffer to 4D array (including ghost cell padding) */
        long long idx = 0;
        for (long long local_z = 0; local_z < lz; local_z++) {
            for (long long local_y = 0; local_y < ly; local_y++) {
                for (long long local_x = 0; local_x < lx; local_x++) {
                    for (long long t = 0; t < k; t++) {
                        local_data[local_x + 1][local_y + 1][local_z + 1][t] = local_inner[idx++];
                    }
                }
            }
        }
    
        double time2=MPI_Wtime();  // End of read phase, start computation
    
        /* Non-blocking halo exchange in all 3 dimensions */
        MPI_Request send_request_x[2], recv_request_x[2];  // X-direction requests
        MPI_Request send_request_y[2], recv_request_y[2];  // Y-direction requests  
        MPI_Request send_request_z[2], recv_request_z[2];  // Z-direction requests
        MPI_Status recv_status_x[2], recv_status_y[2], recv_status_z[2];
        
        /* X-direction communication (left/right neighbors) */
        if (x_high != MPI_PROC_NULL) {
            // Send right boundary plane to x_high neighbor
            MPI_Isend(&local_data[lx][1][1][0], 1, plane_YZ, x_high, 0, MPI_COMM_WORLD, &send_request_x[1]);
            // Receive ghost cells from x_high neighbor
            MPI_Irecv(&local_data[lx + 1][1][1][0], 1, plane_YZ, x_high, 1, MPI_COMM_WORLD, &recv_request_x[1]);
        }
        if (x_low != MPI_PROC_NULL) {
            // Receive ghost cells from x_low neighbor
            MPI_Irecv(&local_data[0][1][1][0], 1, plane_YZ, x_low, 0, MPI_COMM_WORLD, &recv_request_x[0]);
            // Send left boundary plane to x_low neighbor
            MPI_Isend(&local_data[1][1][1][0], 1, plane_YZ, x_low, 1, MPI_COMM_WORLD, &send_request_x[0]);
        }
        
        /* Y-direction communication (front/back neighbors) */
        if (y_high != MPI_PROC_NULL) {
            // Send back boundary plane to y_high neighbor
            MPI_Isend(&local_data[1][ly][1][0], 1, plane_XZ, y_high, 0, MPI_COMM_WORLD, &send_request_y[1]);
            // Receive ghost cells from y_high neighbor
            MPI_Irecv(&local_data[1][ly + 1][1][0], 1, plane_XZ, y_high, 1, MPI_COMM_WORLD, &recv_request_y[1]);
        }
        if (y_low != MPI_PROC_NULL) {
            // Receive ghost cells from y_low neighbor
            MPI_Irecv(&local_data[1][0][1][0], 1, plane_XZ, y_low, 0, MPI_COMM_WORLD, &recv_request_y[0]);
            // Send front boundary plane to y_low neighbor
            MPI_Isend(&local_data[1][1][1][0], 1, plane_XZ, y_low, 1, MPI_COMM_WORLD, &send_request_y[0]);
        }
        
        /* Z-direction communication (top/bottom neighbors) */
        if (z_high != MPI_PROC_NULL) {
            // Send top boundary plane to z_high neighbor
            MPI_Isend(&local_data[1][1][lz][0], 1, plane_XY, z_high, 0, MPI_COMM_WORLD, &send_request_z[1]);
            // Receive ghost cells from z_high neighbor
            MPI_Irecv(&local_data[1][1][lz + 1][0], 1, plane_XY, z_high, 1, MPI_COMM_WORLD, &recv_request_z[1]);
        }
        if (z_low != MPI_PROC_NULL) {
            // Receive ghost cells from z_low neighbor
            MPI_Irecv(&local_data[1][1][0][0], 1, plane_XY, z_low, 0, MPI_COMM_WORLD, &recv_request_z[0]);
            // Send bottom boundary plane to z_low neighbor
            MPI_Isend(&local_data[1][1][1][0], 1, plane_XY, z_low, 1, MPI_COMM_WORLD, &send_request_z[0]);
        }
        
        /* Wait for all halo exchanges to complete */
        if (x_high != MPI_PROC_NULL) {
            MPI_Wait(&recv_request_x[1], MPI_STATUS_IGNORE);
            MPI_Wait(&send_request_x[1], MPI_STATUS_IGNORE);
        }
        if (x_low != MPI_PROC_NULL) {
            MPI_Wait(&recv_request_x[0], MPI_STATUS_IGNORE);
            MPI_Wait(&send_request_x[0], MPI_STATUS_IGNORE);
        }
        if (y_high != MPI_PROC_NULL) {
            MPI_Wait(&recv_request_y[1], MPI_STATUS_IGNORE);
            MPI_Wait(&send_request_y[1], MPI_STATUS_IGNORE);
        }
        if (y_low != MPI_PROC_NULL) {
            MPI_Wait(&recv_request_y[0], MPI_STATUS_IGNORE);
            MPI_Wait(&send_request_y[0], MPI_STATUS_IGNORE);
        }
        if (z_high != MPI_PROC_NULL) {
            MPI_Wait(&recv_request_z[1], MPI_STATUS_IGNORE);
            MPI_Wait(&send_request_z[1], MPI_STATUS_IGNORE);
        }
        if (z_low != MPI_PROC_NULL) {
            MPI_Wait(&recv_request_z[0], MPI_STATUS_IGNORE);
            MPI_Wait(&send_request_z[0], MPI_STATUS_IGNORE);
        }
        /* Process each timestep in the current batch */
        for (long long c = 0; c < k; c++) {
            /* Initialize tracking variables with first data point (excluding ghost cells) */
            sub_global_maxima[c+time] = local_data[1][1][1][c];  // Initialize with first interior point
            sub_global_minima[c+time] = local_data[1][1][1][c];  // Same for minima
            local_minima_count[c+time] = 0;  // Reset counters for this timestep
            local_maxima_count[c+time] = 0;

            /* Scan entire local domain (excluding ghost cells) */
            for (long long x = 1; x < lx + 1; x++) {
                for (long long y = 1; y < ly + 1; y++) {
                    for (long long z = 1; z < lz + 1; z++) {
                        
                        /* Assume current point is both min and max initially */
                        bool mini = true, maxi = true;
                        
                        /* Update global extrema candidates */
                        sub_global_maxima[c+time] = max(sub_global_maxima[c+time], local_data[x][y][z][c]);
                        sub_global_minima[c+time] = min(sub_global_minima[c+time], local_data[x][y][z][c]);

                        /* Check all 6 neighbors (with boundary awareness) */
                        
                        // Left neighbor (x-1)
                        if (x > 1 || (x_low != MPI_PROC_NULL)) {
                            if (local_data[x-1][y][z][c] >= local_data[x][y][z][c]) maxi = false;
                            if (local_data[x-1][y][z][c] <= local_data[x][y][z][c]) mini = false;
                        }
                        
                        // Right neighbor (x+1)
                        if (x < lx || (x_high != MPI_PROC_NULL)) {
                            if (local_data[x+1][y][z][c] >= local_data[x][y][z][c]) maxi = false;
                            if (local_data[x+1][y][z][c] <= local_data[x][y][z][c]) mini = false;
                        }
                        
                        // Front neighbor (y-1)
                        if (y > 1 || (y_low != MPI_PROC_NULL)) {
                            if (local_data[x][y-1][z][c] >= local_data[x][y][z][c]) maxi = false;
                            if (local_data[x][y-1][z][c] <= local_data[x][y][z][c]) mini = false;
                        }
                        
                        // Back neighbor (y+1)
                        if (y < ly || (y_high != MPI_PROC_NULL)) {
                            if (local_data[x][y+1][z][c] >= local_data[x][y][z][c]) maxi = false;
                            if (local_data[x][y+1][z][c] <= local_data[x][y][z][c]) mini = false;
                        }
                        
                        // Bottom neighbor (z-1)
                        if (z > 1 || (z_low != MPI_PROC_NULL)) {
                            if (local_data[x][y][z-1][c] >= local_data[x][y][z][c]) maxi = false;
                            if (local_data[x][y][z-1][c] <= local_data[x][y][z][c]) mini = false;
                        }
                        
                        // Top neighbor (z+1)
                        if (z < lz || (z_high != MPI_PROC_NULL)) {
                            if (local_data[x][y][z+1][c] >= local_data[x][y][z][c]) maxi = false;
                            if (local_data[x][y][z+1][c] <= local_data[x][y][z][c]) mini = false;
                        }

                        /* Update counters if local extrema found */
                        if (maxi) local_maxima_count[c+time]++;  // Found local maximum
                        if (mini) local_minima_count[c+time]++;  // Found local minimum
                    }
                }
            }
        }

        /* End of computation phase timing */
        double time3 = MPI_Wtime();

        /* Accumulate timing measurements */
        t_main += (time3-time2);    // Add computation time
        t_read += (time2-time1);    // Add I/O time
        t_total += (time3-time1);   // Add total batch processing time
    }

    /* Final timing collection */
    double time4 = MPI_Wtime();
    double t_total_max, t_read_max, t_main_max;
    /* Restore original timestep count */
    nc = nc_;

    /* Allocate reduction buffers */
    long long int *local_minima_total = malloc(sizeof(long long int) * nc);
    long long int *local_maxima_total = malloc(sizeof(long long int) * nc);
    float *global_minima = malloc(sizeof(float) * nc);
    float *global_maxima = malloc(sizeof(float) * nc);

    /* Global reduction operations */
    MPI_Reduce(local_maxima_count, local_maxima_total, nc, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_minima_count, local_minima_total, nc, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(sub_global_maxima, global_maxima, nc, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(sub_global_minima, global_minima, nc, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);

    /* Reduce timing statistics to get worst-case across all processes */
    MPI_Reduce(&t_total, &t_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_read, &t_read_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_main, &t_main_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Finalize timing measurement */
    double time5 = MPI_Wtime();
    t_main += time5-time4;  // Include reduction time in computation

    /* Root process writes results */
    if (myrank == 0) {
        FILE *fp = fopen(output_file, "w");
        if (!fp) {
            perror("Failed to open output file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* 
        * Output File Format:
        * ------------------
        * Line 1: (local_min_count, local_max_count), ... for each timestep
        * Line 2: (global_min_value, global_max_value), ... for each timestep  
        * Line 3: max_read_time, max_compute_time, max_total_time (across all processes)
        */

        /* Line 1: Write local extrema counts for each timestep */
        for (int i = 0; i < nc; i++) {
            // Format: (min_count, max_count) for timestep i
            fprintf(fp, "(%lld,%lld)", local_minima_total[i], local_maxima_total[i]);
            
            // Add comma separator except after last element
            if (i != nc - 1) fprintf(fp, ", ");  
        }
        fprintf(fp, "\n");  // End of line 1

        /* Line 2: Write global extrema values for each timestep */
        for (int i = 0; i < nc; i++) {
            // Format: (min_value, max_value) for timestep i
            fprintf(fp, "(%f,%f)", global_minima[i], global_maxima[i]);
            
            // Add comma separator except after last element  
            if (i != nc - 1) fprintf(fp, ", ");
        }
        fprintf(fp, "\n");  // End of line 2

        /* Line 3: Write maximum timing statistics across all processes */
        // Format: read_time, compute_time, total_time (all with 6 decimal places)
        fprintf(fp, "%.6lf, %.6lf, %.6lf\n", t_read_max, t_main_max, t_total_max);

        /* Close the output file */
        fclose(fp);

        /* Only root process (rank 0) executes the file output code above */
    }  // End of if (myrank == 0) block

    /* Clean up MPI derived datatypes */
    MPI_Type_free(&plane_YZ);  // Free YZ-plane communication datatype
    MPI_Type_free(&plane_XZ);  // Free XZ-plane communication datatype  
    MPI_Type_free(&plane_XY);  // Free XY-plane communication datatype

    /* Finalize MPI environment */
    MPI_Finalize();

    /* Successful program termination */
    return 0;
}
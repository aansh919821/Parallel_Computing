#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include "mpi.h"
void read_data_rank0(const char *file1, int nx, int ny, int nz, int nc, float *****data) {
   // Allocate 3D array dynamically
   // printf("%ld\n",data);
   // printf("%ld\n",*data);
   *data = (float ****)malloc(nx * sizeof(float ***));
   // printf("%ld\n",*data);
   for (int x = 0; x < nx; x++) {
       (*data)[x] = (float ***)malloc(ny * sizeof(float **));
       for (int y = 0; y < ny; y++) {
           (*data)[x][y] = (float **)malloc(nz * sizeof(float *));
           for(int z=0;z<nz;z++){
               (*data)[x][y][z]=(float *)malloc(nc*sizeof(float));
           }
       }
   }
   FILE *fp = fopen(file1, "r");
   if (!fp) {
       perror("File open error");
       exit(1);
   }
   printf("file read\n");
   float temp;
   // Read data: XYZ order
   for (int z = 0; z < nz; z++) {
       for (int y = 0; y < ny; y++) {
           for (int x = 0; x < nx; x++) {
               for (int t = 0; t < nc; t++) {
                   if (fscanf(fp, "%f", (&(*data)[x][y][z][t])) != 1) {
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
void allocate_space_local_data( int nx, int ny, int nz, int nc, float *****data){
   *data = (float ****)malloc(nx * sizeof(float ***));
   // printf("%ld\n",*data);
   for (int x = 0; x < nx; x++) {
       (*data)[x] = (float ***)malloc(ny * sizeof(float **));
       for (int y = 0; y < ny; y++) {
           (*data)[x][y] = (float **)malloc(nz * sizeof(float *));
           for(int z=0;z<nz;z++){
               (*data)[x][y][z]=(float *)malloc(nc*sizeof(float));
           }
       }
   }
}
int main(int argc, char *argv[]){
   int myrank,size;
   MPI_Status status;
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank) ;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   if (argc != 10) { 
       if (myrank == 0) {
           fprintf(stderr, "Usage: %s <data_file> <PX> <PY> <PZ> <NX> <NY> <NZ> <NC> <output_file>\n", argv[0]);
       }
       MPI_Finalize();
       return 1;
   }
   const char *file1 = argv[1];
   int px = atoi(argv[2]);
   int py = atoi(argv[3]);
   int pz = atoi(argv[4]);
   int nx = atoi(argv[5]);
   int ny = atoi(argv[6]);
   int nz = atoi(argv[7]);
   int nc = atoi(argv[8]);
   const char *output_file = argv[9];
   if (size != (px * py * pz)) {
       if (myrank == 0) {
           fprintf(stderr, "Error: Expected %d processes, but got %d.\n", px * py * pz, size);
       }
       MPI_Finalize();
       return 1;
   }
   assert(nx%px==0);
   assert(ny%py==0);
   assert(nz%pz==0);
   float ****data = NULL;
   float ****local_data = NULL;
   if (myrank == 0) {
       read_data_rank0(file1, nx, ny, nz, nc, &data);
       // now can use data[x][y][z][t] to do things
       // distribute_data(rank, size, nx, ny, nz, nc, &data, &local_data, px, py, pz);
   }
   int lx=nx/px;int ly=ny/py;int lz=nz/pz;
   allocate_space_local_data(lx+2,ly+2,lz+2,nc,&local_data);
   if(myrank == 0){
       for(int x=0;x<nx;x++){
           for(int y=0;y<ny;y++){
               for(int z=0;z<nz;z++){
                   int process=(x/lx)*py*pz+(y/ly)*pz+z/lz;
                   if(process) MPI_Send(data[x][y][z],nc,MPI_FLOAT,process,(x%lx)*ly*lz+(y%ly)*lz+(z%lz),MPI_COMM_WORLD);
                   else{
                       for(int c=0;c<nc;c++) local_data[x+1][y+1][z+1][c]=data[x][y][z][c];
                   }
               }
           }
       }
   }
   else{
       for(int x=0;x<lx;x++){
           for(int y=0;y<ly;y++){
               for(int z=0;z<lz;z++){
                   MPI_Recv(local_data[x+1][y+1][z+1],nc,MPI_FLOAT,0,x*ly*lz+y*lz+z,MPI_COMM_WORLD,&status);
               }
           }
       }
   }
   // for(int x=0;x<lx;x++){
   //  for(int y=0;y<ly;y++){
   //      for(int z=0;z<lz;z++){
   //          printf("Process %d: %d %d %d  is %f\n",myrank,x,y,z,local_data[x+1][y+1][z+1][0]);
   //      }
   //  }
   // }
  
   int sTime=MPI_Wtime();




   MPI_Finalize();
   return 0;
}

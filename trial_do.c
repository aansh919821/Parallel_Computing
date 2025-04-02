






#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include <stdbool.h>
#include "mpi.h"
// #define max(a,b)             \
// ({                           \
//     __typeof__ (a) _a = (a); \
//     __typeof__ (b) _b = (b); \
//     _a > _b ? _a : _b;       \
// })








// #define min(a,b)             \
// ({                           \
//     __typeof__ (a) _a = (a); \
//     __typeof__ (b) _b = (b); \
//     _a < _b ? _a : _b;       \
// })
float max(float a,float b){
 return (a>b)?a:b;
}
float min(float a,float b){
 return (a<b)?a:b;
}
void read_data_rank0(const char *file1, int nx, int ny, int nz, int nc, float *****data) {
 // Allocate 3D array dynamically
 // printf("%ld\n",data);
 // printf("%ld\n",*data);
//   *data = (float ****)malloc(nx * sizeof(float ***));
//   // printf("%ld\n",*data);
//   for (int x = 0; x < nx; x++) {
//       (*data)[x] = (float ***)malloc(ny * sizeof(float **));
//       for (int y = 0; y < ny; y++) {
//           (*data)[x][y] = (float **)malloc(nz * sizeof(float *));
//           for(int z=0;z<nz;z++){
//               (*data)[x][y][z]=(float *)malloc(nc*sizeof(float));
//           }
//       }
//   }
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
 double time1=MPI_Wtime();
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
 float *lin_data;
 float ****local_data = NULL;
 if (myrank == 0) {
   lin_data=malloc(sizeof(float)*nx*ny*nz*nc);
   data=(float ****)malloc(nx * sizeof(float ***));
   for (int x = 0; x < nx; x++) {
       data[x] = (float ***)malloc(ny * sizeof(float **));
       for (int y = 0; y < ny; y++) {
           data[x][y] = (float **)malloc(nz * sizeof(float *));
           for (int z = 0; z < nz; z++) {
               // Calculate the index in the 1D array
               int index = x * ny * nz * nc + y * nz * nc + z * nc;
               data[x][y][z] = &lin_data[index];
           }
       }
   }
     read_data_rank0(file1, nx, ny, nz, nc, &data);
     // now can use data[x][y][z][t] to do things
     // distribute_data(rank, size, nx, ny, nz, nc, &data, &local_data, px, py, pz);
 }
 int lx=nx/px;int ly=ny/py;int lz=nz/pz;
 float* lin_data2=malloc(sizeof(float)*(lx+2)*(ly+2)*(lz+2)*nc);
 local_data=(float ****)malloc((lx+2) * sizeof(float ***));
 for (int x = 0; x < lx+2; x++) {
       local_data[x] = (float ***)malloc((ly+2) * sizeof(float **));
       for (int y = 0; y < ly+2; y++) {
           local_data[x][y] = (float **)malloc((lz+2) * sizeof(float *));
           for (int z = 0; z < lz+2; z++) {
               // Calculate the index in the 1D array
               int index = x * (ly+2) * (lz+2) * nc + y * (lz+2) * nc + z * nc;
               local_data[x][y][z] = &lin_data2[index];
           }
       }
   }
//   allocate_space_local_data(lx+2,ly+2,lz+2,nc,&local_data);
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
 double time2=MPI_Wtime();
 // printf("Done\n");
 // for(int x=0;x<lx;x++){
 //  for(int y=0;y<ly;y++){
 //      for(int z=0;z<lz;z++){
 //          printf("Process %d: %d %d %d  is %f\n",myrank,x,y,z,local_data[x+1][y+1][z+1][0]);
 //      }
 //  }
 // }
 int array_of_sizes[4]={lx+2,ly+2,lz+2,nc};
 int array_of_subsizes_XY[4]={lx,ly,1,nc};
 int array_of_subsizes_XZ[4]={lx,1,lz,nc};
 int array_of_subsizes_YZ[4]={1,ly,lz,nc};








//    int array_of_starts_XY_low[4]={1,1,1,0};
 int array_of_starts_XY_send_high[4]={1,1,lz,0};
//    int array_of_starts_XZ_low[4]={1,1,1,0};
 int array_of_starts_XZ_send_high[4]={1,ly,1,0};
//    int array_of_starts_YZ_low[4]={1,1,1,0};
 int array_of_starts_YZ_send_high[4]={lx,1,1,0};
 int array_of_starts_XY_recv_low[4]={1,1,0,0};
 int array_of_starts_XZ_recv_low[4]={1,0,1,0};
 int array_of_starts_YZ_recv_low[4]={0,1,1,0};
 int array_of_starts_XY_recv_high[4]={1,1,lz+1,0};
 int array_of_starts_XZ_recv_high[4]={1,ly+1,1,0};
 int array_of_starts_YZ_recv_high[4]={lx+1,1,1,0};


// if(myrank==0){
//     for(int x=0;x<nx;x++){
//         for(int y=0;y<ny;y++){
//             for(int z=0;z<nz;z++){
//                 for(int c=0;c<nc;c++) printf("%d %d %d %d %p\n",x,y,z,c,&data[x][y][z][c]);
//             }
//         }
//     }
// }










 int array_of_starts_send_low[4]={1,1,1,0};
















   int array_of_starts_XY[4]={0,0,0,0};
   int array_of_starts_XZ[4]={0,0,0,0};
   int array_of_starts_YZ[4]={0,0,0,0};
 MPI_Datatype plane_XY,plane_XZ,plane_YZ;
//  MPI_Datatype plane_XY_send_low,plane_XZ_send_low,plane_YZ_send_low;
//  MPI_Datatype plane_XY_send_high,plane_XZ_send_high,plane_YZ_send_high;
//  MPI_Datatype plane_XY_recv_low,plane_XZ_recv_low,plane_YZ_recv_low;
//  MPI_Datatype plane_XY_recv_high,plane_XZ_recv_high,plane_YZ_recv_high;
//  MPI_Type_create_subarray(4,array_of_sizes,array_of_subsizes_XY,array_of_starts_send_low,MPI_ORDER_C, MPI_FLOAT,&plane_XY_send_low);
//  MPI_Type_commit(&plane_XY_send_low);
//  MPI_Type_create_subarray(4,array_of_sizes,array_of_subsizes_XZ,array_of_starts_send_low,MPI_ORDER_C, MPI_FLOAT,&plane_XZ_send_low);
//  MPI_Type_commit(&plane_XZ_send_low);
//  MPI_Type_create_subarray(4,array_of_sizes,array_of_subsizes_YZ,array_of_starts_send_low,MPI_ORDER_C, MPI_FLOAT,&plane_YZ_send_low);
//  MPI_Type_commit(&plane_YZ_send_low);
//  MPI_Type_create_subarray(4,array_of_sizes,array_of_subsizes_XY,array_of_starts_XY_recv_low,MPI_ORDER_C, MPI_FLOAT,&plane_XY_recv_low);
//  MPI_Type_commit(&plane_XY_recv_low);
//  MPI_Type_create_subarray(4,array_of_sizes,array_of_subsizes_XZ,array_of_starts_XZ_recv_low,MPI_ORDER_C, MPI_FLOAT,&plane_XZ_recv_low);
//  MPI_Type_commit(&plane_XZ_recv_low);
//  MPI_Type_create_subarray(4,array_of_sizes,array_of_subsizes_YZ,array_of_starts_YZ_recv_low,MPI_ORDER_C, MPI_FLOAT,&plane_YZ_recv_low);
//  MPI_Type_commit(&plane_YZ_recv_low);
//  MPI_Type_create_subarray(4, array_of_sizes, array_of_subsizes_XY, array_of_starts_XY_send_high, MPI_ORDER_C, MPI_FLOAT, &plane_XY_send_high);
//  MPI_Type_commit(&plane_XY_send_high);
//  MPI_Type_create_subarray(4, array_of_sizes, array_of_subsizes_XZ, array_of_starts_XZ_send_high, MPI_ORDER_C, MPI_FLOAT, &plane_XZ_send_high);
//  MPI_Type_commit(&plane_XZ_send_high);
//  MPI_Type_create_subarray(4, array_of_sizes, array_of_subsizes_YZ, array_of_starts_YZ_send_high, MPI_ORDER_C, MPI_FLOAT, &plane_YZ_send_high);
//  MPI_Type_commit(&plane_YZ_send_high);
//  MPI_Type_create_subarray(4, array_of_sizes, array_of_subsizes_XY, array_of_starts_XY_recv_high, MPI_ORDER_C, MPI_FLOAT, &plane_XY_recv_high);
//  MPI_Type_commit(&plane_XY_recv_high);
//  MPI_Type_create_subarray(4, array_of_sizes, array_of_subsizes_XZ, array_of_starts_XZ_recv_high, MPI_ORDER_C, MPI_FLOAT, &plane_XZ_recv_high);
//  MPI_Type_commit(&plane_XZ_recv_high);
//  MPI_Type_create_subarray(4, array_of_sizes, array_of_subsizes_YZ, array_of_starts_YZ_recv_high, MPI_ORDER_C, MPI_FLOAT, &plane_YZ_recv_high);
//  MPI_Type_commit(&plane_YZ_recv_high);
   MPI_Type_create_subarray(4,array_of_sizes,array_of_subsizes_YZ,array_of_starts_YZ,MPI_ORDER_C, MPI_FLOAT,&plane_YZ);
   MPI_Type_commit(&plane_YZ);
   MPI_Type_create_subarray(4,array_of_sizes,array_of_subsizes_XZ,array_of_starts_XZ,MPI_ORDER_C, MPI_FLOAT,&plane_XZ);
   MPI_Type_commit(&plane_XZ);
   MPI_Type_create_subarray(4,array_of_sizes,array_of_subsizes_XY,array_of_starts_XY,MPI_ORDER_C, MPI_FLOAT,&plane_XY);
   MPI_Type_commit(&plane_XY);
 int process_x_coordinate,process_y_coordinate,process_z_coordinate;
 process_x_coordinate=myrank/(py*pz);
 process_y_coordinate=(myrank/pz)%py;
 process_z_coordinate=(myrank%pz);
 int x_low=(process_x_coordinate>0)?myrank-(py*pz):MPI_PROC_NULL;
 int x_high= (process_x_coordinate<px-1) ? myrank + (py * pz) : MPI_PROC_NULL;
 int y_low=(process_y_coordinate>0)?myrank-pz: MPI_PROC_NULL;
 int y_high=(process_y_coordinate<py-1)?myrank+pz: MPI_PROC_NULL;
 int z_low=(process_z_coordinate>0)?myrank-1: MPI_PROC_NULL;
 int z_high=(process_z_coordinate<pz-1)?myrank+1: MPI_PROC_NULL;
//   printf("%d %d\n",z_low,myrank);
 MPI_Request send_request_x[2], recv_request_x[2],send_request_y[2],recv_request_y[2],send_request_z[2],recv_request_z[2];
 MPI_Status recv_status_x[2],recv_status_y[2],recv_status_z[2];




//  if(x_high!=MPI_PROC_NULL){
//      MPI_Isend(&local_data[0][0][0][0],1,plane_YZ_send_high,x_high,0,MPI_COMM_WORLD,&send_request_x[1]);
//      MPI_Irecv(&local_data[0][0][0][0],1,plane_YZ_recv_high,x_high,1,MPI_COMM_WORLD,&recv_request_x[1]);
//  }
//  if(x_low!=MPI_PROC_NULL){
//      MPI_Irecv(&local_data[0][0][0][0],1,plane_YZ_recv_low,x_low,0,MPI_COMM_WORLD,&recv_request_x[0]);
//      MPI_Isend(&local_data[0][0][0][0],1,plane_YZ_send_low,x_low,1,MPI_COMM_WORLD,&send_request_x[0]);
//  }
//  if(y_high!=MPI_PROC_NULL){
//      MPI_Isend(&local_data[0][0][0][0],1,plane_XZ_send_high,y_high,0,MPI_COMM_WORLD,&send_request_y[1]);
//      MPI_Irecv(&local_data[0][0][0][0],1,plane_XZ_recv_high,y_high,1,MPI_COMM_WORLD,&recv_request_y[1]);
//  }
//  if(y_low!=MPI_PROC_NULL){
//      MPI_Irecv(&local_data[0][0][0][0],1,plane_XZ_recv_low,y_low,0,MPI_COMM_WORLD,&recv_request_y[0]);
//      MPI_Isend(&local_data[0][0][0][0],1,plane_XZ_send_low,y_low,1,MPI_COMM_WORLD,&send_request_y[0]);
//  }
//  if(z_high!=MPI_PROC_NULL){
//      MPI_Isend(&local_data[0][0][0][0],1,plane_XY_send_high,z_high,0,MPI_COMM_WORLD,&send_request_z[1]);
//      MPI_Irecv(&local_data[0][0][0][0],1,plane_XY_recv_high,z_high,1,MPI_COMM_WORLD,&recv_request_z[1]);
//  }
//  if(z_low!=MPI_PROC_NULL){
//      MPI_Irecv(&local_data[0][0][0][0],1,plane_XY_recv_low,z_low,0,MPI_COMM_WORLD,&recv_request_z[0]);
//      MPI_Isend(&local_data[0][0][0][0],1,plane_XY_send_low,z_low,1,MPI_COMM_WORLD,&send_request_z[0]);
//  }
   if(x_high!=MPI_PROC_NULL){
       MPI_Isend(&local_data[lx][1][1][0],1,plane_YZ,x_high,0,MPI_COMM_WORLD,&send_request_x[1]);
       MPI_Irecv(&local_data[lx+1][1][1][0],1,plane_YZ,x_high,1,MPI_COMM_WORLD,&recv_request_x[1]);
   }
   if(x_low!=MPI_PROC_NULL){
       MPI_Irecv(&local_data[0][1][1][0],1,plane_YZ,x_low,0,MPI_COMM_WORLD,&recv_request_x[0]);
       MPI_Isend(&local_data[1][1][1][0],1,plane_YZ,x_low,1,MPI_COMM_WORLD,&send_request_x[0]);
   }     
   if(y_high!=MPI_PROC_NULL){
       MPI_Isend(&local_data[1][ly][1][0],1,plane_XZ,y_high,0,MPI_COMM_WORLD,&send_request_y[1]);
       MPI_Irecv(&local_data[1][ly+1][1][0],1,plane_XZ,y_high,1,MPI_COMM_WORLD,&recv_request_y[1]);
   }
   if(y_low!=MPI_PROC_NULL){
       MPI_Irecv(&local_data[1][0][1][0],1,plane_XZ,y_low,0,MPI_COMM_WORLD,&recv_request_y[0]);
       MPI_Isend(&local_data[1][1][1][0],1,plane_XZ,y_low,1,MPI_COMM_WORLD,&send_request_y[0]);
   }
   if(z_high!=MPI_PROC_NULL){
       MPI_Isend(&local_data[1][1][lz][0],1,plane_XY,z_high,0,MPI_COMM_WORLD,&send_request_z[1]);
       MPI_Irecv(&local_data[1][1][lz+1][0],1,plane_XY,z_high,1,MPI_COMM_WORLD,&recv_request_z[1]);
   }
   if(z_low!=MPI_PROC_NULL){
       MPI_Irecv(&local_data[1][1][0][0],1,plane_XY,z_low,0,MPI_COMM_WORLD,&recv_request_z[0]);
       MPI_Isend(&local_data[1][1][1][0],1,plane_XY,z_low,1,MPI_COMM_WORLD,&send_request_z[0]);
   }
 if(x_high!=MPI_PROC_NULL) {
     MPI_Wait(&recv_request_x[1], MPI_STATUS_IGNORE);
     MPI_Wait(&send_request_x[1], MPI_STATUS_IGNORE);
 }
 if(x_low!=MPI_PROC_NULL) {
     MPI_Wait(&recv_request_x[0], MPI_STATUS_IGNORE);
     MPI_Wait(&send_request_x[0], MPI_STATUS_IGNORE);
 }
 if(y_high!=MPI_PROC_NULL) {
     MPI_Wait(&recv_request_y[1], MPI_STATUS_IGNORE);
     MPI_Wait(&send_request_y[1], MPI_STATUS_IGNORE);
 }
 if(y_low!=MPI_PROC_NULL) {
     MPI_Wait(&recv_request_y[0], MPI_STATUS_IGNORE);
     MPI_Wait(&send_request_y[0], MPI_STATUS_IGNORE);
 }
 if(z_high!=MPI_PROC_NULL) {
     MPI_Wait(&recv_request_z[1], MPI_STATUS_IGNORE);
     MPI_Wait(&send_request_z[1], MPI_STATUS_IGNORE);
 }
 if(z_low!=MPI_PROC_NULL) {
     MPI_Wait(&recv_request_z[0], MPI_STATUS_IGNORE);
     MPI_Wait(&send_request_z[0], MPI_STATUS_IGNORE);
 }
//   if(myrank==0){
//     for(int i=0;i<9;i++) printf("temp %d %f: \n",i,temp[i]);
//   }
//   if(myrank==0){
//       for(int x=0;x<lx+2;x++){
//           for(int y=0;y<ly+2;y++){
//               for(int z=0;z<lz+2;z++){
//                   for(int c=0;c<nc;c++) printf("Process: %d coor %d %d %d %d is %lf\n",myrank,x,y,z,c,local_data[x][y][z][c]);
//               }
//           }
//       }
//   }
//   printf("%d share done\n",myrank);
 long long int* local_minima_count=malloc(sizeof(long long int)*nc);
 long long int* local_maxima_count=malloc(sizeof(long long int)*nc);
 float* sub_global_minima=malloc(sizeof(float)*nc);
 float* sub_global_maxima=malloc(sizeof(float)*nc);
 for(int c=0;c<nc;c++){
     sub_global_maxima[c]=local_data[1][1][1][c];
     sub_global_minima[c]=local_data[1][1][1][c];
     local_minima_count[c]=0;
     local_maxima_count[c]=0;
     for(int x=1;x<lx+1;x++){
         for(int y=1;y<ly+1;y++){
             for(int z=1;z<lz+1;z++){             
                 bool mini=true,maxi=true;
                 sub_global_maxima[c]=max(sub_global_maxima[c],local_data[x][y][z][c]);
                 sub_global_minima[c]=min(sub_global_minima[c],local_data[x][y][z][c]);
                 if(x>1||(x_low!=MPI_PROC_NULL)){
                     if(local_data[x-1][y][z][c]>=local_data[x][y][z][c]) maxi=false;
                     if(local_data[x-1][y][z][c]<=local_data[x][y][z][c]) mini=false;                     
                 }
                 if(x<lx||(x_high!=MPI_PROC_NULL)){
                     if(local_data[x+1][y][z][c]>=local_data[x][y][z][c]) maxi=false;
                     if(local_data[x+1][y][z][c]<=local_data[x][y][z][c]) mini=false;                     
                 }
                 if(y>1||(y_low!=MPI_PROC_NULL)){
                     if(local_data[x][y-1][z][c]>=local_data[x][y][z][c]) maxi=false;
                     if(local_data[x][y-1][z][c]<=local_data[x][y][z][c]) mini=false;                     
                 }
                 if(y<ly||(y_high!=MPI_PROC_NULL)){
                     if(local_data[x][y+1][z][c]>=local_data[x][y][z][c]) maxi=false;
                     if(local_data[x][y+1][z][c]<=local_data[x][y][z][c]) mini=false;                     
                 }
                 if(z>1||(z_low!=MPI_PROC_NULL)){
                     if(local_data[x][y][z-1][c]>=local_data[x][y][z][c]) maxi=false;
                     if(local_data[x][y][z-1][c]<=local_data[x][y][z][c]) mini=false;                     
                 }
                 if(z<lz||(z_high!=MPI_PROC_NULL)){
                     if(local_data[x][y][z+1][c]>=local_data[x][y][z][c]) maxi=false;
                     if(local_data[x][y][z+1][c]<=local_data[x][y][z][c]) mini=false;                     
                 }
                 if(maxi) local_maxima_count[c]++;
                 if(mini) local_minima_count[c]++;
             }
         }
     }
 }
 long long int* local_minima_total=malloc(sizeof(long long int)*nc);
 long long int* local_maxima_total=malloc(sizeof(long long int)*nc);
 float* global_minima=malloc(sizeof(float)*nc);
 float* global_maxima=malloc(sizeof(float)*nc);
 MPI_Reduce(local_maxima_count,local_maxima_total,nc,MPI_LONG_LONG_INT,MPI_SUM,0,MPI_COMM_WORLD);
 MPI_Reduce(local_minima_count,local_minima_total,nc,MPI_LONG_LONG_INT,MPI_SUM,0,MPI_COMM_WORLD);
 MPI_Reduce(sub_global_maxima,global_maxima,nc,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
 MPI_Reduce(sub_global_minima,global_minima,nc,MPI_FLOAT,MPI_MIN,0,MPI_COMM_WORLD);
 if(myrank==0){
     for(int i=0;i<nc;i++) printf("{%lld, %lld}, ",local_minima_total[i],local_maxima_total[i]);
     printf("\n");
     for(int i=0;i<nc;i++) printf("{%f, %f}, ",global_minima[i],global_maxima[i]);
     printf("\n");
 }
 // pro
 int sTime=MPI_Wtime();
//  MPI_Type_free(&plane_XY_send_low);
//  MPI_Type_free(&plane_XZ_send_low);
//  MPI_Type_free(&plane_YZ_send_low);
//  MPI_Type_free(&plane_XY_recv_low);
//  MPI_Type_free(&plane_XZ_recv_low);
//  MPI_Type_free(&plane_YZ_recv_low);
//  MPI_Type_free(&plane_XY_send_high);
//  MPI_Type_free(&plane_XZ_send_high);
//  MPI_Type_free(&plane_YZ_send_high);
//  MPI_Type_free(&plane_XY_recv_high);
//  MPI_Type_free(&plane_XZ_recv_high);
//  MPI_Type_free(&plane_YZ_recv_high);
   MPI_Type_free(&plane_YZ);
   MPI_Type_free(&plane_XZ);
   MPI_Type_free(&plane_XY);












 MPI_Finalize();
 return 0;
}



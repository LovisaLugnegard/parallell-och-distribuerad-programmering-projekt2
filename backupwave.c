/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <mpi.h>

#define WRITE_TO_FILE 
/*#define VERIFY*/ 

double timer();
double initialize(double x, double y, double t);
void save_solution(double *u, int Ny, int Nx, int n);
int rank, nprocs, sqnprocs, nproc_row, nproc_col;

int main(int argc, char *argv[])
{

  MPI_Init(&argc,&argv);

  int n,Nx,Ny,Nt,n_local_rows,n_local_columns,i,j,halo_size,u_size_local,blocklength,stride,count,bonk2;
  double dt, dx, lambda_sq;
  // double *u; //,*u_local;
  double *u_old;//, *u_old_local;
  double *u_new;//,*u_new_local;
  double begin,end;
  int source, dest1, dest2, dest3, dest4;
  MPI_Datatype halo_row, halo_col, local_row;


  Nx=128; //ändrade till 8 för att kunna se vad som händer, är 128 i koden från uppgiften
  if(argc>1)
    Nx=atoi(argv[1]);
  Ny=Nx;
  Nt=Nx;
  n = Nx;
  dx=1.0/(Nx-1);
  dt=0.50*dx;
  lambda_sq = (dt/dx)*(dt/dx);

 
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  sqnprocs = sqrt(nprocs);

  int row_rank,col_rank,n_dims,reorder;
  int dims[2], coords[2], cyclic[2];
  MPI_Comm proc_grid,proc_row,proc_col;
  n_dims = 2;
  reorder = 1;
  dims[0] = sqnprocs;
  dims[1] = sqnprocs;
  cyclic[0] = 0;
  cyclic[1] = 0;

  
  MPI_Cart_create(MPI_COMM_WORLD,n_dims,dims,cyclic,reorder,&proc_grid);
  MPI_Comm_rank(proc_grid,&rank);
  MPI_Cart_coords(proc_grid,rank,n_dims,coords);
  MPI_Comm_split(proc_grid,coords[0],coords[1],&proc_row);
  MPI_Comm_rank(proc_row,&row_rank);
  MPI_Comm_split(proc_grid,coords[1],coords[0],&proc_col);
  MPI_Comm_rank(proc_col,&col_rank);
  MPI_Comm_size(proc_row,&nproc_row);
  MPI_Comm_size(proc_col,&nproc_col);

  MPI_Request request[nprocs];
  MPI_Status status[nprocs];
  MPI_Datatype strided;

  n_local_rows = Ny/nprocs;
  n_local_columns = Nx/nprocs;
  u_size_local = (n_local_columns)*(n_local_rows); //inte tillräckligt stor för halo
  halo_size = n_local_rows + n_local_columns -1;


  //u_old_local = malloc((u_size_local)*sizeof(double));
  //u_new_local = malloc((u_size_local)*sizeof(double));

  double u_local[u_size_local];
  double u_old_local[u_size_local];
  double u_new_local[u_size_local];

  /* Setup IC */
  memset(u_local,0,u_size_local*sizeof(double));
  memset(u_old_local,0,u_size_local*sizeof(double));
  memset(u_new_local,0,u_size_local*sizeof(double));
  stride=Nx;
  count=Nx/sqnprocs;
  blocklength=Nx/sqnprocs;
  double u[n*n];

/* #ifdef VERIFY */
/*     double max_error=0.0; */
/* #endif */

  if(rank==0){
    // u = malloc(Nx*Ny*sizeof(double));
    u_new = malloc(Nx*Ny*sizeof(double));
    memset(u,0,Nx*Ny*sizeof(double));
    memset(u_new,0,Nx*Ny*sizeof(double));
    for(int i = 1; i < (Ny-1); ++i) {
      for(int j = 1; j < (Nx-1); ++j) {
        double x = j*dx;
        double y = i*dx;

        /* u0 */
        u[i*Nx+j] = initialize(x,y,0);

        /* u1 */
        u_new[i*Nx+j] = initialize(x,y,dt);
        //   printf("unew %d",u_new[i*Nx+j]);
      }
    }
    printf("\n IC complete \n");
#ifdef WRITE_TO_FILE
    save_solution(u_new,Ny,Nx,1);
#endif
/* #ifdef VERIFY */
/*     double max_error=0.0; */
/* #endif */

    MPI_Type_vector(count,blocklength,stride,MPI_DOUBLE,&strided);  
    MPI_Type_commit(&strided);

 

    //distribute all grid partitions
    for(i=0; i<sqnprocs; i++) { 
      for(j=0; j<sqnprocs; j++){
        MPI_Cart_rank(proc_grid,coords,&rank);
        //HÄR VAR VI NÄR VI SLUTADE!!!

        MPI_Isend(&u_new[(Nx*Nx/nprocs)*2*i+Nx/sqnprocs*j],1,strided,(j+i*sqnprocs),1,MPI_COMM_WORLD,&request[i*sqnprocs+j]);
      }
    } 
  }

  MPI_Barrier(proc_grid);
  printf("\n after barrier \n");


  MPI_Recv(u_local,Nx*Nx/nprocs,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&status[rank]);

  MPI_Barrier(proc_grid);

  MPI_Get_elements(&status[rank],MPI_DOUBLE,&bonk2);
  printf("proc %d recv %d elements of type MPI_DOUBLE\n",rank,bonk2);
  for (i=0;i<bonk2;i++) 
    printf("proc: %d  %g\n",rank,u_local[i]);


  //nu är alla u_local uppdaterade, nu behöver vi skicka halopunkter, här tror jag att det är en bra idé att använda sendrecv kolonn och radvis
  //y-led
  //memcpy(u_local,(double[25]){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0,0,0,0,0,0,0,0,0,0,0},sizeof(double[25]));
  printf("row_rank: %d col_rank: %d rank: %d \n", row_rank, col_rank, rank);
  MPI_Barrier(proc_grid);
  /* for(i = 1; i<= 16; i++){ */
  /*   u_local[i-1] = i*1.0; */
   
  /* } */


  /* for(i=0; i < 16;i++){ */
  /*   printf(" %g ",u_local[i]); */
  /* }  */ 

  MPI_Type_vector(1,n/sqnprocs,n/sqnprocs,MPI_DOUBLE,&halo_row);
  MPI_Type_commit(&halo_row);
  MPI_Type_vector(n/sqnprocs,1,n/sqnprocs,MPI_DOUBLE,&halo_col);
  MPI_Type_commit(&halo_col);

  //HALOROWTEST
  double halo_data_upper[n/sqnprocs];
  double halo_data_lower[n/sqnprocs];
  double halo_data_left[n/sqnprocs];
  double halo_data_right[n/sqnprocs];


  MPI_Cart_shift(proc_grid, 0, 1, &source, &dest1);
  printf( "rank: %d source: %d dest %d\n",rank,source, dest1);
  if(dest1 != MPI_PROC_NULL){
    MPI_Sendrecv(&u_local[n*n/nprocs - n/sqnprocs],1,halo_row,dest1,10,halo_data_lower, n/sqnprocs,MPI_DOUBLE,dest1,20,proc_grid,&status[rank]);
    printf("I, %d, sent lower halo data to %d \n",rank,dest1);
  }

  // MPI_Barrier(proc_grid);

  MPI_Cart_shift(proc_grid, 0, -1, &source, &dest2);
  if(dest2 != MPI_PROC_NULL){
    MPI_Sendrecv(&u_local[0],1,halo_row,dest2,20,halo_data_upper, n/sqnprocs,MPI_DOUBLE,dest2,10,proc_grid,&status[rank]);
    printf("I SENDRECIEVED %d \n",rank);
    for (i=0;i<n/sqnprocs;i++)
      printf("Halodata upper: %d  %g\n",rank,halo_data_upper[i]);
  }
  // MPI_Barrier(proc_grid);
  MPI_Cart_shift(proc_grid, 1, 1, &source, &dest3);
  printf( "rank: %d source: %d dest %d\n",rank,source, dest3);
  if(dest3 != MPI_PROC_NULL){
    MPI_Sendrecv(&u_local[n/sqnprocs-1],1,halo_col,dest3,30,halo_data_right, n/sqnprocs,MPI_DOUBLE,dest3,40,proc_grid,&status[rank]);
    printf("I SENDRECIEVED %d \n",rank);
  }
  // MPI_Barrier(proc_grid);
  MPI_Cart_shift(proc_grid, 1, -1, &source, &dest4);
  if(dest4 != MPI_PROC_NULL){
    MPI_Sendrecv(&u_local[0],1,halo_col,dest4,40,halo_data_left, n/sqnprocs,MPI_DOUBLE,dest4,30,proc_grid,&status[rank]);
    printf("I SENDRECIEVED %d \n",rank);
  }
  MPI_Barrier(proc_grid);
  for (i=0;i<n/sqnprocs;i++)
    printf("Halodata left: %d  %g\n",rank,halo_data_left[i]);




  // MPI_Sendrecv(
  //  MPI_Finalize();
  /* MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, */
  /*              int dest, int sendtag, */
  /*              void *recvbuf, int recvcount, MPI_Datatype recvtype, */
  /*              int source, int recvtag, */
  /*              MPI_Comm comm, MPI_Status *status) */

  //bortkommenterat bara för att slippa hålla på med det just nu

  MPI_Barrier(proc_grid);

  /* /\* Integrate *\/ */

  begin=timer(); 
  for(int n=2; n<Nt; ++n) { 
    /*     /\* Swap ptrs *\/ */
    double tmp[sizeof(u_local)];
    memcpy(tmp,u_old_local,sizeof(u_local));
    memcpy(u_old_local,u_local,sizeof(u_local));
    memcpy(u_local,u_new_local,sizeof(u_local));
    memcpy(u_new_local,tmp,sizeof(u_local)); 
    // u_old_local = u_local; 
    //u_local = u_new_local; 
    //u_new_local = tmp; 




    /*     /\* Apply stencil *\/ */
    for(int i = 1; i < (n/sqnprocs-1); ++i) { 
      for(int j = 1; j < (n/sqnprocs-1); ++j) { 

        u_new_local[i*n/sqnprocs+j] = 2*u_local[i*n/sqnprocs+j]-u_old_local[i*n/sqnprocs+j]+lambda_sq* 
          (u_local[(i+1)*n/sqnprocs+j] + u_local[(i-1)*n/sqnprocs+j] + u_local[i*n/sqnprocs+j+1] + u_local[i*n/sqnprocs+j-1] -4*u_local[i*n/sqnprocs+j]); 
      } 
    }
    //Do manual computation of EOL etc

    if(dest3!= MPI_PROC_NULL){
      printf("%d in dest3 not null \n", rank);
      //räkna med halopunkter till höger (obs ej hörnpunkt)

      for(int i = 1; i < (n/sqnprocs-1); ++i) { 

        u_new_local[i*n/sqnprocs-1] = 2*u_local[i*n/sqnprocs-1]-u_old_local[i*n/sqnprocs-1]+lambda_sq* 
          (u_local[(i+1)*n/sqnprocs-1] + u_local[(i-1)*n/sqnprocs-1] +halo_data_right[i] + u_local[i*n/sqnprocs-2] -4*u_local[i*n/sqnprocs-1]); 
      } 
       
    }

    if(dest1!= MPI_PROC_NULL){
      printf("%d in dest1 not null \n", rank);
      //räkna med halopunkter nedåt  (obs ej hörnpunkt)

      for(int j = 1; j < (n/sqnprocs-1); ++j) { 

        u_new_local[n*n/nprocs- n/sqnprocs + j -1] = 2*u_local[n*n/nprocs- n/sqnprocs + j -1]-u_old_local[n*n/nprocs- n/sqnprocs + j -1]+lambda_sq* 
          (halo_data_lower[j] + u_local[n*n/nprocs-2* n/sqnprocs + j -1] +  u_local[n*n/nprocs- n/sqnprocs + j]  + u_local[n*n/nprocs- n/sqnprocs + j-2] -4*u_local[n*n/nprocs- n/sqnprocs + j -1]); 
      } 
       
    }



    if(dest2!= MPI_PROC_NULL){
      //räkna med halopunkter uppåt (obs ej hörnpunkt)
      printf("%d in dest2 not null \n", rank);
      for(int j = 1; j < (n/sqnprocs-1); ++j) { 

        u_new_local[j] = 2*u_local[j]-u_old_local[j]+lambda_sq* 
          (halo_data_upper[j] + u_local[j +n/sqnprocs] +  u_local[j+1]  + u_local[j-1] -4*u_local[j]); 
      } 
       
    }


    if(dest4!= MPI_PROC_NULL){
      printf("%d in dest4 not null \n", rank);
      //räkna med halopunkter åt vänster (obs ej hörnpunkt)

      for(int i = 1; i < (n/sqnprocs-1); ++i) { 

        u_new_local[i*n/sqnprocs] = 2*u_local[i*n/sqnprocs]-u_old_local[i*n/sqnprocs]+lambda_sq* 
          (u_local[(i-1)*n/sqnprocs] + u_local[(i+1)*n/sqnprocs] +  u_local[i*n/sqnprocs+1]  + halo_data_left[i] -4*u_local[i*n/sqnprocs]); 
      } 
       
    }


    //måste göra  alla hörn!

    //hörn nere till höger
    if(dest3!= MPI_PROC_NULL && dest1!= MPI_PROC_NULL){
      u_new_local[n*n/nprocs -1 ] = 2*u_local[n*n/sqnprocs]-u_old_local[n*n/sqnprocs]+lambda_sq* 
        (u_local[n*n/nprocs- n/sqnprocs-1] + halo_data_lower[n/sqnprocs-1] +  halo_data_right[n/sqnprocs-1]  + u_local[n*n/nprocs-2] -4*u_local[n*n/nprocs -1]); 
    }
    else{
      u_new_local[n*n/nprocs -1]=0;
    }

    //hör nere till vänster
    if(dest1!= MPI_PROC_NULL && dest4!= MPI_PROC_NULL){
      u_new_local[n*n/nprocs-n/sqnprocs] = 2*u_local[n*n/nprocs-n/sqnprocs]-u_old_local[n*n/sqnprocs-n/sqnprocs]+lambda_sq* 
        (u_local[n*n/nprocs- 2*n/sqnprocs] + halo_data_lower[0] +  u_local[n*n/nprocs-n/sqnprocs+1]  + halo_data_left[n/sqnprocs-1] -4*u_local[n*n/nprocs-n/sqnprocs]); 
    }
    else{
      u_new_local[n*n/nprocs -n/sqnprocs]=0;
    }

    //hör uppe till höger
    if(dest2!= MPI_PROC_NULL && dest3!= MPI_PROC_NULL){
      u_new_local[n/sqnprocs-1] = 2*u_local[n/sqnprocs-1]-u_old_local[n/sqnprocs-1]+lambda_sq* 
        (halo_data_upper[n/sqnprocs-1] + u_local[2*n/sqnprocs-1] +  halo_data_right[0]  + u_local[n/sqnprocs-2] -4*u_local[n/sqnprocs-1]); 
    }
    else{
      u_new_local[n/sqnprocs -1]=0;
    }

    //hörn uppe till vänster
    if(dest2!=MPI_PROC_NULL && dest4!= MPI_PROC_NULL){
      u_new_local[0] = 2*u_local[0]-u_old_local[0]+lambda_sq* 
        (halo_data_upper[0] + u_local[n/sqnprocs] +  u_local[1]  + halo_data_left[0] -4*u_local[0]); 
    }
    else{
      u_new_local[0]=0;
    }


    MPI_Type_vector(n/sqnprocs,1,n/sqnprocs, MPI_DOUBLE, &local_row);
    MPI_Type_commit(&local_row);
    // Gather data in process 0 here!!
    for(i=0; i<nproc_row; ++i){
      if(coords[0]==i){
        for(j=0;j<n/sqnprocs;++j){
          MPI_Gather(&u_local[j*n/sqnprocs], 1, local_row, u, n/sqnprocs, MPI_DOUBLE, 0, proc_grid);
        }}
    }


#ifdef VERIFY 
    double error=0.0; 
    for(int i = 0; i < Ny; ++i) { 
      for(int j = 0; j < Nx; ++j) { 
        double e = fabs(u_new_local[i*Nx+j]-initialize(j*dx,i*dx,n*dt)); 
        if(e>error) 
          error = e; 
      } 
    } 
    if(error > max_error) 
      max_error=error; 
#endif 

#ifdef WRITE_TO_FILE 
    save_solution(u,Ny,Nx,n); 
#endif 

  } 
  end=timer(); 

  printf("Time elapsed: %g s\n",(end-begin)); 

#ifdef VERIFY 
  printf("Maximum error: %g\n",max_error); 
#endif 

  /* free(u);  */
  /* free(u_old);  */
  /* free(u_new);  */
    
  MPI_Finalize();   
  return 0; 

} 

double timer() 
{ 
  struct timeval tv; 
  gettimeofday(&tv, NULL); 
  double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000; 
  return seconds; 
} 

double initialize(double x, double y, double t)
{
  double value = 0;
#ifdef VERIFY
  /* standing wave */
  value=sin(3*M_PI*x)*sin(4*M_PI*y)*cos(5*M_PI*t);
#else
  /* squared-cosine hump */
  const double width=0.1;

  double centerx = 0.25;
  double centery = 0.5;

  double dist = sqrt((x-centerx)*(x-centerx) +
                     (y-centery)*(y-centery));
  if(dist < width) {
    double cs = cos(M_PI_2*dist/width);
    value = cs*cs;
  }
#endif
  return value;
}

void save_solution(double *u, int Ny, int Nx, int n)
{
  char fname[50];
  sprintf(fname,"solution-%d.dat",n);
  FILE *fp = fopen(fname,"w");

  fprintf(fp,"%d %d\n",Nx,Ny);

  for(int j = 0; j < Ny; ++j) {
    for(int k = 0; k < Nx; ++k) {
      fprintf(fp,"%e\n",u[j*Nx+k]);
    }
  }

  fclose(fp);
}

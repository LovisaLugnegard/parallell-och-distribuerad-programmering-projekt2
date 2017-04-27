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

#define WRITE_TO_FILE */
/* #define VERIFY */

double timer();
double initialize(double x, double y, double t);
void save_solution(double *u, int Ny, int Nx, int n);
int rank, nprocs, sqnprocs;

int main(int argc, char *argv[])
{

  MPI_Init(&argc,&argv);

  int Nx,Ny,Nt,n_local_rows,n_local_columns,i,j,halo_size,u_size_local;
  double dt, dx, lambda_sq;
  double *u,*u_local;
  double *u_old,*u_old_local;
  double *u_new,*u_new_local;
  double begin,end;

  Nx=128;
  if(argc>1)
    Nx=atoi(argv[1]);
  Ny=Nx;
  Nt=Nx;
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

  MPI_Request request[nprocs];
  MPI_Status status[nprocs];
  MPI_Datatype strided;

  n_local_rows = Ny/nprocs+1;
  n_local_columns = Nx/nprocs+1;
  u_size_local = (n_local_columns)*(n_local_rows);
  halo_size = n_local_rows + n_local_columns -1;

  if(rank==0){
    u = malloc(Nx*Ny*sizeof(double));
    u_new = malloc(Nx*Ny*sizeof(double));}
  u_old_local = malloc((u_size_local)*sizeof(double));
  u_new_local = malloc((u_size_local)*sizeof(double));
  u_local = malloc((u_size_local)*sizeof(double));

  /* Setup IC */
  if(rank==0){
    memset(u,0,Nx*Ny*sizeof(double));
    memset(u_new,0,Nx*Ny*sizeof(double));}
  memset(u_local,0,u_size_local*sizeof(double));
  memset(u_old_local,0,u_size_local*sizeof(double));
  memset(u_new_local,0,u_size_local*sizeof(double));

  if(rank==0){
    for(int i = 1; i < (Ny-1); ++i) {
      for(int j = 1; j < (Nx-1); ++j) {
        double x = j*dx;
        double y = i*dx;

        /* u0 */
        u[i*Nx+j] = initialize(x,y,0);

        /* u1 */
        u_new[i*Nx+j] = initialize(x,y,dt);
      }
    }
    printf("\n IC complete \n");
#ifdef WRITE_TO_FILE
    save_solution(u_new,Ny,Nx,1);
#endif
#ifdef VERIFY
    double max_error=0.0;
#endif

    MPI_Type_vector(Nx/sqnprocs,Nx/sqnprocs,Nx,MPI_DOUBLE,&strided);  
    MPI_Type_commit(&strided);

  //distribute all grid partitions

    for(i=0; i<sqnprocs; i++) {
      for(j=0; j<sqnprocs; j++){
        MPI_Cart_rank(proc_grid,coords,&rank);
        //HÄR VAR VI NÄR VI SLUTADE!!!
        MPI_Isend(&u_new[0],1,strided,(j+i*sqnprocs),1,proc_grid,&request[i*sqnprocs+j]);
        // MPI_Isend(&dims,1,MPI_INT,(j+i*sqnprocs),1,proc_grid,&request[i*sqnprocs +j]);
        printf("\n After Isend \n");
      }
    }
    MPI_Waitall(nprocs-1,request,status);
    printf("After waitall"); 
  }

  MPI_Barrier(proc_grid);
  printf("\n after barrier \n");
  //OBS ändra till Nx*Ny om de är olika stora
  MPI_Recv(u_local,Nx*Nx/nprocs,MPI_DOUBLE,0,1,proc_grid,&status[rank]);
  printf("\n After recieve \n");


  /* Integrate */

  begin=timer();
  for(int n=2; n<Nt; ++n) {
    /* Swap ptrs */
    double *tmp = u_old;
    u_old = u;
    u = u_new;
    u_new = tmp;

    /* Apply stencil */
    for(int i = 1; i < (Ny-1); ++i) {
      for(int j = 1; j < (Nx-1); ++j) {

        u_new[i*Nx+j] = 2*u[i*Nx+j]-u_old[i*Nx+j]+lambda_sq*
          (u[(i+1)*Nx+j] + u[(i-1)*Nx+j] + u[i*Nx+j+1] + u[i*Nx+j-1] -4*u[i*Nx+j]);
      }
    }

#ifdef VERIFY
    double error=0.0;
    for(int i = 0; i < Ny; ++i) {
      for(int j = 0; j < Nx; ++j) {
        double e = fabs(u_new[i*Nx+j]-initialize(j*dx,i*dx,n*dt));
        if(e>error)
          error = e;
      }
    }
    if(error > max_error)
      max_error=error;
#endif

#ifdef WRITE_TO_FILE
    save_solution(u_new,Ny,Nx,n);
#endif

  }
  end=timer();

  printf("Time elapsed: %g s\n",(end-begin));

#ifdef VERIFY
  printf("Maximum error: %g\n",max_error);
#endif

  free(u);
  free(u_old);
  free(u_new);

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

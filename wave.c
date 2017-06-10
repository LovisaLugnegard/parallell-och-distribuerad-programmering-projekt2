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

//#define WRITE_TO_FILE 
//#define VERIFY 

double timer();
double initialize(double x, double y, double t);
void save_solution(double *u, int Ny, int Nx, int n);
int rank, nprocs, sqnprocs, nproc_row, nproc_col;

int main(int argc, char *argv[])
{

  MPI_Init(&argc,&argv);

  int Nx,Ny,Nt,n_local_rows,n_local_columns,i,j,u_size_local,blocklength,stride,count;
  double dt, dx, lambda_sq;
  double *u, *u_local;
  double *u_old_local;
  double *u_new,*u_new_local;
  double begin,end;
  int source, dest1, dest2, dest3, dest4, rem_col, rem_row;
  MPI_Datatype halo_row, halo_col;


  Nx=128; //ändrade till 8 för att kunna se vad som händer, är 128 i koden från uppgiften
  if(argc>1)
    Nx=atoi(argv[1]);
  Ny=Nx;
  Nt=Nx;
  dx=1.0/(Nx-1);
  dt=0.50*dx;
  lambda_sq = (dt/dx)*(dt/dx);

#ifdef VERIFY 
  double *err_array;
  err_array = malloc(Nt*sizeof(double));
#endif
 
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  // sqnprocs = sqrt(nprocs);

  int row_rank,col_rank,n_dims,reorder;
  int  coords[2], cyclic[2],dims[2];
  MPI_Comm proc_grid,proc_row,proc_col;
  n_dims = 2;
  
  reorder = 1;
  dims[0] = 0;
  dims[1] = 0;
  cyclic[0] = 0;
  cyclic[1] = 0;

  MPI_Dims_create(nprocs,n_dims,dims);  
  MPI_Cart_create(MPI_COMM_WORLD,n_dims,dims,cyclic,reorder,&proc_grid);
  MPI_Comm_rank(proc_grid,&rank);
  MPI_Cart_coords(proc_grid,rank,n_dims,coords);
  MPI_Comm_split(proc_grid,coords[0],coords[1],&proc_row);
  MPI_Comm_rank(proc_row,&row_rank);
  MPI_Comm_split(proc_grid,coords[1],coords[0],&proc_col);
  MPI_Comm_rank(proc_col,&col_rank);
  MPI_Comm_size(proc_row,&nproc_row);
  MPI_Comm_size(proc_col,&nproc_col);

  MPI_Request request[nprocs],request1[nprocs];
  MPI_Status status[nprocs],status1[nprocs];
  MPI_Datatype strided;

  n_local_rows = Ny/nproc_row;
  n_local_columns = Nx/nproc_col;
  u_size_local = (n_local_columns)*(n_local_rows);
  u_local = malloc(2*(u_size_local)*sizeof(double));
  u_old_local = malloc(2*(u_size_local)*sizeof(double));
  u_new_local = malloc(2*(u_size_local)*sizeof(double));

  rem_row = 0;
  rem_col = 0;
  /* Detta var ett försök att implementera fallet då Nx ej är jämnt delbart med nproc_col /L 31/5 */
  if(nproc_col*n_local_columns != Nx){
    rem_col = Nx % nproc_col;
    printf("\n %d in first if rem=%d\n", rank,rem_col);
    for(i=0; i<rem_col; i++){
      if(rank == i){
        n_local_columns = n_local_columns+1;
        printf("\n %d  in if1.2, i=%d\n", rank,i);
      }
    }
  }

  if(nproc_row*n_local_rows != Ny){
    rem_row = Ny % nproc_row;
    printf("\n %d in second if, rem = %d\n", rank, rem_row);
    for(i=0; i<rem_row; i++){
      if(rank == i){
        n_local_rows = n_local_rows+1;
      }
    }
  }
  /*Här slutar försöket /L 31/5 */ 

  printf("\nrank %d nproc_row: %d nproc_col: %d n_local_rows: %d n_local_cols: %d row_rank: %d col_rank: %d\n",rank, nproc_row, nproc_col, n_local_rows, n_local_columns, coords[0], coords[1]);
  
  /* Setup IC */
  memset(u_local,0,u_size_local*sizeof(double));
  memset(u_old_local,0,u_size_local*sizeof(double));
  memset(u_new_local,0,u_size_local*sizeof(double));

  stride = Nx;
  count = n_local_rows;
  blocklength=n_local_columns;

  u = malloc((Nx*Ny+2*Nx)*sizeof(double));

  if(rank==0){
   
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
      }
    }

#ifdef WRITE_TO_FILE
    save_solution(u_new,Ny,Nx,1);
#endif

    MPI_Type_vector(count,blocklength,stride,MPI_DOUBLE,&strided);  
    MPI_Type_commit(&strided);

    //distribute all grid partitions 
    //Ändrade till j+i*nproc_col, blir rätt med distributionen då (dvs det funkar med olika många processer i x och y-led /L 31/5
    for(i=0; i<nproc_row; i++) {//process rows 
      for(j=0; j<nproc_col; j++){//process columns
        if(rem_col == 0 && rem_row == 0){ //Denna ifsats hör till försöket att implementera Nx%nproc_col !=0
          MPI_Isend(&u_new[j*n_local_columns + i*Nx*n_local_rows],1,strided,(j+i*nproc_col),1,MPI_COMM_WORLD,&request[j+i*nproc_col]);
          MPI_Isend(    &u[j*n_local_columns + i*Nx*n_local_rows],1,strided,(j+i*nproc_col),9,MPI_COMM_WORLD,&request1[j+i*nproc_col]);//[(Nx*Nx/nprocs)*2*i+Nx/sqnprocs*j]
          printf("\n 1 Did send to %d\n", j+i*nproc_col);
        }
        /*Allt detta hör till försöket att implementera fallet då Nx%nproc_col !=0 L 31/5*/
        else if(j < rem_col && rem_row == 0){
          MPI_Isend(&u_new[j*n_local_columns + i*Nx*n_local_rows],1,strided,(j+i*nproc_col),1,MPI_COMM_WORLD,&request[j+i*nproc_col]);
          MPI_Isend(    &u[j*n_local_columns + i*Nx*n_local_rows],1,strided,(j+i*nproc_col),9,MPI_COMM_WORLD,&request1[j+i*nproc_col]);//[(Nx*Nx/nprocs)*2*i+Nx/sqnprocs*j]
          printf("\n 2a Did send to %d\n", j+i*nproc_col);
        }

        else if(j >= rem_col && rem_row == 0){
          MPI_Datatype strided2;
          MPI_Type_vector(count,blocklength-1,stride,MPI_DOUBLE,&strided2);  
          MPI_Type_commit(&strided2);
          MPI_Isend(&u_new[j*n_local_columns + i*Nx*n_local_rows],1,strided2,(j+i*nproc_col),1,MPI_COMM_WORLD,&request[j+i*nproc_col]);
          MPI_Isend(    &u[j*n_local_columns + i*Nx*n_local_rows],1,strided2,(j+i*nproc_col),9,MPI_COMM_WORLD,&request1[j+i*nproc_col]);//[(Nx*Nx/nprocs)*2*i+Nx/sqnprocs*j]
          printf("\n 2b Did send to %d\n", j+i*nproc_col);
        }

        /* else if(j < rem_col && i <= rem_row){ */
        /* MPI_Isend(&u_new[j*n_local_columns + i*Nx*n_local_rows],1,strided,(j+i*nproc_col),1,MPI_COMM_WORLD,&request[j+i*nproc_col]); */
        /* MPI_Isend(    &u[j*n_local_columns + i*Nx*n_local_rows],1,strided,(j+i*nproc_col),9,MPI_COMM_WORLD,&request1[j+i*nproc_col]);//[(Nx*Nx/nprocs)*2*i+Nx/sqnprocs*j] */
        /*          printf("\n 2 Did send to %d\n", j+i*nproc_col); */
        /* } */
        /* else if(j<=rem_col && i > rem_row){ */
        /*   MPI_Isend(&u_new[j*n_local_columns + i*Nx*(n_local_rows-1)],1,strided,(j+i*nproc_col),1,MPI_COMM_WORLD,&request[j+i*nproc_col]); */
        /*   MPI_Isend(    &u[j*n_local_columns + i*Nx*(n_local_rows-1)],1,strided,(j+i*nproc_col),9,MPI_COMM_WORLD,&request1[j+i*nproc_col]);    */
        /*         printf("\n 3 Did send to %d\n", j+i*nproc_col); */
        /* } */
        /* else if(j>=rem_col && i<= rem_row){ */
        /*   MPI_Isend(&u_new[j*(n_local_columns-1) + i*Nx*n_local_rows],1,strided,(j+i*nproc_col),1,MPI_COMM_WORLD,&request[j+i*nproc_col]); */
        /*   MPI_Isend(    &u[j*(n_local_columns-1) + i*Nx*n_local_rows],1,strided,(j+i*nproc_col),9,MPI_COMM_WORLD,&request1[j+i*nproc_col]); */
        /*  printf("\n 4 Did send to %d\n", j+i*nproc_col);} */
        /* else{ */
        /*   MPI_Isend(&u_new[j*(n_local_columns-1) + i*Nx*(n_local_rows-1)],1,strided,(j+i*nproc_col),1,MPI_COMM_WORLD,&request[j+i*nproc_col]); */
        /*   MPI_Isend(    &u[j*(n_local_columns-1) + i*Nx*n_local_rows],1,strided,(j+i*nproc_col),9,MPI_COMM_WORLD,&request1[j+i*nproc_col]); */
        /*  printf("\n 5 Did send to %d\n", j+i*nproc_col); */
        /*      } */

        /*Här slutar det experimentet */

      }

    }
  } 
  //  }
#ifdef VERIFY
  double *max_error;
  max_error = malloc(sizeof(double));
  max_error[0] = 0.0;
#endif

  MPI_Recv(&u_new_local[0],u_size_local,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&status[rank]);
  printf("\n GOT HERE %d\n", rank);
  MPI_Recv(    &u_local[0],u_size_local,MPI_DOUBLE,0,9,MPI_COMM_WORLD,&status1[rank]);
 
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Type_vector(1,Nx/nproc_col,Nx/nproc_col,MPI_DOUBLE,&halo_row);
  MPI_Type_commit(&halo_row);
  MPI_Type_vector(Ny/nproc_row,1,Ny/nproc_row,MPI_DOUBLE,&halo_col);
  MPI_Type_commit(&halo_col);

  //HALO ROWS
  double halo_data_upper[Nx/nproc_col];
  double halo_data_lower[Nx/nproc_col];
  double halo_data_left[Ny/nproc_row];
  double halo_data_right[Ny/nproc_row];


  MPI_Cart_shift(proc_grid, 0, 1, &source, &dest1);
  if(dest1 != MPI_PROC_NULL){
    MPI_Sendrecv(&u_local[Nx*Ny/nprocs - Nx/nproc_col],1,halo_row,dest1,10,halo_data_lower, Nx/nproc_col,MPI_DOUBLE,dest1,20,proc_grid,&status[rank]);
  }

  MPI_Cart_shift(proc_grid, 0, -1, &source, &dest2);
  if(dest2 != MPI_PROC_NULL){
    MPI_Sendrecv(&u_local[0],1,halo_row,dest2,20,halo_data_upper, Nx/nproc_col,MPI_DOUBLE,dest2,10,proc_grid,&status[rank]);
  }

  MPI_Cart_shift(proc_grid, 1, 1, &source, &dest3);
  if(dest3 != MPI_PROC_NULL){
    MPI_Sendrecv(&u_local[Nx/nproc_col-1],1,halo_col,dest3,30,halo_data_right, Ny/nproc_row,MPI_DOUBLE,dest3,40,proc_grid,&status[rank]);
  }
 
  MPI_Cart_shift(proc_grid, 1, -1, &source, &dest4);
  if(dest4 != MPI_PROC_NULL){
    MPI_Sendrecv(&u_local[0],1,halo_col,dest4,40,halo_data_left, Ny/nproc_row,MPI_DOUBLE,dest4,30,proc_grid,&status[rank]);

  }
  MPI_Barrier(proc_grid);

  /* /\* Integrate *\/ */
  begin=timer(); 
  for(int n=2; n<Nt; ++n) { 
    /*     /\* Swap data in arrays *\/ */
    double *tmp;
    tmp = malloc((u_size_local)*sizeof(double));
    memcpy(tmp,u_local,(u_size_local)*sizeof(double));
    memcpy(u_local,u_new_local,(u_size_local)*sizeof(double));
    memcpy(u_old_local,tmp,(u_size_local)*sizeof(double));
    memset(u_new_local,0,u_size_local*sizeof(double));

    /*     /\* Apply stencil *\/ */
    for(int i = 1; i < (Ny/nproc_row-1); ++i) {//Rows 
      for(int j = 1; j < (Nx/nproc_col-1); ++j) { //Columns
        u_new_local[i*n_local_rows+j] = 2*u_local[i*n_local_rows+j]-u_old_local[i*n_local_rows+j]+lambda_sq* 
          (u_local[(i+1)*n_local_rows+j] + u_local[(i-1)*n_local_rows+j] + u_local[i*n_local_rows+j+1] + u_local[i*n_local_rows+j-1] -4*u_local[i*n_local_rows+j]);
 
      } 
    }
  
    //Do manual computation of EOL etc
    if(dest3!= MPI_PROC_NULL){
      //räkna med halopunkter till höger (obs ej hörnpunkt)
      for(int i = 1; i < (n_local_rows-1); ++i) { 
        u_new_local[(i+1)*n_local_columns-1] = 2*u_local[(i+1)*n_local_columns-1] - u_old_local[(i+1)*n_local_columns-1] + lambda_sq* 
          (u_local[(i+2)*n_local_columns-1] + u_local[i*n_local_columns-1] + halo_data_right[i] + u_local[(i+1)*n_local_columns-2] -4*u_local[(i+1)*n_local_columns-1]); 
      }        
    }

    if(dest1!= MPI_PROC_NULL){
      //räkna med halopunkter nedåt  (obs ej hörnpunkt)
      for(int j = 1; j < (n_local_columns-1); ++j) { 
        u_new_local[Nx*Ny/nprocs- n_local_columns + j] = 2*u_local[Nx*Ny/nprocs- n_local_columns + j] - u_old_local[Nx*Ny/nprocs- n_local_columns + j] + lambda_sq* 
          (halo_data_lower[j] + u_local[Nx*Ny/nprocs-2* n_local_columns + j] +  u_local[Nx*Ny/nprocs- n_local_columns + j+1]  + u_local[Nx*Ny/nprocs- n_local_columns + j-1] - 4*u_local[Nx*Ny/nprocs- n_local_columns + j]); 
      }        
    }

    if(dest2!= MPI_PROC_NULL){
      //räkna med halopunkter uppåt (obs ej hörnpunkt)
      for(int j = 1; j < (n_local_columns-1); ++j) { 
        u_new_local[j] = 2*u_local[j] - u_old_local[j] + lambda_sq* 
          (halo_data_upper[j] + u_local[j + n_local_columns] + u_local[j+1] + u_local[j-1] - 4*u_local[j]); 
      }       
    }

    if(dest4!= MPI_PROC_NULL){
      //räkna med halopunkter åt vänster (obs ej hörnpunkt)
      for(int i = 1; i < (n_local_rows-1); ++i) { 
        u_new_local[i*n_local_columns] = 2*u_local[i*n_local_columns]-u_old_local[i*n_local_columns]+lambda_sq* 
          (u_local[(i-1)*n_local_columns] + u_local[(i+1)*n_local_columns] +  u_local[i*n_local_columns+1] + halo_data_left[i] - 4*u_local[i*n_local_columns]); 
      }     
    }

    //hörn nere till höger
    if(dest3!= MPI_PROC_NULL && dest1!= MPI_PROC_NULL){
      u_new_local[Nx*Ny/nprocs - 1] = 2*u_local[Nx*Ny/nprocs - 1] - u_old_local[Nx*Ny/nprocs - 1] + lambda_sq* 
        (u_local[Nx*Ny/nprocs-1-n_local_columns] + halo_data_lower[n_local_columns-1] + halo_data_right[n_local_rows-1]  + u_local[Nx*Ny/nprocs - 2] -4*u_local[Nx*Ny/nprocs - 1]); 
    }
    else{
      u_new_local[Nx*Ny/nprocs -1]=0;
    }

    //hör nere till vänster
    if(dest1!= MPI_PROC_NULL && dest4!= MPI_PROC_NULL){
      u_new_local[Nx*Ny/nprocs-n_local_columns] = 2*u_local[Nx*Ny/nprocs-n_local_columns]-u_old_local[Nx*Ny/nprocs-n_local_columns]+lambda_sq* 
        (u_local[Nx*Ny/nprocs-2*n_local_columns] + halo_data_lower[0] +  u_local[Nx*Ny/nprocs-n_local_columns+1] + halo_data_left[n_local_rows-1] - 4*u_local[Nx*Ny/nprocs-n_local_columns]); 
    }
    else{
      u_new_local[Nx*Ny/nprocs-n_local_columns]=0;
    }

    //hör uppe till höger
    if(dest2!= MPI_PROC_NULL && dest3!= MPI_PROC_NULL){
      u_new_local[n_local_columns-1] = 2*u_local[n_local_columns-1]-u_old_local[n_local_columns-1]+lambda_sq* 
        (halo_data_upper[n_local_columns-1] + u_local[2*n_local_columns-1] +  halo_data_right[0]  + u_local[n_local_columns-2] - 4*u_local[n_local_columns-1]); 
    }
    else{
      u_new_local[n_local_columns-1]=0;
    }

    //hörn uppe till vänster
    if(dest2!=MPI_PROC_NULL && dest4!= MPI_PROC_NULL){
      u_new_local[0] = 2*u_local[0]-u_old_local[0]+lambda_sq* 
        (halo_data_upper[0] + u_local[n_local_columns] +  u_local[1]  + halo_data_left[0] -4*u_local[0]); 
    }
    else{
      u_new_local[0]=0;
    }

    MPI_Barrier(MPI_COMM_WORLD);


    MPI_Isend(u_new_local, Nx*Ny/nprocs, MPI_DOUBLE, 0, 4, proc_grid, &request[rank]);  //[C,double] 

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank==0){
      for (i=0; i<nprocs; i++) {
        MPI_Probe(i,4,proc_grid, &status[i]);
        MPI_Cart_coords(proc_grid, i,2,coords);      
        MPI_Recv(&u[coords[1]*n_local_columns + coords[0]*Nx*n_local_rows],1, strided, i,4,proc_grid, &status[i]); //[C,type]
      }
    }
    MPI_Wait(&request[rank],&status[rank]);


    /* #ifdef VERIFY */
    /*   MPI_Reduce(&max_error[0],&err_array[n],1,MPI_DOUBLE,MPI_MAX,0,proc_grid); */
    /* #endif */

    if(rank==0){
#ifdef WRITE_TO_FILE 
      save_solution(u,Ny,Nx,n); 
#endif 

#ifdef VERIFY 
      double error=0.0; 
      for(int i = 0; i < Ny; ++i) { 
        for(int j = 0; j < Nx; ++j) { 
          double e = fabs(u[i*Nx+j]-initialize(j*dx,i*dx,n*dt)); 
          if(e>error) 
            error = e; 
        } 
      } 
      if(error > max_error[0]) 
        max_error[0]=error;  
      // printf("Maximum error: %g\n",err_array[n]); 
      printf("Maximum error: %g\n",max_error[0]); 
#endif 
    }

    /* skicka halopunkter */

    MPI_Cart_shift(proc_grid, 0, 1, &source, &dest1);
    if(dest1 != MPI_PROC_NULL){
      MPI_Sendrecv(&u_new_local[Nx*Ny/nprocs - Nx/nproc_col],1,halo_row,dest1,10,halo_data_lower, Nx/nproc_col,MPI_DOUBLE,dest1,20,proc_grid,&status[rank]);
    }

    MPI_Cart_shift(proc_grid, 0, -1, &source, &dest2);
    if(dest2 != MPI_PROC_NULL){
      MPI_Sendrecv(&u_new_local[0],1,halo_row,dest2,20,halo_data_upper, Nx/nproc_col,MPI_DOUBLE,dest2,10,proc_grid,&status[rank]);
    }

    MPI_Cart_shift(proc_grid, 1, 1, &source, &dest3);
    if(dest3 != MPI_PROC_NULL){
      MPI_Sendrecv(&u_new_local[Nx/nproc_col-1],1,halo_col,dest3,30,halo_data_right, Ny/nproc_row,MPI_DOUBLE,dest3,40,proc_grid,&status[rank]);
    }
 

    MPI_Cart_shift(proc_grid, 1, -1, &source, &dest4);
    if(dest4 != MPI_PROC_NULL){
      MPI_Sendrecv(&u_new_local[0],1,halo_col,dest4,40,halo_data_left, Ny/nproc_row,MPI_DOUBLE,dest4,30,proc_grid,&status[rank]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  } 
  if(rank ==0){
    end=timer(); 
    printf("Time elapsed: %g s\n",(end-begin));
  } 
  free(u);
  free(u_old_local);
  free(u_new_local);
  free(u_local);
    
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

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

  int Nx,Ny,Nt,n_local_rows,n_local_columns,i,u_size_local;//,blocklength,stride,count;
  double dt, dx, lambda_sq;
  double *u_local, *u_old_local, *u_new_local;
  double begin,end;
  int source, dest1, dest2, dest3, dest4, rem_col, rem_row;
  MPI_Datatype halo_row, halo_col;


  Nx=128;
  if(argc>1)
    Nx=atoi(argv[1]);
  Ny=Nx;
  Nt=128;
  dx=1.0/(Nx-1);
  dt=0.50*dx;
  lambda_sq = (dt/dx)*(dt/dx);

#ifdef VERIFY 
  double *err_array;
  err_array = malloc(Nt*sizeof(double));
#endif
 
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

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
  MPI_Status status[nprocs];
  
  if(rank ==0)
    printf("For Nx = Ny = %d and Nt = %d, nprocs = %d\n",Nx,Nt,nprocs);
  n_local_rows = Ny/nproc_row;
  n_local_columns = Nx/nproc_col;
  
  rem_row = 0;
  rem_col = 0;

  /* Implementing if Nx is not divisible with nproc_col*/ 
  if(nproc_col*n_local_columns != Ny){
    rem_col = Ny % nproc_col;
    //printf("\n %d in first if rem=%d\n", rank,rem_col);
    for(i=0; i<rem_col; i++){
      if(col_rank == i){
        n_local_columns = n_local_columns+1;
        //  printf("\n %d  in if 1.2, i=%d\n", rank,i);
      }
    }
  }

  if(nproc_row*n_local_rows != Nx){
    rem_row = Nx % nproc_row;
    //printf("\n %d in second if, rem = %d\n", rank, rem_row);
    for(i=0; i<rem_row; i++){
      if(row_rank == i){
        n_local_rows = n_local_rows+1;
      }
    }
  }

  u_size_local = (n_local_columns)*(n_local_rows);
  u_local = malloc(2*(u_size_local)*sizeof(double));
  u_old_local = malloc(2*(u_size_local)*sizeof(double));
  u_new_local = malloc(2*(u_size_local)*sizeof(double));

  //printf("\nrank %d nproc_row: %d nproc_col: %d n_local_rows: %d n_local_cols: %d coords[0]: %d coords[1]: %d, row_rank: %d, col_rank: %d\n",rank, nproc_row, nproc_col, n_local_rows, n_local_columns, coords[0], coords[1], row_rank, col_rank);
 
  
  /* Setup IC */
  memset(u_local,0,u_size_local*sizeof(double));
  memset(u_old_local,0,u_size_local*sizeof(double));
  memset(u_new_local,0,u_size_local*sizeof(double));

  //printf(" rank %d  y-offset test  %d ",rank,coords[1]*Ny/nproc_row);
  //printf(" rank %d  x-offset test  %d ",rank,coords[0]*Nx/nproc_col);
 
  int start_i, start_j, end_i, end_j;
  start_i = 0;
  start_j = 0;
  end_i = n_local_rows;
  end_j = n_local_columns;
  if(col_rank==0){
    start_j = 1;
  }

  if(row_rank==0){
    start_i = 1;
  }

  if(col_rank==nproc_col-1){
    end_j = end_j - 1;
  }

  if(row_rank==nproc_row-1){
    end_i = end_i - 1;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  
  double x,y;
  for(int i = start_i; i < end_i; ++i) {
    for(int j = start_j; j < end_j; ++j) {

      if (rem_col == 0) {
        x = (j+col_rank*n_local_columns)*dx;
      }
      else if(col_rank<rem_col) {
        x = (j + (Nx - (nproc_col - col_rank) * (Nx/nproc_col) - rem_col+col_rank)) * dx;
      }
      else {
        // printf("\n rank %d in col else, i = %d, j =%d ", rank, i, j);
        x = (j + (Nx - (nproc_col - col_rank) * (Nx/nproc_col))) * dx;
      }
            
      if (rem_row == 0) {
        y = (i+row_rank*n_local_rows)*dx;//bör ge rätt offset men blir bara noll
        // y = ((i-1) + local_Ny * y_coord) * dx;
      } 
      else if(row_rank < rem_row) {
        // printf("\n rank %d in row else, i = %d, j =%d ", rank, i, j);
        y = (i + (Ny - (nproc_row - row_rank) * (Ny/nproc_row) -rem_row+row_rank)) * dx;
      }
      else {
        // printf("\n rank %d in row else, i = %d, j =%d ", rank, i, j);
        y = (i + (Ny - (nproc_row - row_rank) * Ny/nproc_row)) * dx;
      }

      /* u0 */
      u_local[i*n_local_columns+j] = initialize(x,y,0);
      //  printf("i=%d\n",i);
      /* u1 */
      u_new_local[i*n_local_columns+j] = initialize(x,y,dt);
      /* printf("\n rank %d x= %g  y= %g u_new= %g",rank, x ,y , initialize(x,y,dt));          */
      /* printf("\n"); */
    }
  }

#ifdef WRITE_TO_FILE
  save_solution(u_new_local,n_local_rows,n_local_columns,1);
#endif

               
#ifdef VERIFY
  double *max_error;
  max_error = malloc(sizeof(double));
  max_error[0] = 0.0;
#endif

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Type_vector(1,n_local_columns,n_local_columns,MPI_DOUBLE,&halo_row);
  MPI_Type_vector(n_local_rows,1,n_local_columns,MPI_DOUBLE,&halo_col);
  MPI_Type_commit(&halo_row);
  MPI_Type_commit(&halo_col);

  //HALO ROWS
  double halo_data_upper[n_local_columns];
  double halo_data_lower[n_local_columns];
  double halo_data_left[n_local_rows];
  double halo_data_right[n_local_rows];
  
  MPI_Cart_shift(proc_grid, 0, 1, &source, &dest1);
  if(dest1 != MPI_PROC_NULL){
    // printf("Dest 1 not null for %d sending to %d\n",rank,dest1);
    MPI_Sendrecv(&u_new_local[n_local_columns-1],1,halo_col,dest1,10,halo_data_right, n_local_rows,MPI_DOUBLE,dest1,20,proc_grid,&status[rank]); 
  }

  MPI_Cart_shift(proc_grid, 0, -1, &source, &dest2);
  if(dest2 != MPI_PROC_NULL){
    MPI_Sendrecv(&u_new_local[0],1,halo_col,dest2,20,halo_data_left, n_local_rows,MPI_DOUBLE,dest2,10,proc_grid,&status[rank]);  
  }

  MPI_Cart_shift(proc_grid, 1, 1, &source, &dest3);
  if(dest3 != MPI_PROC_NULL){
    //  printf("Dest 3 not null for %d, sending to %d \n",rank,dest3);   
       MPI_Sendrecv(&u_new_local[u_size_local-n_local_columns],1,halo_row,dest3,30,halo_data_lower, n_local_columns,MPI_DOUBLE,dest3,40,proc_grid,&status[rank]); 
  }
 
  MPI_Cart_shift(proc_grid, 1, -1, &source, &dest4);
  if(dest4 != MPI_PROC_NULL){
    //    printf("Dest 4 not null for %d, sending to %d \n",rank,dest4);    
        MPI_Sendrecv(&u_new_local[0],1,halo_row,dest4,40,halo_data_upper, n_local_columns,MPI_DOUBLE,dest4,30,proc_grid,&status[rank]);    
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
    for(int i = 1; i < (n_local_rows-1); ++i) {//Rows 
      for(int j = 1; j < (n_local_columns-1); ++j) { //Columns
        u_new_local[i*n_local_columns+j] = 2*u_local[i*n_local_columns+j]-u_old_local[i*n_local_columns+j]+lambda_sq* 
          (u_local[(i+1)*n_local_columns+j] + u_local[(i-1)*n_local_columns+j] + u_local[i*n_local_columns+j+1] + u_local[i*n_local_columns+j-1] -4*u_local[i*n_local_columns+j]);
 
      } 
    }
    
    /* Do manual computation of edges and corners */
    if(dest1!= MPI_PROC_NULL){
      //if(dest3!= MPI_PROC_NULL){ 
      //räkna med halopunkter till höger (obs ej hörnpunkt)
      for(int i = 1; i < (n_local_rows-1); ++i) { 
        u_new_local[(i+1)*n_local_columns-1] = 2*u_local[(i+1)*n_local_columns-1] - u_old_local[(i+1)*n_local_columns-1] + lambda_sq* 
          (u_local[(i+2)*n_local_columns-1] + u_local[i*n_local_columns-1] + halo_data_right[i] + u_local[(i+1)*n_local_columns-2] -4*u_local[(i+1)*n_local_columns-1]); 
      }        
    }

    if(dest3!= MPI_PROC_NULL){
      //if(dest1!= MPI_PROC_NULL){
      //räkna med halopunkter nedåt  (obs ej hörnpunkt)
      for(int j = 1; j < (n_local_columns-1); ++j) { 
        u_new_local[u_size_local- n_local_columns + j] = 2*u_local[u_size_local- n_local_columns + j] - u_old_local[u_size_local- n_local_columns + j] + lambda_sq* 
          (halo_data_lower[j] + u_local[u_size_local-2* n_local_columns + j] +  u_local[u_size_local- n_local_columns + j+1]  + u_local[u_size_local- n_local_columns + j-1] - 4*u_local[u_size_local- n_local_columns + j]); 
      }        
    }

    if(dest4!= MPI_PROC_NULL){
      //  if(dest2!= MPI_PROC_NULL){    
      //räkna med halopunkter uppåt (obs ej hörnpunkt)
      for(int j = 1; j < (n_local_columns-1); ++j) { 
        u_new_local[j] = 2*u_local[j] - u_old_local[j] + lambda_sq* 
          (halo_data_upper[j] + u_local[j + n_local_columns] + u_local[j+1] + u_local[j-1] - 4*u_local[j]); 
      }       
    }


    if(dest2!= MPI_PROC_NULL){
      //räkna med halopunkter åt vänster (obs ej hörnpunkt)
      for(int i = 1; i < (n_local_rows-1); ++i) { 
        u_new_local[i*n_local_columns] = 2*u_local[i*n_local_columns]-u_old_local[i*n_local_columns]+lambda_sq* 
          (u_local[(i-1)*n_local_columns] + u_local[(i+1)*n_local_columns] +  u_local[i*n_local_columns+1] + halo_data_left[i] - 4*u_local[i*n_local_columns]); 
      }     
    }


   
    //hörn nere till höger
    if(dest3!= MPI_PROC_NULL && dest1!= MPI_PROC_NULL){   
      u_new_local[u_size_local - 1] = 2*u_local[u_size_local - 1] - u_old_local[u_size_local - 1] + lambda_sq* 
        (u_local[u_size_local-1-n_local_columns] + halo_data_lower[n_local_columns-1] + halo_data_right[n_local_rows-1]  + u_local[u_size_local - 2] -4*u_local[u_size_local - 1]); 
    }
    else{
      u_new_local[u_size_local -1]=0;
    }

    //hör nere till vänster
    if(dest2!= MPI_PROC_NULL && dest3!= MPI_PROC_NULL){   
      //if(dest1!= MPI_PROC_NULL && dest4!= MPI_PROC_NULL){
      u_new_local[u_size_local-n_local_columns] = 2*u_local[u_size_local-n_local_columns]-u_old_local[u_size_local-n_local_columns]+lambda_sq* 
        (u_local[u_size_local-2*n_local_columns] + halo_data_lower[0] +  u_local[u_size_local-n_local_columns+1] + halo_data_left[n_local_rows-1] - 4*u_local[u_size_local-n_local_columns]); 
    }
    else{
      u_new_local[u_size_local-n_local_columns]=0;
    }

    //hör uppe till höger
    if(dest4!= MPI_PROC_NULL && dest1!= MPI_PROC_NULL){   
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

    if(col_rank == nproc_col-1 && n_local_columns == 1){
      memset(u_new_local,0,u_size_local*sizeof(double));
    }

    if(col_rank == 0 && n_local_columns == 1){
      memset(u_new_local,0,u_size_local*sizeof(double));
    }
 


    /* om raden är ett element tjock */
    if(n_local_columns == 1 && col_rank!=0 && col_rank != nproc_col-1){
      for(int i = 1; i < (n_local_rows-1); ++i) { 
        u_new_local[i] = 2*u_local[i]-u_old_local[i]+lambda_sq* 
          (u_local[(i-1)] + u_local[(i+1)] +  halo_data_right[i] + halo_data_left[i] - 4*u_local[i]); 
      }

      /*hörnet i detta specialfall */
      /*hörn uppe */
      if(dest4!=MPI_PROC_NULL){
        u_new_local[0] = 2*u_local[0]-u_old_local[0]+lambda_sq* 
          (halo_data_upper[0] + u_local[1] +  halo_data_right[0] + halo_data_left[0] - 4*u_local[0]); 
      }
      else{
        u_new_local[0] = 0;
      }  

      /*hörn nere */
      if(dest3!=MPI_PROC_NULL){
        u_new_local[n_local_rows-1] = 2*u_local[n_local_rows-1]-u_old_local[n_local_rows-1]+lambda_sq* 
          (u_local[n_local_rows-2] + halo_data_lower[0] +  halo_data_right[n_local_rows-1] + halo_data_left[n_local_rows-1] - 4*u_local[n_local_rows-1]); 
      }
      else{
        u_new_local[0] = 0;
      } 
    }

#ifdef WRITE_TO_FILE 
    save_solution(u_local,n_local_rows,n_local_columns,n); 
#endif 
    
#ifdef VERIFY 
    double error=0.0; 
    for(int i = 0; i < n_local_rows; ++i) { 
      for(int j = 0; j < n_local_columns; ++j) { 


      if (rem_col == 0) {
        x = (j+col_rank*n_local_columns)*dx;
      }
      else if(col_rank<rem_col) {
        x = (j + (Nx - (nproc_col - col_rank) * (Nx/nproc_col) - rem_col+col_rank)) * dx;
      }
      else {
        x = (j + (Nx - (nproc_col - col_rank) * (Nx/nproc_col))) * dx;
      }
            
      if (rem_row == 0) {
        y = (i+row_rank*n_local_rows)*dx;
      } 
      else if(row_rank < rem_row) {
        y = (i + (Ny - (nproc_row - row_rank) * (Ny/nproc_row) -rem_row+row_rank)) * dx;
      }
      else {
        y = (i + (Ny - (nproc_row - row_rank) * Ny/nproc_row)) * dx;
      }

      double e = fabs(u_new_local[i*n_local_columns+j]-initialize(x,y,n*dt)); 
      if(e>error) 
        error = e; 
      } 
    } 
    if(error > max_error[0]) 
      max_error[0]=error;  
#endif 
    

#ifdef VERIFY
    MPI_Reduce(&max_error[0],&err_array[n],1,MPI_DOUBLE,MPI_MAX,0,proc_grid);
#endif

    /* send halopoints */
    
    MPI_Cart_shift(proc_grid, 0, 1, &source, &dest1);
    if(dest1 != MPI_PROC_NULL){     
      MPI_Sendrecv(&u_new_local[n_local_columns-1],1,halo_col,dest1,10,halo_data_right, n_local_rows,MPI_DOUBLE,dest1,20,proc_grid,&status[rank]);
    }

    MPI_Cart_shift(proc_grid, 0, -1, &source, &dest2);    
    if(dest2 != MPI_PROC_NULL){
      MPI_Sendrecv(&u_new_local[0],1,halo_col,dest2,20,halo_data_left, n_local_rows,MPI_DOUBLE,dest2,10,proc_grid,&status[rank]);
    }

    MPI_Cart_shift(proc_grid, 1, 1, &source, &dest3);
    if(dest3 != MPI_PROC_NULL){   
      MPI_Sendrecv(&u_new_local[u_size_local-n_local_columns],1,halo_row,dest3,30,halo_data_lower, n_local_columns,MPI_DOUBLE,dest3,40,proc_grid,&status[rank]);
    }
 

    MPI_Cart_shift(proc_grid, 1, -1, &source, &dest4);   
    if(dest4 != MPI_PROC_NULL){
      MPI_Sendrecv(&u_new_local[0],1,halo_row,dest4,40,halo_data_upper, n_local_columns,MPI_DOUBLE,dest4,30,proc_grid,&status[rank]);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
  } 

  /* printf("\nrank %d U: ", rank); */
  /* for(i=0; i<n_local_rows*n_local_columns;i++) */
  /*   printf(" %g ", u_new_local[i]); */
  /* printf("\n"); */
  

  if(rank ==0){
    end=timer(); 
    printf("Time elapsed: %g s\n",(end-begin));
  } 
  
  free(u_old_local);
  free(u_new_local);
  free(u_local);

#ifdef VERIFY
  if(rank==0){
    for(int i=0;i<Nt;i++)
      printf("%g\n",err_array[i]);
     }
  free(err_array);
#endif

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

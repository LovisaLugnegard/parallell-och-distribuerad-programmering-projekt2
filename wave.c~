/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

/* #define WRITE_TO_FILE */
/* #define VERIFY */

double timer();
double initialize(double x, double y, double t);
void save_solution(double *u, int Ny, int Nx, int n);

int main(int argc, char *argv[])
{
  int Nx,Ny,Nt;
  double dt, dx, lambda_sq;
  double *u;
  double *u_old;
  double *u_new;
  double begin,end;

  Nx=128;
  if(argc>1)
    Nx=atoi(argv[1]);
  Ny=Nx;
  Nt=Nx;
  dx=1.0/(Nx-1);
  dt=0.50*dx;
  lambda_sq = (dt/dx)*(dt/dx);

  u = malloc(Nx*Ny*sizeof(double));
  u_old = malloc(Nx*Ny*sizeof(double));
  u_new = malloc(Nx*Ny*sizeof(double));

  /* Setup IC */

  memset(u,0,Nx*Ny*sizeof(double));
  memset(u_old,0,Nx*Ny*sizeof(double));
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
#ifdef VERIFY
  double max_error=0.0;
#endif

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

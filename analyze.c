/*
 * Copyright IBM Corp. 2024
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char * argv[])
{
  FILE * fp;
  int i, npts;
  double avg, tmax, tmin, val, * array;
  int numbpd, totbins, bin, * histo;
  double log10_tmin, log10_tmax, dnumbpd, botp, botn, topp, topn;
  double exp1, exp2, hmin, hmax;

  fp = fopen(argv[1], "r");
  if (fp == NULL) {
    printf("can't open the input file ... exiting\n");
    return 0;
  }
  i = 0;
  while (EOF != fscanf(fp, "%lf", &val)) i++;
  npts = i;
  printf("got npts = %d\n", npts);
  rewind(fp);

  array = (double *) malloc(npts*sizeof(double));

  avg = 0.0;
  tmin = 1.0e30;
  tmax = 0.0;
  for (i=0; i<npts; i++) {
    fscanf(fp, "%lf", &val);
    array[i] = val;
    if (val < tmin) tmin = val;
    if (val > tmax) tmax = val;
    avg += val;
  }

  avg = avg / ((double) npts);
  printf("avg = %.6lf tmin = %.6lf tmax = %.6lf\n", avg, tmin, tmax);

  // use log-scale binning
  log10_tmin = log10(tmin);
  log10_tmax = log10(tmax);

  numbpd = 10;

  dnumbpd = (double) numbpd;
  botp = floor(log10_tmin);
  botn = floor(dnumbpd*(log10_tmin - botp));
  topp = floor(log10_tmax);
  topn = ceil(dnumbpd*(log10_tmax - topp));

  // total number of histogram bins
  totbins = (int) round( (dnumbpd*topp + topn) - (dnumbpd*botp + botn) );

  histo = (int *) malloc(totbins*sizeof(int));

  for (bin = 0; bin < totbins; bin++) histo[bin] = 0;

  for (i = 0; i < npts; i++) {
    bin = (int) ( (dnumbpd*log10(array[i])) - (dnumbpd*botp + botn) );
    if ((bin >= 0) && (bin < totbins)) histo[bin]++;
  }

  // sum the histograms over all ranks
//MPI_Allreduce(MPI_IN_PLACE, histo, totbins, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  printf("histogram of times in msec for all ranks\n");
  printf(" [     min -        max ):      count\n");
  for (bin = 0; bin < totbins; bin++) {
    exp1 = botp + (botn + ((double) bin)) / dnumbpd;
    exp2 = exp1 + 1.0 / dnumbpd;
    hmin = 1.0e3*pow(10.0, exp1);
    hmax = 1.0e3*pow(10.0, exp2);
    printf("%10.3lf - %10.3lf  : %10d \n", hmin, hmax, histo[bin]);
  }
  printf("\n");

  return 0;
}

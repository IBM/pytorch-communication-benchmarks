/*
 * Copyright IBM Corp. 2024
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void sortx(double * arr , int n, int * ind, int flag);

int main(int argc, char * argv[])
{
  FILE * fp;
  int i, npts;
  double avg, tmax, tmin, median, val, * array, * sorted_times;
  int numbpd, totbins, bin, * histo, * index;
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

  sorted_times = (double *) malloc(npts*sizeof(double));

  index = (int *) malloc(npts*sizeof(int));

  avg = 0.0;
  tmin = 1.0e30;
  tmax = 0.0;
  for (i=0; i<npts; i++) {
    fscanf(fp, "%lf", &val);
    array[i] = val;
    sorted_times[i] = val;
    if (val < tmin) tmin = val;
    if (val > tmax) tmax = val;
    avg += val;
  }

  avg = avg / ((double) npts);

  sortx(sorted_times, npts, index, 1); // sort times in increasing order

  median = sorted_times[(npts - 1)/2];  // median time

  printf("avg = %.3lf median = %.3lf tmin = %.3lf tmax = %.3lf msec\n", 1.0e3*avg, 1.0e3*median, 1.0e3*tmin, 1.0e3*tmax);

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

//===========================================================================
// incremental Shell sort with increment array: inc[k] = 1 + 3*2^k + 4^(k+1)
//===========================================================================
void sortx(double * arr , int n, int * ind, int flag)
{
   int h, i, j, k, inc[20];
   int numinc, pwr2, pwr4;
   double val;

   if (n <= 1) {
      ind[0] = 0;
      return;
   }

   pwr2 = 1;
   pwr4 = 4;

   numinc = 0;
   h = 1;
   inc[numinc] = h;
   while (numinc < 20) {
      h = 1 + 3*pwr2 + pwr4;
      if (h > n) break;
      numinc++;
      inc[numinc] = h;
      pwr2 *= 2;
      pwr4 *= 4;
   }

   for (i=0; i<n; i++) ind[i] = i;

   if (flag > 0) { // sort in increasing order
      for (; numinc >= 0; numinc--) {
         h = inc[numinc];
         for (i = h; i < n; i++) {
            val = arr[i];
            k   = ind[i];

            j = i;

            while ( (j >= h) && (arr[j-h] > val) ) {
               arr[j] = arr[j-h];
               ind[j] = ind[j-h];
               j = j - h;
            }

            arr[j] = val;
            ind[j] = k;
         }
      }
   }
   else { // sort in decreasing order 
      for (; numinc >= 0; numinc--) {
         h = inc[numinc]; 
         for (i = h; i < n; i++) {
            val = arr[i];
            k   = ind[i];
   
            j = i;

            while ( (j >= h) && (arr[j-h] < val) ) {
               arr[j] = arr[j-h];
               ind[j] = ind[j-h];
               j = j - h;
            }
   
            arr[j] = val;
            ind[j] = k;
         }
      }
   }
}

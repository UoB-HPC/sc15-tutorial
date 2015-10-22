/*
**  PROGRAM: Center of mass computation.  For playing 
**                     with vectorization, SOA and AOS
*/
#ifdef APPLE
#include <stdlib.h>
#else
#include <malloc.h>
#endif
#include <stdio.h>
#include <omp.h>
#include "random.h"

#define N          100000
#define MASS_LOW   0.5
#define MASS_HI    1.5
#define COORD_LOW  0.0
#define COORD_HI   100.0
 
#define TYPE float

typedef struct{
     TYPE mass;
     TYPE x;
     TYPE y;
     TYPE z;
} part_t;

void  init(part_t *particles){
    int i;
    seed(MASS_LOW, MASS_HI);    // seed with range from 0.5 to 1.5
    for (i=0; i<N; i++){
       (particles+i)->mass = drandom();
    }
    seed(COORD_LOW, COORD_HI);  // seed for coords from 0 to 100
    for(i=0; i<N; i++){
       (particles+i)->x = drandom();
       (particles+i)->y = drandom();
       (particles+i)->z = drandom();  
   }
} 

void cm_calc(part_t *particles, part_t *cm){
   int i;
   TYPE m_i;
   cm->x =cm->y = cm->z = (TYPE) 0.0;
   cm->mass = (TYPE) 0.0;

// #pragma simd
//#pragma ivdep
#pragma simd
   for (i=0; i<N; i++){
      m_i = (particles+i)->mass;
      cm->mass += m_i;
      cm->x += m_i* (particles+i)->x;
      cm->y += m_i* (particles+i)->y;
      cm->z += m_i* (particles+i)->z;
   }
   cm->x = (cm->x)/(cm->mass); 
   cm->y = (cm->y)/(cm->mass); 
   cm->z = (cm->z)/(cm->mass); 
}

void  pretest(part_t *particles){
   int i;
   TYPE ave_mass = 0.0;
   TYPE ave_x = 0.0, ave_y = 0.0, ave_z = 0.0;

   for (i=0;i<N;i++)
      ave_mass+= (particles+i)->mass;

      ave_mass = ave_mass/N;
      printf("\n ave_mass = %f, it should be around %f \n",
           ave_mass,(MASS_LOW + (MASS_HI-MASS_LOW)/2.0));

      for (i=0; i<N; i++){
          ave_x += (particles+i)->x;
          ave_y += (particles+i)->y;
          ave_z += (particles+i)->z;
      }
      ave_x=ave_x/N;    ave_y=ave_y/N;   ave_z = ave_z/N;
      printf("\n ave_coord = %f, it should be around %f \n",
                    ave_x,(COORD_LOW + (COORD_HI-COORD_LOW)/2.0));
      printf("\n ave_coord = %f, it should be around %f \n",
                    ave_y,(COORD_LOW + (COORD_HI-COORD_LOW)/2.0));
      printf("\n ave_coord = %f, it should be around %f \n",
                    ave_z,(COORD_LOW + (COORD_HI-COORD_LOW)/2.0));
}
 
void   post_test(part_t *cm){

     printf("\n mass_tot = %f, it should be around %f \n",
                    cm->mass,N*(MASS_LOW + (MASS_HI-MASS_LOW)/2.0));
     printf("\n cm_coord x = %f, it should be around %f \n",
                    cm->x, (COORD_LOW + (COORD_HI-COORD_LOW)/2.0));
     printf("\n cm_coord y = %f, it should be around %f \n",
                    cm->y, (COORD_LOW + (COORD_HI-COORD_LOW)/2.0));
     printf("\n cm_coord z = %f, it should be around %f \n",
                    cm->z, (COORD_LOW + (COORD_HI-COORD_LOW)/2.0));
}


int  main(int argc, char **argv)
{
   part_t particles[N];	
   part_t cm;

   double start_time, run_time;
   
   init(particles);

   pretest(particles);

   start_time = omp_get_wtime();

   cm_calc(particles, &cm);

   run_time = omp_get_wtime() - start_time;
   printf(" cm calc in %f secs\n",run_time);

   post_test(&cm);

   printf("\n all done \n");
}

#!/bin/bash
for i in {3001..3080}
do
   echo "Submitting job id:  $i "
   sbatch "$i.sh"
done
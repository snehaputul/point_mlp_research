#!/bin/bash
for i in {3000..3080}
do
   echo "Submitting job id:  $i "
   sbatch "$i.sh"
done
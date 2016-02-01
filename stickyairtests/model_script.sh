#!/bin/bash
for i in  2500 5000 10000 20000 40000
do
    mpirun -np 16 python R01.py $i
done


#!/bin/bash
for i in  2500 5000 10000 20000 40000
do
    mpirun -np 16 python R02.py $i
done

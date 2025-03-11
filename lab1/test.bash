for i in {1..3}
do
    time (mpirun -np 1 lab1second 800) 2>> timesecond.txt
done
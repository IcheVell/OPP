for i in {1..3}
do
    (time OMP_NUM_THREADS=16 ./lab2second 1300) 2>> time.txt
done
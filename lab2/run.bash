for i in {1..3}
do
    (time OMP_NUM_THREADS=8 ./lab2 1300) 2>> time.txt
done
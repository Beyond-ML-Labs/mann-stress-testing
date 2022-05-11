for limit in 1000 2000 5000 10000 20000 30000 40000 50000 60000
do
    python ./experiment.py -l $limit > limit${limit}.txt
done

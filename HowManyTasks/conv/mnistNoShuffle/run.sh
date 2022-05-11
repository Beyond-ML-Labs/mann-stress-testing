for block in 2 4 6
do
    for filter in 8 16 32
    do
	python ./mnist.py -b $block -f $filter > ./${block}blocks${filter}filters.txt
    done
done

runs=(1 2 3 4 5)
for run in ${runs[@]}; do
    python3 main.py --env Stack --run_num t$run --cuda
    wait
done

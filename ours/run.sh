# runs=(1 2 3 4 5)
# for run in ${runs[@]}; do
#     python3 oat_random.py --env PickPlace --object can --num_wp 3 --run_num t$run
#     wait
# done

# runs=(3 4 5)
# for run in ${runs[@]}; do
#     python3 oat_random_door.py --env Door --num_wp 3 --run_num t$run
#     wait
# done

# python3 oat_random_eval.py --env Stack --num_wp 2 --run_num test
# wait
# python3 oat_random_eval.py --env NutAssembly --num_wp 2 --run_num test
# wait
# python3 oat_random_eval.py --env PickPlace --object bread --num_wp 2 --run_num test --n_eval 100 --render
# wait
# python3 oat_random_eval.py --env PickPlace --object milk --num_wp 2 --run_num test --n_eval 100 --render
# wait
python3 oat_random_eval.py --env PickPlace --object can --num_wp 2 --run_num test --n_eval 100 --render
# wait
# python3 oat_random_eval.py --env Door --num_wp 2 --run_num test
# wait
# python3 oat_random_eval.py --env Door --use_latch --num_wp 2 --run_num test



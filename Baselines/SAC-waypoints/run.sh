# runs=(1 2 3 4 5)
# for run in ${runs[@]}; do
#     python3 main.py --env Stack --run_num test --cuda
#     wait
# done


python3 eval.py --env Lift --num_eval 100 --run_num test --cuda
wait
python3 eval.py --env Stack --num_eval 100 --run_num test --cuda
wait
python3 eval.py --env NutAssembly --num_eval 100 --run_num test --cuda
wait
python3 eval.py --env PickPlace --object bread --num_eval 100 --run_num test --cuda
wait
python3 eval.py --env PickPlace --object milk --num_eval 100 --run_num test --cuda
wait
python3 eval.py --env PickPlace --object can --num_eval 100 --run_num test --cuda
wait
python3 eval_door.py --env Door --num_eval 100 --run_num without_latch_test --cuda
wait
python3 eval_door.py --env Door --num_eval 100 --use_latch --run_num with_latch_test --cuda
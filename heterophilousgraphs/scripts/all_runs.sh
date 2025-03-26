
python -m train  --name BUNN --dataset roman-empire --model BUNN --num_layers 8 --num_gnn 8 --device cuda:3 --num_bundles 256 --hidden_dim 512 --max_degree 8 --tau 0.4 --wandb --orth_meth euler 
python -m train  --name BUNN --dataset amazon-ratings --model BUNN --num_layers 2 --num_gnn 0 --device cuda:3 --num_bundles 256 --hidden_dim 512 --max_degree 4 --tau 1.5 --wandb --orth_meth euler 
python -m train  --name BUNN --dataset minesweeper --model BUNN --num_layers 8 --num_gnn 8 --device cuda:3 --num_bundles 256 --hidden_dim 512 --max_degree 8 --tau 1 --wandb --orth_meth euler 
python -m train  --name BUNN --dataset tolokers --model BUNN --num_layers 6 --num_gnn 7 --device cuda:0 --num_bundles 256 --hidden_dim 512 --max_degree 8 --tau 1 --wandb --orth_meth euler 
python -m train  --name BUNN --dataset questions --model BUNN --num_layers 6 --num_gnn 6 --device cuda:0 --num_bundles 128 --hidden_dim 256 --max_degree 8 --tau 1 --wandb  --orth_meth euler 

python -m train --name minesweeper-best-spec --dataset minesweeper --model SpecBuNN --bundle_dim 2 --num_layers 8 --num_gnn 8 --gnn_type SAGE --device cuda:3 --num_bundles 16 --hidden_dim 512 --tau 100 --num_runs 10 --num_steps 1000 --orth_meth euler --wandb
python -m train --name tolokers-best-spec --dataset tolokers --model SpecBuNN --bundle_dim 2 --num_layers 8 --num_gnn 8 --gnn_type SAGE --device cuda:3 --num_bundles 32 --hidden_dim 512 --tau 10 --num_runs 10 --num_steps 1000 --orth_meth euler --lr 0.0003 --lr_tau 0.01 --wandb
python -m train --name amazon-best-spec --dataset amazon-ratings --model SpecBuNN --bundle_dim 2 --num_layers 1 --num_gnn 2 --gnn_type SAGE --device cuda:3 --num_bundles 64 --hidden_dim 512 --tau 1 --num_runs 10 --num_steps 1000 --orth_meth euler --dropout 0.2 --gnn_method diff --tau_method fixed --lr 0.0003 --lr_tau 0.01 --wandb
python -m train --name roman-empire-best-spec --dataset roman-empire --model SpecBuNN-sep --bundle_dim 2 --num_layers 8 --num_gnn 6 --gnn_type SAGE --device cuda:3 --num_bundles 64 --hidden_dim 512 --tau 100 --num_runs 10 --num_steps 1000 --orth_meth euler --dropout 0.2 --gnn_method diff --tau_method fixed --lr 0.0003 --lr_tau 0.001 --wandb
python -m train --name questions-best-spec --dataset questions --model SpecBuNN-sep --bundle_dim 2 --num_layers 6 --num_gnn 4 --gnn_type SAGE --device cuda:3 --num_bundles 64 --hidden_dim 256 --tau 1 --num_runs 10 --num_steps 1000 --orth_meth euler --dropout 0.2 --gnn_method shared-recomp --tau_method fixed --lr 0.00003 --lr_tau 0.01 --wandb

python -m train  --name SHEAF --dataset roman-empire --model SHEAF --num_layers 8 --device cuda:0 --hidden_dim 512
python -m train  --name SHEAF --dataset amazon-ratings --model SHEAF --num_layers 3 --device cuda:0 --hidden_dim 512
python -m train  --name SHEAF --dataset minesweeper --model SHEAF --num_layers 8 --device cuda:0 --hidden_dim 512
python -m train  --name SHEAF --dataset tolokers --model SHEAF --num_layers 2 --device cuda:0 --hidden_dim 128
python -m train  --name SHEAF --dataset questions --model SHEAF --num_layers 2 --device cuda:0 --hidden_dim 256


current LEO+MAF model trainning command:
`$ python runner.py --data_path=$EMBEDDINGS --dataset_name=miniImageNet  --num_tr_examples_per_class=1 --outer_lr=2.739071e-4  --num_MAF_layers=3 --kl_weight=0.756143  --encoder_penalty_weight=1e-4 --dropout=0.307651  --l2_penalty_weight=3.623413e-10  --orthogonality_penalty_weight=303`

current LEO model trainning command:
`$ python runner.py --data_path=$EMBEDDINGS --dataset_name=miniImageNet  --num_tr_examples_per_class=1  --num_MAF_layers=0 `
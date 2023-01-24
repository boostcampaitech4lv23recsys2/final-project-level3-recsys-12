import sys


import pandas as pd
from torch.utils.data import DataLoader

from args import get_args
from dataloader import *
from Multi_DAE import *
from Multi_VAE import *
from utils import *


def main(args):
    if args.wandb != "NO_USE":
        import wandb

        wandb.login()
        wandb.init(project=args.wandb, entity="movie-recsys-12")
        wandb.run.name = f"autoencoder_bs:{args.batch_size}_lr:{args.lr}"
        wandb.config = vars(args)
    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    make_matrix_data_set = MakeMatrixDataSet(args=args)
    user_train, user_valid = make_matrix_data_set.get_train_valid_data()
    user_decoder = make_matrix_data_set.user_decoder
    item_decoder = make_matrix_data_set.item_decoder
    ae_dataset = AEDataSet(
        num_user=make_matrix_data_set.num_user,
    )
    data_loader = DataLoader(
        ae_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    if args.model == "MultiDAE":
        model = MultiDAE(
            p_dims=args.p_dims + [make_matrix_data_set.num_item],
            dropout_rate=args.dropout_rate,
        ).to(args.device)
        criterion = LossFunc(loss_type=args.loss_type)
    elif args.model == "MultiVAE":
        model = MultiVAE(
            p_dims=args.p_dims + [make_matrix_data_set.num_item],
            dropout_rate=args.dropout_rate,
        ).to(args.device)
        criterion = LossFunc(loss_type=args.loss_type, model_type="VAE")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    loss_list = []
    ndcg_list = []
    hit_list = []
    recall_list = []
    for epoch in range(1, args.num_epochs + 1):

        train_loss = train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=data_loader,
            make_matrix_data_set=make_matrix_data_set,
            config=args,
        )

        ndcg, hit, recall = evaluate(
            model=model,
            data_loader=data_loader,
            user_train=user_train,
            user_valid=user_valid,
            make_matrix_data_set=make_matrix_data_set,
            config=args,
        )
        if args.wandb != "NO_USE":
            wandb.log(
                {
                    "RECALL@10": recall,
                    "NDCG@10": ndcg,
                    "train_loss": train_loss,
                }
            )

        loss_list.append(train_loss)
        ndcg_list.append(ndcg)
        hit_list.append(hit)
        recall_list.append(recall)

        print(
            f"Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}| RECALL@10: {recall:.5f}"
        )

    pd.DataFrame(
        make_submission(
            model, data_loader, user_decoder, item_decoder, make_matrix_data_set, args
        ),
        columns=["user", "item"],
    ).sort_values("user").to_csv(
        args.save_path + f"/{args.model}_submission.csv", index=False
    )
    if args.wandb != "NO_USE":
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    main(args)

import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="MultiDAE",
        type=str,
        help="select model [MultiDAE, MultiVAE]",
    )
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--data_path", default="../../backend/data", type=str)
    parser.add_argument(
        "--p_dims",
        nargs="+",
        type=int,
        default=[200, 600],
        help="scheduler lr milestones",
    )
    parser.add_argument("--valid_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_anneal_steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--dropout_rate", type=float, default=0.3, help="hidden dropout p"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="hidden dropout p"
    )
    parser.add_argument(
        "--anneal_cap", type=float, default=0.2, help="hidden dropout p"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="lr")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="Multinomial",
        help="select loss type [Multinomial, Gaussian, Logistic]",
    )
    parser.add_argument(
        "--wandb", type=str, default="NO_USE", help="option for running wandb"
    )

    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--csv", type=int, default=0)
    parser.add_argument(
        "--model_path", type=str, default="../saved_model/best_model.pt"
    )
    parser.add_argument(
        "--inference_path", type=str, default="../../backend/inference/best_model.pt"
    )

    args = parser.parse_args()

    return args

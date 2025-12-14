import argparse
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed


from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./yaml/sasrec.yaml')
    parser.add_argument("--dataset", default="SASRec_dataset", type=str)

    # train args
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    config = Config(model='SASRec',
                    config_file_list=[args.config_file],
                    dataset=args.dataset,
                    )
    print('Config loaded')
    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)

    # model loading and initialization
    model = SASRec(config, train_data.dataset).to(config['device'])
    print("model information : ",model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])

if __name__ == "__main__":
    main()

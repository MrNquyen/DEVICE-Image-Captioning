from utils.flags import Flags
from utils.configs import Config
from utils.utils import load_yml
from projects.models.device_model import DEVICE
from utils.trainer import Trainer


if __name__=="__main__":
    # -- Get Parser
    flag = Flags()
    args = flag.get_parser()

    # -- Get device
    # print(args)
    device = args.device

    # -- Get config
    config = load_yml(args.config)
    config_container = Config(config)

    # -- Trainer
    trainer = Trainer(config, args)
    if args.run_type=="train":
        trainer.train()
    elif args.run_type=="inference":
        trainer.test()
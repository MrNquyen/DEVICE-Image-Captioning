from utils.flags import Flags
from utils.configs import Config
from utils.utils import load_yml
from projects.models.device_model import DEVICE 



if __name__=="__main__":
    # -- Get Parser
    flag = Flags()
    args = flag.get_parser()

    # -- Get device
    device = args.device

    # -- Get config
    config = load_yml(args.config)
    config_container = Config(config)

    # -- Dataset
    

    # -- Model
    model = DEVICE(
        config=config_container.config_model,
        device=device 
    )

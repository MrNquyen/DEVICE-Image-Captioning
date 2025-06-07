class Config():
    def __init__(self, config):
        self.config = config
        self.build_config()

    def build_config(self):
        self.config_dataset = self.config["dataset_attributes"]
        self.config_model = self.config["model_attributes"]
        self.config_optimizer = self.config["optimizer_attributes"]
        self.config_training = self.config["training_parameters"]
    
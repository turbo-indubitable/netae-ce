from pyfiglet import Figlet
import time

from networkae.pipeline.pipeline import PipelineRunner


if __name__ == "__main__":

    f = Figlet(font='doom')
    print(f.renderText('NETWORKAE'))
    time.sleep(3)  # dramatic pause so the logo lingers

    runner = PipelineRunner(
        csv_path="/home/phaze/PycharmProjects/NetworkModel/Code/data/testdata.csv",
        config_path="/home/phaze/PycharmProjects/NetworkModel/Code/networkae/config/config.yaml",
        persist_models=True
    )
    runner.run()
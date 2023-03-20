from shapemaker import * 

class Register():

    def __init__(self, file):
        __file_name__ = os.path.splitext(file)[0]

        # id for this particular experiment
        mid = f"{__file_name__}_{datetime.now():%Y%m%d_%H%M%S}"
        os.mkdir(f"logs/{mid}")

        # save a coopy of this script
        shutil.copy(__file__, f"logs/{mid}/_script.py")

        # region Logging
        self.logger = logging.getLogger('new_logger')
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        fh = logging.FileHandler(f"logs/{mid}/_output.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.logger.info(f"ModelID: {mid}")
        self.start_time = datetime.now()
        # endregion

    def logging(self, i, total_loss, stepsize):
        self.logger.info(f"Iter {i}: "
                f"Loss = {total_loss.detach().cpu().numpy() }, "
                f"stepsize = {stepsize}, "
                f"Time = {datetime.now() - self.start_time}")

    def finished(self):
        self.logger.info("Finished")
        print("Finished!")




from qrfitter3.rnodes.qrfitter3_node import *
from qrfitter3.reports.utils import rpt_gen
from qrfitter3.utils import read_json


RNode.RNODE_ROOT = "/data/bees1/safe/ruud/equity_fitting/"  # change to your own directory
RNode.GLOBAL_TASKNAME = "expt_res"

qf_config = read_json('test/roll_xgb_fitting_cfg.json')
qf3 = QRFitter3Node(qf_config, label=qf_config['fitting_name'], clean_dir_name=True)
qf3.gen_tasks()
qf3.run()
rpt_gen(os.path.join(RNode.RNODE_ROOT, RNode.GLOBAL_TASKNAME, qf_config['fitting_name']), os.path.join(RNode.RNODE_ROOT, RNode.GLOBAL_TASKNAME, qf_config['fitting_name'], 'report.html'))


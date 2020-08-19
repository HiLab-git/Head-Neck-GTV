# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
from pymic.util.parse_config import parse_config
from pymic.net_run.net_run_agent import  NetRunAgent
from pymic.net.net_dict import NetDict
from network.baseunet2d5_att_pe import Baseunet2d5_att_pe 

my_net_dict = NetDict
my_net_dict['Baseunet2d5_att_pe'] = Baseunet2d5_att_pe

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python train_infer.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)

    # use custormized CNN and loss function
    agent  = NetRunAgent(config, stage)
    agent.set_network_dict(my_net_dict)
    agent.run()

if __name__ == "__main__":
    main()

from __future__ import print_function
from platoon.channel import Controller
import numpy as np

class LMController(Controller):

    def __init__(self, control_port):
        Controller.__init__(self, control_port)
        self.wps = {}

    def handle_control(self, req, worker_id):
        control_response = ""

        if req == 'done':
            self.worker_is_done(worker_id)
            print("Worker {}: Done training.".format(worker_id))
        elif 'wps' in req:
            print("Worker {}: {} wps for epoch {}.".format(worker_id, req['wps'], req['epoch']))
            wps = self.wps.get(req['epoch'], [])
            wps += [req['wps']]
            self.wps[req['epoch']] = wps
	elif 'type' in req:
	    print(req)

        return control_response

if __name__ == '__main__':
    l = LMController(control_port=5567)
    print("Controller is ready.")
    l.serve()

    total = []
    print("\n## RESULTS ##")
    for e in l.wps:
	total += [np.sum(l.wps[e])]

    print("Mean wps: {}, STD: {}".format(np.mean(total), np.std(total)))

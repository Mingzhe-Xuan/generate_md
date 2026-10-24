"""Read and validate the optimization budget used in experiment artifacts."""
import json


def steps_for(arm):
    protocol = json.loads((arm/'protocol.json').read_text())
    steps = protocol['steps_per_A']
    if protocol['cycles'] != 1 or type(steps) is not int or steps not in (200, 600):
        raise ValueError('Require a single completed A200 or A600 phase.')
    return steps


def full_steps(root):
    steps = {steps_for(root/strategy) for strategy in ('neutral', 'unmasked')}
    if len(steps) != 1:
        raise ValueError('Initialization strategies must use the same A budget.')
    return steps.pop()

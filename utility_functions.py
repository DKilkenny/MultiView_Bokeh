import numpy as np

def stack_outputs(outputs_dict):
    output_lists_to_stack = []
    for values in outputs_dict.values():
        output_lists_to_stack.append(np.asarray(values))

    return np.stack(output_lists_to_stack, axis=-1)
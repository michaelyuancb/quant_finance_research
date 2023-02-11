import torch.nn

from quant_finance_research.tools.util import *
from quant_finance_research.tools.nn import *
from quant_finance_research.model.base_dnn import *


class DebugNeuralNetworkWrapper:

    def debug_cuda(self):
        x = np.random.randn(64, 4)
        y = np.random.randn(64, 1)
        dnn = Base_DNN(input_dim=4, hidden_dim=2, dropout_rate=0)
        optimizer = torch.optim.Adam(dnn.parameters(), lr=1e-3)
        wrapper = NeuralNetworkWrapper(dnn, optimizer, device='cpu')
        loss_func = torch.nn.MSELoss()
        val = wrapper.train(x, y, x, y, loss_func,
                            seed=0,
                            early_stop=1,
                            max_epoch=2,
                            epoch_print=1,
                            num_workers=0,
                            batch_size=32)
        print(val)
        print(f"type(val)={type(val)}")

        param = wrapper.get_best_param()
        # print(param)

        wrapper.load_param(param)
        # print(wrapper.nn.state_dict())
        print("success")
        v = wrapper.predict(x, batch_size=32)
        print(v)
        print(type(v))


if __name__ == "__main__":
    debugger = DebugNeuralNetworkWrapper()
    debugger.debug_cuda()

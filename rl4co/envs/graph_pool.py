import torch
class variable_pool:
    def __init__(self, data_number, var_node_size, var_shape, var_scale) -> None:
        self.data_number = data_number
        self.var_node_size = var_node_size
        self.var_shape = var_shape
        self.min_scale, self.max_scale = var_scale
        
        
    def generate_variable(self):
        self.datas = (
                torch.FloatTensor(self.data_number, self.var_node_size, self.var_shape)
                .uniform_(self.min_scale, self.max_scale)
                
            )
        return self.datas
        
# vp = variable_pool(1, 20, 2, [0, 1])
# vp.generate_variable()

# print(vp.datas[0])

class svrp_graph_pool:
    def __init__(self, num_loc) -> None:
        self.locs_scale = [0, 1]
        self.demand_scale = [1, 10]
        
        # locs包括depot， 所以+1
        self.locs_pool = variable_pool(1, num_loc + 1, 2, self.locs_scale)
        self.demand_pool = variable_pool(1, num_loc, 1, self.demand_scale)
    
    def generate_datas(self):
        #
        self.locs_data = self.locs_pool.generate_variable()
        self.demand_data = self.demand_pool.generate_variable()
        self.demand_data = self.demand_data.squeeze(-1)
        
# sgp = svrp_graph_pool(20)
# sgp.generate_datas()
# print(sgp.demand_data.shape)